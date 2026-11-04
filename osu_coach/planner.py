"""
Session plans. Weaknesses decide what gets the time, predictions decide which maps, and every plan is saved
so the next run can check it: did you play it, did the predictions hold, did the goals move? Plans that ran
easier than predicted make the next one harder, and the other way around.
"""
import json

import numpy as np
import pandas as pd

OVERHEAD_S = 20     # song select, retries, breathing
CHALLENGE_STEP = 1.0
CHALLENGE_MAX = 6.0
MIN_PLAYS_TO_ADJUST = 6
BIAS_TO_ADJUST = 2.0    # accuracy points between prediction and result
GOAL_STEP = 0.10

AREA_ROLE = {"aim": "focus_aim", "speed": "focus_speed"}
GOAL_METRICS = {"aim": ("adj_miss_pct", "misses per 100 objects at equal difficulty", -1),
                "speed": ("adj_ok_pct", "100s/50s per 100 objects at equal difficulty", -1)}


def load_state(cfg):
    p = cfg.out / "coach_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"challenge": 0.0, "reviewed": []}


def save_state(cfg, state):
    (cfg.out / "coach_state.json").write_text(json.dumps(state, indent=1), encoding="utf-8")


def fill_block(maps, minutes):
    """Take maps in order until the block's time is used. Short blocks still get one map."""
    budget, out = minutes * 60, []
    for m in maps:
        cost = float(m.get("length_s") or 90) + OVERHEAD_S
        if out and budget < cost * 0.5:
            break
        out.append(dict(m, plays=1))
        budget -= cost
    i = 0
    while out and budget > 0 and i < 20:
        m = out[i % len(out)]
        cost = float(m.get("length_s") or 90) + OVERHEAD_S
        if budget < cost * 0.5:
            break
        m["plays"] += 1
        budget -= cost
        i += 1
    return out


def choose_focus(scan, ceilings):
    """The top two distinct skill areas from the scan, falling back to whichever ceiling you reach first."""
    areas = []
    for f in scan["findings"]:
        if f.area in AREA_ROLE and f.area not in [a for a, _ in areas]:
            areas.append((f.area, f))
    if not areas and ceilings:
        areas.append((ceilings["limiter"], None))
    return areas[:2]


def split_minutes(total, n_focus, has_revisit, overreach):
    """Share of the session per block. If most attempts don't finish, stretch time goes back to focus work."""
    warm, cool = 0.15, 0.10
    extra = 0.15 if has_revisit else 0.0
    if not has_revisit and not overreach:
        extra = 0.10  # stretch block instead
    focus = 1 - warm - cool - extra
    parts = [0.65, 0.35] if n_focus == 2 else [1.0]
    return {"warmup": warm * total, "focus": [focus * p * total for p in parts], "extra": extra * total, "cool": cool * total}


def build_plan(cfg, lists, revisit, scan, ceilings, sessions, challenge, now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    focus = choose_focus(scan, ceilings)
    habits = {h.key: h for h in scan["habits"]}
    overreach = "overreach" in habits
    mins = split_minutes(cfg.session_minutes, max(1, len(focus)), bool(revisit), overreach)
    warm = list(lists.get("warmup") or [])
    blocks = [{"name": "Warm-up", "minutes": mins["warmup"], "why": "Get your hands going, aim for clean plays.",
               "maps": fill_block(warm[:3], mins["warmup"])}]
    for (area, f), m in zip(focus, mins["focus"]):
        why = f.text if f else f"You reach your {area} limit first as maps get harder."
        blocks.append({"name": f"Focus: {f.drill if f else area}", "minutes": m, "why": why, "how": f.how if f else "",
                       "maps": fill_block(lists.get(AREA_ROLE[area]) or [], m)})
    in_plan = {m["set"] for b in blocks for m in b["maps"]}
    revisit = [m for m in revisit if m["set"] not in in_plan]
    if revisit:
        blocks.append({"name": "Revisit", "minutes": mins["extra"], "why": "Maps you got far on or finished with weak accuracy.",
                       "maps": fill_block(revisit, mins["extra"])})
    elif mins["extra"] and lists.get("stretch"):
        blocks.append({"name": "Stretch", "minutes": mins["extra"], "why": "One step up, just to see it.",
                       "maps": fill_block(lists["stretch"], mins["extra"])})
    blocks.append({"name": "Cool-down", "minutes": mins["cool"], "why": "End on something you can play well.",
                   "maps": fill_block(warm[3:] or warm, mins["cool"])})

    goals = []
    last = sessions.tail(2) if sessions is not None and len(sessions) else None
    for area, _ in focus:
        if area in GOAL_METRICS and last is not None and len(last):
            col, label, sign = GOAL_METRICS[area]
            base = float(last[col].mean())
            goals.append({"metric": col, "label": label, "baseline": base, "target": base * (1 + sign * GOAL_STEP), "sign": sign})
    if overreach:
        fin = float(habits["overreach"].extra.get("finish", 0))
        goals.append({"metric": "finish", "label": "share of attempts finished", "baseline": fin,
                      "target": min(fin + 0.15, 1.0), "sign": 1})
    return {"id": now.strftime("%Y-%m-%d_%H%M%S"), "created": now.isoformat(), "minutes": cfg.session_minutes,
            "challenge": challenge, "blocks": blocks, "goals": goals, "habits": [h.how for h in scan["habits"]],
            "week": week_schedule(scan, ceilings, cfg.sessions_per_week)}


def week_schedule(scan, ceilings, n):
    """Spread sessions over focus areas in proportion to how much each one costs you."""
    cost = {}
    for f in scan["findings"]:
        if f.area in AREA_ROLE:
            cost[f.drill] = cost.get(f.drill, 0.0) + max(f.cost_pp, 0.1)
    if not cost:
        return [ceilings["limiter"] if ceilings else "mixed"] * n
    top = sorted(cost.items(), key=lambda kv: -kv[1])[:3]
    tot = sum(v for _, v in top)
    counts = {k: int(np.floor(n * v / tot)) for k, v in top}
    rest = sorted(top, key=lambda kv: -(n * kv[1] / tot - np.floor(n * kv[1] / tot)))
    for k, _ in rest[: n - sum(counts.values())]:
        counts[k] += 1
    out, left = [], dict(counts)
    while len(out) < n:  # interleave, biggest first, so the same focus isn't back to back
        for k, _ in top:
            if left[k] > 0 and len(out) < n:
                out.append(k)
                left[k] -= 1
    return out


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def save_plan(cfg, plan):
    d = cfg.out / "plans"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{plan['id']}.json").write_text(json.dumps(plan, indent=1, default=_json_safe), encoding="utf-8")


def latest_plan(cfg):
    d = cfg.out / "plans"
    files = sorted(d.glob("*.json")) if d.exists() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def review(plan, rall, sessions):
    """Compare a saved plan with what happened after it was made. Returns None when nothing has been played since."""
    if not plan or rall is None or rall.empty:
        return None
    created = pd.Timestamp(plan["created"])
    new = rall[rall["date"] > created]
    if new.empty:
        return None
    planned = {}
    for b in plan["blocks"]:
        for m in b["maps"]:
            planned[int(m["id"])] = m
    hit = new[new["beatmap_id"].isin(planned)]
    out = {"plan": plan["id"], "attempts": len(new), "on_plan": len(hit), "adherence": len(hit) / len(new),
           "maps_played": int(hit["beatmap_id"].nunique()), "maps_planned": len(planned), "bias": None, "goals": []}
    if len(hit):
        pred = hit["beatmap_id"].map(lambda i: planned[int(i)].get("acc_pred"))
        ok = pred.notna() & hit["passed"]
        if ok.sum():
            out["bias"] = float((hit.loc[ok, "acc"] - pred[ok].astype(float)).mean())
            out["bias_n"] = int(ok.sum())
        out["finish_rate"] = float(hit["passed"].mean())
    if sessions is not None and len(sessions):
        before, after = sessions[sessions["date"] < created].tail(2), sessions[sessions["date"] >= created]
        for g in plan.get("goals", []):
            if g["metric"] == "finish":
                now = float(new["passed"].mean())
                was = g["baseline"]
            elif len(before) and len(after) and g["metric"] in sessions:
                now, was = float(after[g["metric"]].mean()), float(before[g["metric"]].mean())
            else:
                continue
            want = g["target"] if g["metric"] == "finish" else was * (1 + g["sign"] * GOAL_STEP)
            met = now >= want if g["sign"] > 0 else now <= want
            out["goals"].append({"label": g["label"], "was": was, "now": now, "want": want, "met": bool(met)})
    return out


def adjust_challenge(state, rev):
    """Closed loop: played well above prediction -> harder next time, well below -> easier. Each plan counts once."""
    if not rev or rev["plan"] in state["reviewed"]:
        return state
    if rev.get("bias") is not None and rev.get("bias_n", 0) >= MIN_PLAYS_TO_ADJUST:
        if rev["bias"] >= BIAS_TO_ADJUST:
            state["challenge"] = min(state["challenge"] + CHALLENGE_STEP, CHALLENGE_MAX)
        elif rev["bias"] <= -BIAS_TO_ADJUST:
            state["challenge"] = max(state["challenge"] - CHALLENGE_STEP, -CHALLENGE_MAX)
        state["reviewed"].append(rev["plan"])
    return state


def review_markdown(rev, state):
    if not rev:
        return ""
    md = [f"## How the last plan went ({rev['plan']})", "",
          f"{rev['attempts']} attempts since then, {rev['on_plan']} on planned maps ({100 * rev['adherence']:.0f}%), "
          f"{rev['maps_played']} of {rev['maps_planned']} maps played."]
    if rev.get("bias") is not None:
        side = "above" if rev["bias"] > 0 else "below"
        md.append(f"Finished plays on those maps came in {abs(rev['bias']):.1f} pp {side} the prediction (n={rev['bias_n']}). "
                  f"Difficulty adjustment is now {state['challenge']:+.0f} pp.")
    if rev["goals"]:
        md += ["", "| goal | before | now | needed | met |", "|---|---|---|---|---|"]
        for g in rev["goals"]:
            md.append(f"| {g['label']} | {g['was']:.2f} | {g['now']:.2f} | {g['want']:.2f} | {'yes' if g['met'] else 'no'} |")
    md.append("")
    return "\n".join(md)


def map_line(m):
    name = f"{m.get('artist')} - {m.get('title')} [{m.get('version')}]"
    mods = "" if m.get("mods") in (None, "NM") else f" +{m['mods']}"
    plays = f"{int(m.get('plays', 1))}x"
    acc = m.get("acc_pred")
    pf = m.get("p_finish")
    acc_s = f"{acc:.0f}%" if acc is not None and np.isfinite(acc) else "-"
    pf_s = f"{100 * pf:.0f}%" if pf is not None and np.isfinite(pf) else "-"
    return (f"| {plays} | {name}{mods} | {m.get('stars', 0):.2f} | {m.get('aim', 0):.2f} | {m.get('speed', 0):.2f} | "
            f"{mmss(m.get('length_s') or 0)} | {acc_s} | {pf_s} | https://osu.ppy.sh/beatmaps/{int(m['id'])} |")


def mmss(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def plan_markdown(plan, rev_md, method, trusted):
    md = ["# Next session", "", f"{plan['minutes']} minutes. Predicted accuracy comes from the {method}.", ""]
    if trusted:
        md += ["Finish chance comes from your own history, restarts included, so it says as much about how often you restart "
               "as about the map. Use predicted accuracy to judge difficulty.", ""]
    else:
        md += ["The finish-chance model didn't beat your plain base rate on unseen maps yet, so treat that column as a rough guess.", ""]
    if rev_md:
        md.append(rev_md)
    for b in plan["blocks"]:
        md += [f"## {b['name']} ({b['minutes']:.0f} min)", "", b["why"], ""]
        if b.get("how"):
            md += [b["how"], ""]
        if b["maps"]:
            md += ["| plays | map | stars | aim | speed | length | predicted acc | finish chance | link |",
                   "|---|---|---|---|---|---|---|---|---|"] + [map_line(m) for m in b["maps"]] + [""]
        else:
            md += ["No map fit this block. Try more sources in config.toml, or `osu-coach maps --refresh`.", ""]
    if plan["goals"]:
        md += ["## Goals until the next review", "", "| goal | now | target |", "|---|---|---|"]
        md += [f"| {g['label']} | {g['baseline']:.2f} | {g['target']:.2f} |" for g in plan["goals"]] + [""]
    if plan["habits"]:
        md += ["## Habits", ""] + [f"- {h}" for h in plan["habits"]] + [""]
    if plan["week"]:
        md += ["## This week", "", " -> ".join(plan["week"]), ""]
    return "\n".join(md)
