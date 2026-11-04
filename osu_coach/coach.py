"""Ties the skill model, the weakness scan, the map picks and the planner together."""
import json

import pandas as pd

from . import charts, diagnose, mapfinder, model, planner, recommend
from .api import Client
from .constants import ERA_NEW


def analyze(ds):
    rall_new = ds.rall[ds.rall["era"] == ERA_NEW] if not ds.rall.empty else ds.rall
    m = model.build(rall_new, ds.att)
    return {"model": m, "scan": diagnose.scan(ds, m["predictor"])}


def _fmt_ci(ci):
    return "n/a" if not ci or ci[0] is None else f"{ci[0]:.2f}-{ci[1]:.2f}"


def skill_markdown(res, ds):
    m = res["model"]
    md = ["## Skill estimate", ""]
    c = m["ceilings"]
    if c:
        aim = "n/a" if c["aim"] is None else f"{c['aim']:.2f}"
        speed = "n/a" if c["speed"] is None else f"{c['speed']:.2f}"
        md += ["| ceiling | value | 95% CI | meaning |", "|---|---|---|---|",
               f"| aim | {aim} | {_fmt_ci(c['aim_ci'])} | you average {model.MISS_LINE:.0f} misses per 100 objects here |",
               f"| speed | {speed} | {_fmt_ci(c['speed_ci'])} | you average {model.OK_LINE:.0f} 100s/50s per 100 objects here |",
               "", f"Moving up along your usual map balance (speed/aim {c['ratio']:.2f}), you hit the **{c['limiter']}** limit "
               f"first ({100 * c['limiter_conf']:.0f}% of bootstrap samples). Based on your last {c['n']} attempts.", ""]
    else:
        md += ["Not enough logged attempts yet (needs 40+ with 15+ maps). Run this after each session; the API only keeps 24 hours.", ""]
    s = m["sessions"]
    if len(s):
        md += ["Sessions at equal difficulty (so playing harder maps doesn't look like getting worse):", "",
               "| session | attempts | misses/100 | 100s/100 | acc | finished |", "|---|---|---|---|---|---|"]
        for _, r in s.tail(8).iterrows():
            md.append(f"| {r['date']:%Y-%m-%d %H:%M} | {int(r['n'])} | {r['adj_miss_pct']:.2f} | {r['adj_ok_pct']:.2f} | "
                      f"{r['adj_acc']:.1f} | {100 * r['finish']:.0f}% |")
        md.append("")
        tr = {k: model.trend(s, k) for k in ["adj_miss_pct", "adj_ok_pct"]}
        if all(v is None for v in tr.values()):
            md += ["Trend: needs 5+ sessions over 3+ days before it means anything.", ""]
        else:
            for k, lab in [("adj_miss_pct", "misses per 100 objects"), ("adj_ok_pct", "100s per 100 objects")]:
                v = tr[k]
                if v:
                    md.append(f"Trend, {lab}: {v['per_week']:+.2f} per week (+/- {2 * v['se']:.2f}), "
                              + ("clear." if v["clear"] else "not distinguishable from flat."))
            md.append("")
        md += ["![](8_sessions.png)", ""]
    return "\n".join(md)


def model_markdown(res):
    m = res["model"]
    md = ["## Can the model be trusted?", ""]
    t = m["table"]
    if not t:
        md += [f"Not tested yet: {m['n']} attempts logged, {model.MIN_ATTEMPTS}+ needed.", ""]
    else:
        md += [f"Finish chance, tested on {m['tested']} later attempts on maps the model hadn't seen. Lower log loss is "
               "better; gain is against the base rate, with a 95% interval from resampling maps.", "",
               "| model | log loss | gain | 95% interval |", "|---|---|---|---|"]
        for k, v in t.items():
            md.append(f"| {k} | {v['log_loss']:.3f} | {v['gain']:+.3f} | {v['gain_lo']:+.3f} to {v['gain_hi']:+.3f} |")
        verdict = (f"The {m['chosen']} model beats the base rate, so plans use it." if m["trusted"] else
                   "No model clearly beats the base rate yet, so finish chances are a rough guess.")
        md += ["", verdict + " 'Finish' means the attempt reached the end; on clients that submit restarts, "
               "a restart counts as not finishing.", ""]
        if m["calibration"]:
            md += ["| predicted finish | actual | attempts |", "|---|---|---|"]
            md += [f"| {100 * c['predicted']:.0f}% | {100 * c['actual']:.0f}% | {c['n']} |" for c in m["calibration"]] + [""]
    if m["errors"]:
        md += ["Held-out maps, mean absolute error against just guessing your average:", "",
               "| predicting | model | guessing average | used |", "|---|---|---|---|"]
        names = {"acc": "accuracy", "miss_pct": "misses per 100", "ok_pct": "100s per 100"}
        md += [f"| {names[k]} | {v['mae']:.2f} | {v['mae_base']:.2f} | {'yes' if v['useful'] else 'no'} |"
               for k, v in m["errors"].items()] + [""]
    return "\n".join(md)


def report_section(res, ds, out):
    charts.chart_sessions(res["model"]["sessions"], out)
    charts.chart_findings(res["scan"], out)
    parts = [skill_markdown(res, ds), diagnose.to_markdown(res["scan"])]
    if res["scan"]["findings"]:
        parts.append("![](9_weaknesses.png)\n")
    parts.append(model_markdown(res))
    return "\n".join(parts)


def metrics(res, M):
    m = res["model"]
    c = m["ceilings"] or {}
    M.update({"ceil_aim": c.get("aim"), "ceil_speed": c.get("speed"), "limiter": c.get("limiter"),
              "model_trusted": m["trusted"], "model_gain": (m["table"] or {}).get(m["chosen"], {}).get("gain"),
              "n_weaknesses": len(res["scan"]["findings"])})
    s = m["sessions"]
    if len(s):
        M.update({"adj_miss_last": float(s["adj_miss_pct"].iloc[-1]), "adj_ok_last": float(s["adj_ok_pct"].iloc[-1])})


def list_markdown(lists, method):
    md = ["# Practice maps", "", f"Picked by predicted accuracy ({method}), skill focus, favourites and variety. "
          "Maps from your top plays are left out.", ""]
    for role, maps in lists.items():
        md += [f"## {recommend.ROLE_TITLES[role]}", ""]
        if not maps:
            md += ["Nothing fit. Add sources in config.toml or run `osu-coach maps --refresh`.", ""]
            continue
        md += ["| map | mods | stars | AR | BPM | length | aim | speed | predicted acc | source | link |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in maps:
            md.append(f"| {m['artist']} - {m['title']} [{m['version']}] | {m['mods']} | {m['stars']:.2f} | {m['eff_ar']:.1f} | "
                      f"{m['eff_bpm']:g} | {planner.mmss(m['length_s'])} | {m['aim']:.2f} | {m['speed']:.2f} | {m['acc_pred']:.0f}% | "
                      f"{', '.join(m['sources'])} | https://osu.ppy.sh/beatmaps/{int(m['id'])} |")
        md.append("")
    return "\n".join(md)


def make_plan(cfg, ds, res, offline=False, refresh=False):
    """Pool -> candidates -> lists -> plan. Writes plan.md, practice_maps.md/json and the plan history."""
    m, scan = res["model"], res["scan"]
    state = planner.load_state(cfg)
    rev = planner.review(planner.latest_plan(cfg), ds.rall, m["sessions"])
    state = planner.adjust_challenge(state, rev)
    client = Client(cfg.data / "attr_cache.json", root=cfg.root, offline=offline, mode=cfg.mode)
    window = mapfinder.star_window(ds)
    pool, hits = mapfinder.build_pool(cfg, client, window, refresh)
    exclude = set(ds.top["beatmap_id"].dropna().astype(int)) if len(ds.top) else set()
    cands = mapfinder.candidates(cfg, client, pool, window, exclude)
    if cands.empty:
        print("No candidate maps. Check the sources in config.toml.")
        return None
    c, method = recommend.annotate(cands, m["predictor"], ds)
    lists = recommend.build_lists(c, cfg, m["ceilings"], state["challenge"])
    revisit = recommend.revisit_queue(ds.rall, m["predictor"])
    plan = planner.build_plan(cfg, lists, revisit, scan, m["ceilings"], m["sessions"], state["challenge"])
    planner.save_plan(cfg, plan)
    planner.save_state(cfg, state)
    (cfg.out / "plan.md").write_text(planner.plan_markdown(plan, planner.review_markdown(rev, state), method, m["trusted"]),
                                     encoding="utf-8")
    (cfg.out / "practice_maps.md").write_text(list_markdown(lists, method), encoding="utf-8")
    sets = sorted({int(x["set"]) for v in lists.values() for x in v} | {int(x["set"]) for x in revisit if pd.notna(x["set"])})
    (cfg.out / "practice_maps.json").write_text(json.dumps({"sets": sets}), encoding="utf-8")
    charts.chart_targets(c, lists, m["ceilings"], ds.att, cfg.out)
    return plan
