"""The analysis run: turns a Dataset into report.md, charts and metrics.json."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from . import charts, coach
from .constants import COMFORT, ERA_NEW, ERA_OLD, MIN_N, PUSH
from .history import compare, previous_metrics
from .stats import binned, f, hinge, joint, rng, simple, zones


def analyze(cfg, ds):
    """Write report.md, charts and metrics.json. Returns (metrics, previous run's metrics, coach results)."""
    out, hist = cfg.out, cfg.hist
    exp, top, recent_now, rall = ds.export, ds.top, ds.recent_now, ds.rall
    cur, cur_label, att = ds.cur, ds.cur_label, ds.att
    user = exp.get("user") or {}
    st = user.get("statistics") or {}
    era_labels = {ERA_OLD: ds.old_label, ERA_NEW: ds.since}

    M = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "exported_at": exp.get("exported_at"), "username": user.get("username"),
         "era_since": ds.since, "era_old": ds.old_label,
         "pp": st.get("pp"), "global_rank": st.get("global_rank"), "hit_accuracy": st.get("hit_accuracy"),
         "play_count": st.get("play_count"), "n_top": len(top), "n_recent_now": len(recent_now),
         "n_recent_all": len(rall), "n_cur_passes": len(cur)}
    R = []  # report sections

    # 1. hypothesis -------------------------------------------------------
    fits = {k: simple(cur, k, "acc") for k in ["aim", "speed", "ratio", "stars"]}
    mfits = {k: simple(cur, k, "miss_pct") for k in ["aim", "speed", "ratio"]}
    ofits = {k: simple(cur, k, "ok_pct") for k in ["aim", "speed"]}
    j = joint(cur, "acc", ["aim", "speed"])
    jm = joint(cur, "miss_pct", ["aim", "speed"])
    jo = joint(cur, "ok_pct", ["aim", "speed"])
    jr = joint(cur, "acc", ["stars", "ratio"])
    ja = joint(att, "acc", ["aim", "speed"]) if len(att) else None
    jam = joint(att, "miss_pct", ["aim", "speed"]) if len(att) else None
    charts.chart_hypothesis(cur, out)

    def line(name, d, unit="acc %"):
        if not d:
            return f"| {name} | n/a | | | |"
        return f"| {name} | {d['slope']:+.2f} {unit} per 1.0 | {d['r']:+.2f} | {d['p']:.3g} | {d['n']} |"

    def hurts(model, k, sign):
        """Significant (p<0.05) effect of k in the direction that means 'harder -> worse'."""
        if not model:
            return False
        c = model["coef"][k]
        return c["p"] < 0.05 and np.sign(c["b"]) == sign

    def jrow(name, model, k, unit):
        if not model:
            return f"| {name} | n/a | | | |"
        c = model["coef"][k]
        return f"| {name} | {k} | {c['b']:+.2f} | {c['per_sd']:+.2f} {unit} | {c['p']:.3g} | {model['n']} |"

    speed_acc, aim_acc = hurts(j, "speed", -1), hurts(j, "aim", -1)
    speed_ok = hurts(jo, "speed", 1)
    aim_miss = hurts(jm, "aim", 1)
    speed_weak = speed_acc or speed_ok
    aim_bad = aim_acc or aim_miss
    if not j:
        verdict = "insufficient data"
    elif speed_weak and not aim_bad:
        verdict = "speed is the limit: speed difficulty costs you accuracy, aim difficulty doesn't"
    elif speed_weak and aim_bad:
        verdict = ("both matter: speed costs accuracy (100s) and "
                   + ("misses rise with aim difficulty" if aim_miss else "accuracy also falls with aim"))
    elif aim_bad:
        verdict = "aim is the limit: aim difficulty, not speed, drives your errors"
    else:
        verdict = "unclear: neither aim nor speed separately predicts accuracy or misses in this sample"
    M["verdict"] = verdict
    if j:
        ca, cs = j["coef"]["aim"], j["coef"]["speed"]
        M.update({"acc_per_sd_aim": ca["per_sd"], "acc_per_sd_speed": cs["per_sd"], "p_aim": ca["p"], "p_speed": cs["p"]})
    if jm:
        M.update({"miss_per_sd_aim": jm["coef"]["aim"]["per_sd"], "miss_per_sd_speed": jm["coef"]["speed"]["per_sd"]})
    for name, model in [("acc", j), ("miss", jm), ("ok", jo), ("att_acc", ja), ("att_miss", jam)]:
        if model:
            M[f"{name}_n"] = model["n"]
            for k in ["aim", "speed"]:
                M[f"{name}_sd_{k}"] = model["coef"][k]["per_sd"]
                M[f"{name}_p_{k}"] = model["coef"][k]["p"]
    for k in ["aim", "speed", "ratio"]:
        if fits[k]:
            M[f"r_acc_{k}"] = fits[k]["r"]
            M[f"slope_acc_{k}"] = fits[k]["slope"]
        if mfits[k]:
            M[f"r_miss_{k}"] = mfits[k]["r"]

    s = [f"## 1. Aim vs speed: {verdict}", "",
         f"Sample: {len(cur)} {cur_label} ({(cur['source'] == 'top').sum()} from top plays, "
         f"{(cur['source'] == 'recent').sum()} extra from recent logs). Top plays are selected for high pp, which "
         "flattens slopes, so the all-attempts check further down is the cross-check.", "",
         "| accuracy vs | slope | r | p | n |", "|---|---|---|---|---|",
         line("aim", fits["aim"]), line("speed", fits["speed"]), line("speed/aim", fits["ratio"]),
         line("stars", fits["stars"]), "",
         "| per 100 objects vs | slope | r | p | n |", "|---|---|---|---|---|",
         line("misses ~ aim", mfits["aim"], "misses"), line("misses ~ speed", mfits["speed"], "misses"),
         line("misses ~ speed/aim", mfits["ratio"], "misses"),
         line("100s+50s ~ aim", ofits["aim"], "100s"), line("100s+50s ~ speed", ofits["speed"], "100s"), ""]
    if j:
        s += [f"Aim and speed difficulty rise together on these maps (r = {j['x_corr']:+.2f}), so each simple slope above "
              "carries part of the other's effect. Fitting both at once separates them (per SD = effect of a typical step "
              "up in that skill, holding the other fixed):", "",
              "| joint model | skill | coef per 1.0 | per 1 SD | p | n |", "|---|---|---|---|---|---|",
              jrow("accuracy %", j, "aim", "pp"), jrow("accuracy %", j, "speed", "pp"),
              jrow("misses / 100 obj", jm, "aim", ""), jrow("misses / 100 obj", jm, "speed", ""),
              jrow("100s+50s / 100 obj", jo, "aim", ""), jrow("100s+50s / 100 obj", jo, "speed", ""),
              jrow("all attempts incl. fails: acc", ja, "aim", "pp"), jrow("all attempts incl. fails: acc", ja, "speed", "pp"),
              jrow("all attempts incl. fails: misses", jam, "aim", ""),
              jrow("all attempts incl. fails: misses", jam, "speed", ""), ""]
    if jr:
        s.append(f"Holding star rating fixed, +0.1 speed/aim ratio moves accuracy {jr['coef']['ratio']['b'] / 10:+.2f} pp "
                 f"(p {jr['coef']['ratio']['p']:.3g}).")
    hard = cur[cur["stars"] >= cur["stars"].median()].sort_values("acc", ascending=False).head(5)
    if len(hard):
        pa = lambda v, col: 100 * (cur[col] < v).mean()  # noqa: E731
        s += ["", f"Your best accuracies on maps at or above your median star rating ({cur['stars'].median():.2f}*), "
              "with where the map sits among your plays (percentile):", "",
              "| map | acc | stars | aim (pct) | speed (pct) | misses | 100s |", "|---|---|---|---|---|---|---|"]
        for _, r in hard.iterrows():
            s.append(f"| {r['title']} [{r['diff']}] | {r['acc']:.2f} | {r['stars']:.2f} | {r['aim']:.2f} ({pa(r['aim'], 'aim'):.0f}) | "
                     f"{r['speed']:.2f} ({pa(r['speed'], 'speed'):.0f}) | {int(r['miss'])} | {int(r['n100'])} |")
    s += ["", "![](1_acc_miss_vs_aim_speed.png)", ""]
    R.append("\n".join(s))

    # 2. error types ------------------------------------------------------
    charts.chart_errors(cur, out)
    et = cur.groupby("err_type").agg(n=("acc", "size"), acc=("acc", "mean"), aim=("aim", "mean"),
                                     speed=("speed", "mean"), ratio=("ratio", "mean"), bpm=("eff_bpm", "mean"),
                                     miss=("miss_pct", "mean"), ok=("ok_pct", "mean"), n50=("n50_pct", "mean"))
    et["share"] = 100 * et["n"] / et["n"].sum()
    src = pd.crosstab(cur["source"], cur["err_type"], normalize="index") * 100
    for t in et.index:
        M[f"share_{t}"] = et.loc[t, "share"]
    # Accuracy points lost per error type: a miss costs a full hit, a 100 two thirds, a 50 five sixths
    hits = cur["hits"].where(cur["hits"] > 0)
    M["loss_miss"] = float((100 * cur["miss"] / hits).mean())
    M["loss_ok"] = float((100 * (cur["n100"] * 2 / 3 + cur["n50"] * 5 / 6) / hits).mean())
    M["miss_per100"], M["ok_per100"] = float(cur["miss_pct"].mean()), float(cur["ok_pct"].mean())
    s = ["## 2. Error type per play", "",
         f"Accuracy lost on average: {M['loss_ok']:.1f} pp to 100s/50s ({M['ok_per100']:.1f} per 100 objects), "
         f"{M['loss_miss']:.1f} pp to misses ({M['miss_per100']:.1f} per 100 objects).", "",
         "Rules: clean = at most 1 miss and under 2% 100s/50s; miss-heavy = 2+ misses and fewer than 4 100s/50s per miss; "
         "timing-heavy = at most 1 miss, or 12+ 100s/50s per miss; mixed = the rest.", "",
         "| type | plays | share | acc | aim | speed | speed/aim | eff BPM | miss/100 | 100s+50s/100 | 50s/100 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in et.sort_values("n", ascending=False).iterrows():
        s.append(f"| {t} | {int(r['n'])} | {r['share']:.0f}% | {r['acc']:.2f} | {r['aim']:.2f} | {r['speed']:.2f} | "
                 f"{r['ratio']:.2f} | {r['bpm']:.0f} | {r['miss']:.2f} | {r['ok']:.2f} | {r['n50']:.2f} |")
    s += ["", "By source (% of passes): " + "; ".join(
        f"{i}: " + ", ".join(f"{c} {v:.0f}%" for c, v in row.items() if v > 0) for i, row in src.iterrows()), ""]
    th = et.loc["timing-heavy"] if "timing-heavy" in et.index else None
    mh = et.loc["miss-heavy"] if "miss-heavy" in et.index else None
    if th is not None and mh is not None:
        s.append(f"Timing-heavy plays sit at speed/aim {th['ratio']:.2f} and {th['bpm']:.0f} BPM; miss-heavy plays at "
                 f"{mh['ratio']:.2f} and {mh['bpm']:.0f} BPM.")
    s += ["", "![](2_error_types.png)", ""]
    R.append("\n".join(s))

    # 3. BPM ---------------------------------------------------------------
    h = hinge(cur, "eff_bpm", "acc")
    attempts = rall if not rall.empty else recent_now
    charts.chart_bpm(cur, attempts, h, out)
    bb = binned(cur, "eff_bpm", 20)
    bb = bb[bb["n"] >= 1]
    zb = zones(binned(cur, "eff_bpm", 20))
    M["bpm_first_bin_below_93"] = zb["first_below_93"]
    s = ["## 3. Effective BPM", "", "| eff BPM | plays | mean acc | median acc |", "|---|---|---|---|"]
    s += [f"| {r['lo']:.0f}-{r['hi']:.0f} | {int(r['n'])} | {f(r['mean'])} | {f(r['median'])} |" for _, r in bb.iterrows()]
    s.append("")
    if h:
        M.update({"bpm_break_p": h["p"], "bpm_break_gain": h["sse_gain_pct"], "bpm_lin_slope": h["lin"]["slope"] * 10,
                  "bpm_lin_p": h["lin"]["p"], "bpm_break_k_tested": h["k"]})
        test = (f"tested against a straight line by refitting the breakpoint search on 1,000 simulated no-break "
                f"datasets: best break at {h['k']:.0f} BPM explains {h['sse_gain_pct']:.1f}% more variance, p={h['p']:.3f}")
        if h["significant"]:
            M.update({"bpm_break": h["k"], "bpm_slope_before": h["before"] * 10, "bpm_slope_after": h["after"] * 10})
            s.append(f"Breakpoint: accuracy changes {h['before'] * 10:+.2f} pp per 10 BPM below **{h['k']:.0f} BPM** and "
                     f"{h['after'] * 10:+.2f} pp per 10 BPM above it ({test}; n={h['n']}).")
        else:
            why = "no better than a straight line" if h["p"] >= 0.05 else "not a drop-off (accuracy does not fall after it)"
            s.append(f"No BPM breakpoint: the best two-segment fit is {why} ({test}; n={h['n']}). Straight line: "
                     f"{h['lin']['slope'] * 10:+.2f} pp per 10 BPM (r={h['lin']['r']:+.2f}, p={h['lin']['p']:.3f}). "
                     "BPM is left out of the practice plan.")
    if zb["comfort"] is None:
        s.append("No 20-BPM bin averages 93%+, so the drop-off is relative to your own level, not to 93%.")
    elif zb["first_below_93"] is not None:
        s.append(f"First 20-BPM bin (n>={MIN_N}) averaging under 93%: {zb['first_below_93']:.0f} BPM.")
    s.append("BPM is the map's main BPM; stream density (1/4 vs 1/2 snap) is captured by speed difficulty, not here.")
    s += ["", "![](3_bpm_dropoff.png)", ""]
    R.append("\n".join(s))

    # 4. comfort / push ----------------------------------------------------
    charts.chart_zones(cur, out)
    Z = {}
    s = ["## 4. Comfort and push zones", "",
         f"Binned: mean accuracy per bin over {cur_label}, a bin needs {MIN_N}+ plays. Fit: where the straight-line fit of "
         "accuracy on that measure crosses 93% and 85% (less noisy than bins, but assumes a straight line).", "",
         "| measure | comfort 93%+ (bins) | push 85-92% (bins) | fit: 93% at or below | fit: 85% at or below |",
         "|---|---|---|---|---|"]
    for x, step in [("stars", 0.5), ("aim", 0.25), ("speed", 0.25), ("eff_bpm", 20)]:
        Z[x] = zones(binned(cur, x, step))
        fit = simple(cur, x, "acc")
        Z[x]["fit93"] = Z[x]["fit85"] = None
        if fit and fit["slope"] < 0 and fit["p"] < 0.05:
            Z[x]["fit93"] = (COMFORT - fit["intercept"]) / fit["slope"]
            Z[x]["fit85"] = (PUSH[0] - fit["intercept"]) / fit["slope"]
        nd = 0 if x == "eff_bpm" else 2
        s.append(f"| {x} | {rng(Z[x]['comfort'], nd)} | {rng(Z[x]['push'], nd)} | {f(Z[x]['fit93'], nd)} | {f(Z[x]['fit85'], nd)} |")
        M[f"comfort_{x}_max"] = Z[x]["comfort"][1] if Z[x]["comfort"] else None
        M[f"push_{x}_max"] = Z[x]["push"][1] if Z[x]["push"] else None
    st_ = cur["stars"]
    M.update({"acc_3_5": cur[(st_ >= 3) & (st_ < 5)]["acc"].mean(), "n_3_5": int(((st_ >= 3) & (st_ < 5)).sum()),
              "n_2_3": int(((st_ >= 2) & (st_ < 3)).sum()), "n_cur": len(cur),
              "n93_hard": int(((cur["acc"] >= COMFORT) & (st_ >= 2.5)).sum()), "n93": int((cur["acc"] >= COMFORT).sum())})
    f35 = simple(cur[(st_ >= 3) & (st_ < 5)], "stars", "acc")
    if f35:
        M.update({"slope_3_5": f35["slope"], "p_3_5": f35["p"]})
    sb = binned(cur, "stars", 0.5)
    sb = sb[sb["n"] >= 1]
    s += ["", "| stars | plays | mean acc | median acc |", "|---|---|---|---|"]
    s += [f"| {r['lo']:.1f}-{r['hi']:.1f} | {int(r['n'])} | {f(r['mean'])} | {f(r['median'])} |" for _, r in sb.iterrows()]
    played = att if len(att) else cur
    M["attempt_median_stars"] = played["stars"].median()
    s += ["", f"For scale: the median star rating of your logged attempts is {played['stars'].median():.2f}* "
          f"(aim {played['aim'].median():.2f}, speed {played['speed'].median():.2f}).",
          "", "![](4_comfort_zones.png)", ""]
    R.append("\n".join(s))

    # 5. fails / retries ----------------------------------------------------
    s = ["## 5. Fails and retries", ""]
    if attempts.empty:
        s.append("No recent plays in the log yet. Run the pipeline right after a session (the API only keeps 24h).")
    else:
        charts.chart_fails(attempts, out)
        a = attempts.sort_values("date").copy()
        fr = 100 * (1 - a["passed"].mean())
        fr_now = 100 * (1 - recent_now["passed"].mean()) if not recent_now.empty else None
        M.update({"fail_rate_all": fr, "fail_rate_now": fr_now, "attempts_all": len(a)})
        span = f"{a['date'].min():%Y-%m-%d %H:%M} to {a['date'].max():%Y-%m-%d %H:%M} UTC"
        a["key"] = a["beatmap_id"].astype("Int64").astype(str) + " " + a["mods"].astype(str)
        a["streak"] = (a["key"] != a["key"].shift()).cumsum()
        per = a.groupby("key").agg(
            map=("title", "first"), diff=("diff", "first"), mods=("mods", "first"), tries=("passed", "size"),
            passes=("passed", "sum"), best=("acc", "max"), stars=("stars", "first"), aim=("aim", "first"),
            speed=("speed", "first"), bpm=("eff_bpm", "first"))
        fails = a[~a["passed"]]
        per["fail_at"] = fails.groupby("key")["completion"].median() * 100
        per["fails"] = per["tries"] - per["passes"]
        gave_up = per[(per["passes"] == 0) & (per["tries"] >= 2)]
        top_r = per.sort_values(["tries", "fails"], ascending=False).iloc[0]
        M.update({"never_passed": len(gave_up), "n_maps_tried": len(per), "top_retry": f"{top_r['map']} [{top_r['diff']}]",
                  "top_retry_tries": int(top_r["tries"]), "top_retry_fails": int(top_r["fails"]),
                  "fail_first_half": float(100 * (a.loc[~a["passed"], "completion"] < 0.5).mean())})
        streaks = a.groupby("streak").size()
        s += [f"{len(a)} submitted attempts logged ({span}), {len(per)} map/mod combos. Fail rate {fr:.0f}% "
              f"(this pull: {f(fr_now, 0)}%). A fail is any attempt that ended without a pass; whether quits/restarts "
              "count depends on the client (lazer submits them, stable does not), and the playcount check under "
              "'Since last run' shows which applies to you."
              + (f" This pull hit the API's 100-attempt limit, so anything before {recent_now['date'].min():%H:%M} UTC "
                 "that day was not captured: run the pipeline mid-session on long days." if len(recent_now) >= 100 else ""),
              "",
              f"Attempts per map: median {per['tries'].median():.0f}, {int((per['tries'] >= 3).sum())} maps with 3+ tries, "
              f"longest back-to-back streak {streaks.max()} on the same map. Maps tried 2+ times and never passed: "
              f"{len(gave_up)}.", ""]
        if len(fails):
            M["fail_point_median"] = 100 * fails["completion"].median()
            s.append(f"Fails happen at a median {100 * fails['completion'].median():.0f}% through the map "
                     f"({100 * (fails['completion'] < 0.33).mean():.0f}% in the first third, "
                     f"{100 * (fails['completion'] > 0.66).mean():.0f}% in the last third).")
            fm = a.assign(fail=~a["passed"]).groupby("passed")[["stars", "aim", "speed", "ratio", "eff_bpm"]].mean()
            if len(fm) == 2:
                s.append(f"Failed vs passed attempts, mean: stars {fm.loc[False, 'stars']:.2f} vs {fm.loc[True, 'stars']:.2f}, "
                         f"aim {fm.loc[False, 'aim']:.2f} vs {fm.loc[True, 'aim']:.2f}, speed {fm.loc[False, 'speed']:.2f} vs "
                         f"{fm.loc[True, 'speed']:.2f}, speed/aim {fm.loc[False, 'ratio']:.2f} vs {fm.loc[True, 'ratio']:.2f}, "
                         f"BPM {fm.loc[False, 'eff_bpm']:.0f} vs {fm.loc[True, 'eff_bpm']:.0f}.")
                # logistic-free check: which of aim/speed separates fails better (point-biserial r)
                for k in ["aim", "speed", "eff_bpm"]:
                    d = a[[k, "passed"]].dropna()
                    if d[k].nunique() > 2 and d["passed"].nunique() == 2:
                        M[f"r_fail_{k}"] = stats.pointbiserialr(~d["passed"], d[k])[0]
                s.append("Correlation of failing with: " + ", ".join(
                    f"{k} r = {M[f'r_fail_{k}']:+.2f}" for k in ["aim", "speed", "eff_bpm"] if f"r_fail_{k}" in M) + ".")
        s += ["", "Most-retried maps:", "",
              "| map | mods | tries | fails | best acc | stars | aim | speed | eff BPM | median fail point |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in per.sort_values(["tries", "fails"], ascending=False).head(12).iterrows():
            s.append(f"| {r['map']} [{r['diff']}] | {r['mods']} | {int(r['tries'])} | {int(r['fails'])} | {f(r['best'])} | "
                     f"{f(r['stars'])} | {f(r['aim'])} | {f(r['speed'])} | {f(r['bpm'], 0)} | "
                     f"{f(r['fail_at'], 0) + '%' if np.isfinite(r['fail_at']) else '-'} |")
        s += ["", "![](5_fails.png)", ""]
    R.append("\n".join(s))

    # 6. activity -----------------------------------------------------------
    monthly = pd.DataFrame(user.get("monthly_playcounts") or [])
    if not monthly.empty:
        monthly["month"] = pd.to_datetime(monthly["start_date"])
        monthly = monthly.sort_values("month")
    rank = ((user.get("rank_history") or {}).get("data")) or []
    charts.chart_activity(monthly, rank, out)
    s = ["## 6. Activity and rank", "",
         f"Profile: {f(st.get('pp'), 0)} pp, global rank {st.get('global_rank')}, {f(st.get('hit_accuracy'))}% acc, "
         f"{st.get('play_count')} plays, {f((st.get('play_time') or 0) / 3600, 0)} h played.", ""]
    if not monthly.empty:
        yearly = monthly.groupby(monthly["month"].dt.year)["count"].sum()
        s.append("Plays per year: " + ", ".join(f"{y}: {c}" for y, c in yearly.items()) + ".")
        grid = monthly.assign(y=monthly["month"].dt.year, m=monthly["month"].dt.month) \
            .pivot_table(index="y", columns="m", values="count", aggfunc="sum")
        s += ["", "| year | " + " | ".join(f"{m:02d}" for m in range(1, 13)) + " | total |", "|---" * 14 + "|"]
        for y, r in grid.iterrows():
            s.append(f"| {y} | " + " | ".join("" if pd.isna(r.get(m)) else str(int(r[m])) for m in range(1, 13))
                     + f" | {int(yearly[y])} |")
        M["plays_this_month"] = int(monthly["count"].iloc[-1])
    if rank:
        M.update({"rank_90d_ago": rank[0], "rank_best_90d": min(rank)})
        s += ["", f"Rank history (90 days): {rank[0]} -> {rank[-1]} (best {min(rank)}, worst {max(rank)}). "
              f"Last 7 days: {rank[-8] if len(rank) >= 8 else rank[0]} -> {rank[-1]}; last 30: "
              f"{rank[-31] if len(rank) >= 31 else rank[0]} -> {rank[-1]}."]
    most = exp.get("most_played") or []
    mpa = exp.get("most_played_attrs") or {}
    if most:
        mp = pd.DataFrame([{"count": m.get("count"), "bid": str(m.get("beatmap_id")),
                            "title": (m.get("beatmapset") or {}).get("title"),
                            "diff": (m.get("beatmap") or {}).get("version"),
                            "stars": (m.get("beatmap") or {}).get("difficulty_rating"),
                            "length": (m.get("beatmap") or {}).get("total_length")} for m in most])
        mp["aim"] = mp["bid"].map(lambda b: (mpa.get(b) or {}).get("aim_difficulty"))
        mp["speed"] = mp["bid"].map(lambda b: (mpa.get(b) or {}).get("speed_difficulty"))
        mp["ratio"] = mp["speed"] / mp["aim"]
        w = mp.dropna(subset=["ratio"])
        tot = mp["count"].sum()
        s += ["", f"Most-played maps: {len(mp)} maps, {tot} plays. Playcount-weighted nomod stars "
              f"{np.average(mp['stars'], weights=mp['count']):.2f}, length {np.average(mp['length'], weights=mp['count']):.0f}s"
              + (f"; over the {len(w)} most played with attributes, speed/aim {np.average(w['ratio'], weights=w['count']):.2f} "
                 f"(your 2023+ passes: {cur['ratio'].mean():.2f})." if len(w) else "."),
              "", "| map | plays | stars | aim | speed | length |", "|---|---|---|---|---|---|"]
        for _, r in mp.sort_values("count", ascending=False).head(10).iterrows():
            s.append(f"| {r['title']} [{r['diff']}] | {r['count']} | {f(r['stars'])} | {f(r['aim'])} | {f(r['speed'])} | {r['length']}s |")
    s += ["", "![](6_activity.png)", ""]
    R.append("\n".join(s))

    # 7. old vs new -----------------------------------------------------------
    charts.chart_eras(top, out, era_labels)
    s = [f"## 7. {ds.old_label} vs {ds.since}", ""]
    pool = pd.concat([top, rall[rall["passed"]]] if not rall.empty else [top]).drop_duplicates("uid")
    fs = simple(pool, "stars", "acc")
    if fs:
        pool["acc_resid"] = pool["acc"] - (fs["intercept"] + fs["slope"] * pool["stars"])
    rows = []
    for e in [ERA_OLD, ERA_NEW]:
        d = pool[pool["era"] == e]
        if d.empty:
            continue
        rows.append(f"| {era_labels[e]} | {len(d)} | {d['acc'].mean():.2f} | {f(d['acc_resid'].mean()) if fs else 'n/a'} | "
                    f"{d['stars'].median():.2f} | {d['aim'].median():.2f} | {d['speed'].median():.2f} | "
                    f"{d['ratio'].median():.2f} | {d['eff_bpm'].median():.0f} | {d['length_s'].median():.0f} | "
                    f"{d['miss_pct'].mean():.2f} | {d['ok_pct'].mean():.2f} | {100 * d['dt'].mean():.0f}% | "
                    f"{100 * (d['err_type'] == 'timing-heavy').mean():.0f}% | {100 * (d['err_type'] == 'miss-heavy').mean():.0f}% |")
        M[f"acc_{e}"] = d["acc"].mean()
        M[f"resid_{e}"] = d["acc_resid"].mean() if fs else None
        M[f"stars_{e}"] = d["stars"].median()
        M[f"n_{e}"] = len(d)
    s += ["Passed plays (top plays + logged recent passes). Acc vs expected = accuracy minus what the star-rating fit "
          "predicts, so eras are compared at equal difficulty.", "",
          "| era | plays | acc | acc vs expected | med stars | med aim | med speed | med speed/aim | med eff BPM | "
          "med length s | miss/100 | 100s/100 | DT | timing-heavy | miss-heavy |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    old_ct = int((top["era"] == ERA_OLD).sum())
    s += ["", f"{old_ct} of your top {len(top)} plays are from {ds.old_label}.", "", "![](7_old_vs_new.png)", ""]
    R.append("\n".join(s))

    # Watch list ---------------------------------------------------------------
    lk = pd.DataFrame(exp.get("lookups") or [])
    if not lk.empty:
        mine = pd.concat([top, rall]).drop_duplicates("uid") if not rall.empty else top
        s = ["## Maps on your watch list", "", "| map | diff | nomod stars | aim | speed | BPM | length | aim pct | speed pct | your best |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        M["watch"] = []
        for _, r in lk.sort_values("stars").iterrows():
            pa, ps = 100 * (cur["aim"] < r["aim"]).mean(), 100 * (cur["speed"] < r["speed"]).mean()
            yours = mine[mine["beatmap_id"] == r["beatmap_id"]]["acc"]
            best = float(yours.max()) if len(yours) else None
            s.append(f"| {r['artist']} - {r['title']} | {r['version']} | {f(r['stars'])} | {f(r['aim'])} | {f(r['speed'])} | "
                     f"{r['bpm']} | {r['length_s']}s | {pa:.0f} | {ps:.0f} | {f(best)} |")
            M["watch"].append({"name": f"{r['title']} [{r['version']}]", "stars": r["stars"], "aim": r["aim"], "speed": r["speed"],
                               "aim_pct": pa, "speed_pct": ps, "best": best})
        R.append("\n".join(s) + "\n")

    # Map types you named ----------------------------------------------------
    fs_cur = simple(cur, "stars", "acc")
    s = ["## Map types", "",
         "Tagged by title (see [tags] in config.toml). Acc vs expected = accuracy minus what your star-rating fit predicts for that map.", "",
         "| tag | passes | mean acc | acc vs expected | attempts logged | fail rate |", "|---|---|---|---|---|---|"]
    M["tags"] = list(cfg.tags)
    for tag, rx in cfg.tags.items():
        d = cur[cur["title"].astype(str).str.contains(rx, case=False, regex=True)]
        o = cur.drop(d.index)
        ad = att[att["title"].astype(str).str.contains(rx, case=False, regex=True)] if len(att) else att
        oa = att.drop(ad.index) if len(att) else att
        for lab, dd, aa in [(tag, d, ad), (f"not {tag}", o, oa)]:
            if not len(dd):
                continue
            resid = (dd["acc"] - (fs_cur["intercept"] + fs_cur["slope"] * dd["stars"])).mean() if fs_cur else np.nan
            fail = 100 * (1 - aa["passed"].mean()) if len(aa) else None
            M[f"tag_{lab}_n"], M[f"tag_{lab}_resid"], M[f"tag_{lab}_fail"], M[f"tag_{lab}_att"] = len(dd), resid, fail, len(aa)
            s.append(f"| {lab} | {len(dd)} | {dd['acc'].mean():.2f} | {resid:+.2f} | {len(aa)} | "
                     f"{f(fail, 0) + '%' if fail is not None else 'n/a'} |")
    R.append("\n".join(s) + "\n")

    # Practice plan -------------------------------------------------------------
    plan = practice_plan(cur, Z, h, attempts)
    M["plan"] = plan["targets"]
    for w in ["comfort", "push"]:
        b = plan["targets"].get(f"{w}_stars")
        M[f"plan_{w}_stars_top"] = b[1] if b else None

    # Skill model and weakness scan ---------------------------------------------
    coach_res = coach.analyze(ds)
    coach_md = coach.report_section(coach_res, ds, out)
    coach.metrics(coach_res, M)

    # Compare with previous run ------------------------------------------------
    prev = previous_metrics(hist, M["exported_at"])
    cmp_md = compare(prev, M, rall)

    head = [f"# osu! skill analysis: {M['username']}", "",
            f"Run {M['run_at']} (data exported {M['exported_at']}). {len(top)} top plays, {len(recent_now)} recent plays this pull, "
            f"{len(rall)} recent plays logged in total. Next session: see plan.md.", "", "**Aim vs speed:** " + verdict + ".", ""]
    report = "\n".join(head + [coach_md, findings(M), plan["md"], cmp_md, "# Details", ""] + R)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(M, indent=1, default=float), encoding="utf-8")
    return M, prev, coach_res


STEPS = {"stars": 0.5, "aim": 0.25, "speed": 0.25, "eff_bpm": 20}


def practice_plan(cur, Z, h, attempts):
    def ladder(x):
        """Comfort = the top two bins where you already average 85%+; push = the two bins above that."""
        step = STEPS[x]
        b = binned(cur, x, step)
        ok = b[(b["n"] >= MIN_N) & (b["mean"] >= PUSH[0])]
        top = float(ok["hi"].max()) if len(ok) else Z[x]["fit85"]
        if top is None or not np.isfinite(top):
            return None, None
        lo = max(top - 2 * step, 0.0)
        return (lo, top), (top, top + 2 * step)

    def show(b, nd=2):
        if not b:
            return "n/a"
        lo, hi = b
        return f"<= {hi:.{nd}f}" if lo is None else f"{lo:.{nd}f}-{hi:.{nd}f}"

    tgt = {}
    for x in ["stars", "aim", "speed"]:
        tgt[f"comfort_{x}"], tgt[f"push_{x}"] = ladder(x)
    # BPM only enters the plan when a significant drop-off exists: comfort below it, push just above it
    use_bpm = bool(h and h["significant"])
    if use_bpm:
        tgt["comfort_eff_bpm"] = (None, h["k"])
        tgt["push_eff_bpm"] = (h["k"], h["k"] + 20)

    # Star level where most attempts fail: a cap for push attempts
    cap = None
    if not attempts.empty:
        a = attempts.copy()
        a["bin"] = (a["stars"] // 0.5) * 0.5
        g = a.groupby("bin")["passed"].agg(["size", "mean"])
        g = g[g["size"] >= 5]
        hi_fail = g[g["mean"] < 0.25]
        if len(hi_fail):
            cap = float(hi_fail.index.min())
            tgt["fail_cap_stars"] = cap
            tgt["fail_rate_above_cap"] = float(100 * (1 - a[a["stars"] >= cap]["passed"].mean()))

    bpm_note = ("BPM comes from the breakpoint in section 3." if use_bpm else
                "BPM is left out: no significant BPM drop-off in your data (section 3).")
    bpm_col = lambda w: f" {show(tgt[f'{w}_eff_bpm'], 0)} |" if use_bpm else ""  # noqa: E731
    md = ["## Difficulty bands", "",
          "Comfort = the top two bins where you already average 85%+ (goal: lift them to 93%). Push = the two bins above "
          f"that (goal: pass, then 85-92%). {bpm_note} The ranges move up as you improve; rerun after sessions.", "",
          "| map type | stars | aim | speed |" + (" eff BPM |" if use_bpm else "") + " goal |",
          "|---|---|---|---|" + ("---|" if use_bpm else "") + "---|",
          f"| comfort (accuracy reps) | {show(tgt['comfort_stars'])} | {show(tgt['comfort_aim'])} | {show(tgt['comfort_speed'])} |"
          f"{bpm_col('comfort')} 93%+, at most 1 miss |",
          f"| push | {show(tgt['push_stars'])} | {show(tgt['push_aim'])} | {show(tgt['push_speed'])} |"
          f"{bpm_col('push')} pass, then 85-92% |", ""]
    if cap is not None:
        md.append(f"Cap: attempts at {cap:.1f}*+ end in a fail {tgt['fail_rate_above_cap']:.0f}% of the time. Keep those to a "
                  "small share of a session until the push band is at 90%.")
    md.append("")
    return {"md": "\n".join(md), "targets": tgt}


def ordinal(x):
    n = int(round(x))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def findings(M):
    """One paragraph per finding, written from this run's numbers. Paragraphs drop out when the data is missing."""
    g = M.get
    P = ["## Findings", ""]

    # Every direction or comparison word below is chosen from the numbers, never fixed text
    def pv(p):
        return "p<0.001" if p < 0.001 else f"p={p:.3f}"

    def sig(p):
        return p is not None and p < 0.05

    def acc_effect(v):
        return f"costs {abs(v):.1f} pp accuracy" if v < 0 else f"adds {v:.1f} pp accuracy"

    def compare(a, b, what_a, what_b, unit):
        """'a is about the same as b' / 'a is larger than b', by magnitude, both values shown."""
        big, small = max(abs(a), abs(b)), min(abs(a), abs(b))
        if big == 0 or small / big >= 2 / 3:
            return f"{what_a} and {what_b} are about the same ({a:+.1f} vs {b:+.1f} {unit})"
        lead = what_a if abs(a) > abs(b) else what_b
        return f"{lead} is the larger effect ({what_a} {a:+.1f}, {what_b} {b:+.1f} {unit})"

    if g("acc_sd_speed") is not None and g("miss_sd_aim") is not None:
        t = (f"**Aim vs speed.** Fitting aim and speed together on your {g('acc_n')} passes {g('era_since')}: a typical step up "
             f"in speed difficulty {acc_effect(g('acc_sd_speed'))} ({pv(g('acc_p_speed'))}); the same step in aim "
             f"{acc_effect(g('acc_sd_aim'))} ({pv(g('acc_p_aim'))}). Per 100 objects, speed changes 100s/50s by "
             f"{g('ok_sd_speed'):+.1f} ({pv(g('ok_p_speed'))}) and misses by {g('miss_sd_speed'):+.2f} "
             f"({pv(g('miss_p_speed'))}); aim changes 100s/50s by {g('ok_sd_aim'):+.1f} ({pv(g('ok_p_aim'))}) and misses by "
             f"{g('miss_sd_aim'):+.2f} ({pv(g('miss_p_aim'))}).")
        clean = (sig(g("ok_p_speed")) and g("ok_sd_speed") > 0 and not sig(g("miss_p_speed"))
                 and sig(g("miss_p_aim")) and g("miss_sd_aim") > 0 and not sig(g("ok_p_aim")))
        if clean:
            t += " The error types split cleanly: speed makes 100s, aim makes misses."
        if g("att_acc_sd_aim") is not None:
            t += (f" Across all {g('att_acc_n')} logged attempts including fails (no top-list selection), "
                  + compare(g("att_acc_sd_aim"), g("att_acc_sd_speed"), "aim", "speed", "pp accuracy per step"))
            am, ap = g("att_miss_sd_aim"), g("att_miss_p_aim")
            if sig(ap):
                t += f", and aim {'adds' if am > 0 else 'removes'} {abs(am):.1f} misses per 100 objects per step ({pv(ap)})."
            else:
                t += f"; aim's effect on misses there is not significant ({am:+.1f}, {pv(ap)})."
        t += f" Verdict: {M['verdict']}."
        P += [t, ""]

    if g("share_miss-heavy") is not None:
        mh, mx = g("share_miss-heavy") or 0, g("share_mixed") or 0
        lo_, lm = g("loss_ok"), g("loss_miss")
        cost = ("100s/50s cost you more accuracy than misses" if lo_ > lm else
                "misses cost you more accuracy than 100s/50s" if lm > lo_ else "100s/50s and misses cost the same accuracy")
        P += [f"**Error type.** {mh:.0f}% of your passes {g('era_since')} are miss-heavy, {mx:.0f}% mixed, "
              f"{g('share_timing-heavy') or 0:.0f}% timing-heavy (few misses, many 100s) and {g('share_clean') or 0:.0f}% clean. "
              f"In accuracy terms {cost}: {lo_:.1f} pp per play from 100s/50s ({g('ok_per100'):.1f} per 100 objects) "
              f"against {lm:.1f} pp from misses ({g('miss_per100'):.1f} per 100 objects). Misses also break combo and "
              "drain HP, which is what fails a play.", ""]

    if g("fail_rate_all") is not None:
        u = g("unsubmitted_est")
        what = (" (did not finish: your playcount matches submitted attempts, so these include quits and restarts)"
                if u is not None and u <= 1 else "")
        P += [f"**Fails.** {g('fail_rate_all'):.0f}% of {g('attempts_all')} logged attempts are fails{what}. Your median attempt is "
              f"{g('attempt_median_stars'):.2f}*, and you fail at a median {g('fail_point_median'):.0f}% through the map "
              f"({g('fail_first_half'):.0f}% of fails come in the first half). {g('never_passed')} of {g('n_maps_tried')} maps were "
              f"tried 2+ times without a pass; the most-retried was {g('top_retry')} ({g('top_retry_fails')} fails in "
              f"{g('top_retry_tries')} tries).", ""]

    if g("acc_3_5") is not None and g("n_3_5"):
        sl, sp_ = g("slope_3_5"), g("p_3_5")
        trend = ("" if sl is None else
                 f", falling {abs(sl):.1f} pp per star across that range ({pv(sp_)})" if sig(sp_) and sl < 0 else
                 f", rising {sl:.1f} pp per star across that range ({pv(sp_)})" if sig(sp_) else
                 f", with no significant trend across that range ({sl:+.1f} pp per star, {pv(sp_)})")
        n23 = g("n_2_3")
        P += [f"**Accuracy by difficulty.** Between 3.0 and 5.0* your passes average {g('acc_3_5'):.1f}% (n={g('n_3_5')})"
              f"{trend}. {g('n93')} of {g('n_cur')} passes {g('era_since')} reach 93%, {g('n93_hard')} of them at 2.5*+. "
              f"You have {n23} passes between 2.0 and 3.0*"
              + (", so the band where you would build clean accuracy is barely played." if n23 < 10 else "."), ""]

    if g("bpm_break") is not None:
        P += [f"**BPM.** Accuracy changes {g('bpm_slope_before'):+.1f} pp per 10 BPM up to {g('bpm_break'):.0f} BPM, then "
              f"{g('bpm_slope_after'):+.1f} pp per 10 BPM above it (breakpoint p={g('bpm_break_p'):.3f}).", ""]
    elif g("bpm_break_p") is not None:
        P += [f"**BPM.** No drop-off point: the best breakpoint ({g('bpm_break_k_tested'):.0f} BPM) explains "
              f"{g('bpm_break_gain'):.1f}% more variance than a straight line (p={g('bpm_break_p'):.2f}), and the straight "
              f"line is {g('bpm_lin_slope'):+.1f} pp per 10 BPM (p={g('bpm_lin_p'):.2f}). Within the BPM you play, BPM itself "
              "does not predict accuracy, so it is left out of the plan; speed difficulty already covers stream density.", ""]

    parts = []
    for tag in g("tags") or []:
        if g(f"tag_{tag}_n"):
            fr, fo = g(f"tag_{tag}_fail"), g(f"tag_not {tag}_fail")
            parts.append(f"{tag} maps: {g(f'tag_{tag}_n')} passes, {g(f'tag_{tag}_resid'):+.1f} pp vs the accuracy your "
                         f"star-rating fit predicts" + (f", fail rate {fr:.0f}% vs {fo:.0f}% on other maps" if fr is not None
                                                        and fo is not None else ""))
    if parts:
        P += ["**Map types.** " + "; ".join(parts) + ".", ""]

    for w in g("watch") or []:
        side = ("harder on aim than on speed" if w["aim_pct"] > w["speed_pct"] else
                "harder on speed than on aim" if w["speed_pct"] > w["aim_pct"] else "evenly on aim and speed")
        t = (f"**{w['name']}** is {w['stars']:.2f}* nomod (aim {w['aim']:.2f}, {ordinal(w['aim_pct'])} percentile of your plays; "
             f"speed {w['speed']:.2f}, {ordinal(w['speed_pct'])} percentile). Relative to the maps you play, it rates {side}.")
        if w["best"] is not None:
            t += f" Your best on it is {w['best']:.2f}%."
        P += [t, ""]

    if g("acc_old") is not None and g("acc_new") is not None:
        t = (f"**Old vs new.** Passes {g('era_old')}: {g('acc_old'):.1f}% at median {g('stars_old'):.2f}* (n={g('n_old')}). {g('era_since').capitalize()}: "
             f"{g('acc_new'):.1f}% at {g('stars_new'):.2f}* (n={g('n_new')}).")
        if g("resid_old") is not None and g("resid_new") is not None:
            d = g("resid_new") - g("resid_old")  # positive = better now than the old self at equal stars
            where = ("about level with" if abs(d) < 0.5 else f"{abs(d):.1f} pp {'above' if d > 0 else 'below'}")
            t += (f" Against what your star-rating fit predicts, old plays sit at {g('resid_old'):+.2f} pp and new ones at "
                  f"{g('resid_new'):+.2f} pp, so at equal star rating you are {where} your old self.")
        P += [t, ""]

    if g("rank_90d_ago") is not None:
        P += [f"**Rank.** {g('rank_90d_ago'):,} 90 days ago, {g('global_rank'):,} now; {g('play_count')} lifetime plays, "
              f"{g('plays_this_month')} this month.", ""]
    return "\n".join(P)
