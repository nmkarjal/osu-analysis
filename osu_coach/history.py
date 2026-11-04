"""Run history: snapshots, runs.csv and the comparison with the previous run."""
import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from .data import read_csv

KEYS = [("pp", "pp", 1), ("global_rank", "global rank", 0), ("hit_accuracy", "profile acc %", 2),
        ("play_count", "play count", 0), ("n_cur_passes", "passes analysed", 0),
        ("acc_per_sd_aim", "acc per SD aim", 2), ("acc_per_sd_speed", "acc per SD speed", 2),
        ("miss_per_sd_aim", "misses per SD aim", 2), ("ok_sd_speed", "100s per SD speed", 2),
        ("bpm_break", "BPM breakpoint", 0), ("acc_3_5", "acc at 3-5*", 2), ("n_2_3", "passes at 2-3*", 0),
        ("plan_comfort_stars_top", "comfort band top (stars)", 2), ("plan_push_stars_top", "push band top (stars)", 2),
        ("share_timing-heavy", "timing-heavy %", 0), ("share_miss-heavy", "miss-heavy %", 0),
        ("fail_rate_now", "fail rate this pull %", 0), ("n_recent_now", "attempts this pull", 0),
        ("attempt_median_stars", "median attempt stars", 2), ("ceil_aim", "aim ceiling", 2),
        ("ceil_speed", "speed ceiling", 2), ("adj_miss_last", "misses/100 last session, equal difficulty", 2),
        ("adj_ok_last", "100s/100 last session, equal difficulty", 2)]


def previous_metrics(hist, exported_at):
    snaps = sorted(p for p in hist.iterdir() if p.is_dir() and (p / "metrics.json").exists())
    for p in reversed(snaps):
        m = json.loads((p / "metrics.json").read_text(encoding="utf-8"))
        if m.get("exported_at") != exported_at:
            m["_dir"] = p.name
            return m
    return None


def compare(prev, M, rall):
    if not prev:
        return "## Since last run\n\nFirst run; nothing to compare yet.\n"
    md = [f"## Since last run ({prev['_dir']})", "", "| metric | previous | now | change |", "|---|---|---|---|"]
    for k, lab, nd in KEYS:
        a, b = prev.get(k), M.get(k)
        if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
            continue
        md.append(f"| {lab} | {a:.{nd}f} | {b:.{nd}f} | {b - a:+.{nd}f} |")
    # Quits/restarts are never submitted: playcount gain minus submitted attempts in the same window
    try:
        t0, t1 = pd.Timestamp(prev["exported_at"]), pd.Timestamp(M["exported_at"])
        if t0.tzinfo is None:
            t0, t1 = t0.tz_localize("UTC"), t1.tz_localize("UTC")
        gain = M["play_count"] - prev["play_count"]
        if t1 - t0 <= pd.Timedelta(hours=24) and not rall.empty:
            sub = int(((rall["date"] > t0) & (rall["date"] <= t1)).sum())
            M["unsubmitted_est"] = gain - sub
            if gain <= 0:
                md.append(f"\nBetween runs: playcount +{gain}.")
            elif gain - sub <= max(1, 0.05 * gain):
                md.append(f"\nBetween runs: playcount +{gain}, submitted attempts {sub}. Every play was submitted, so your "
                          "client submits quits/restarts as failed scores (lazer does): read 'fail' as 'did not finish'.")
            else:
                md.append(f"\nBetween runs: playcount +{gain}, submitted attempts {sub}, so about {gain - sub} plays were "
                          f"quit/restarted without being submitted ({100 * (gain - sub) / gain:.0f}% of plays).")
        else:
            md.append(f"\nPlaycount +{gain} since the previous run. Runs were more than 24h apart, so plays in between "
                      "were partly missed; run right after each session for a full log.")
    except (KeyError, ValueError, TypeError):
        pass
    md.append("")
    return "\n".join(md)


def snap_run(cfg, M):
    stamp = pd.Timestamp(M["exported_at"]).strftime("%Y-%m-%d_%H%M%S") if M.get("exported_at") else \
        datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d = cfg.hist / stamp
    d.mkdir(parents=True, exist_ok=True)
    for name in ["top.csv", "recent.csv", "export.json"]:
        if (cfg.data / name).exists():
            shutil.copy2(cfg.data / name, d / name)
    for p in cfg.out.glob("*.png"):
        shutil.copy2(p, d / p.name)
    for name in ["report.md", "metrics.json", "plan.md"]:
        if (cfg.out / name).exists():
            shutil.copy2(cfg.out / name, d / name)
    row = {k: M.get(k) for k in ["run_at", "exported_at"] + [k for k, _, _ in KEYS] + ["verdict"]}
    runs = cfg.hist / "runs.csv"
    df = read_csv(runs)
    if not df.empty:
        df = df[df["exported_at"] != row["exported_at"]]
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(runs, index=False)
