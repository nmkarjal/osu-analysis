"""
Weakness scan. Each check asks one question with difficulty held fixed, for example: at equal aim and
speed, do you miss more on high-AR maps? All p-values are map-clustered and corrected together
(Benjamini-Hochberg), so running many checks doesn't turn noise into advice. A check that finds nothing
is listed as checked, not hidden.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import FDR
from .stats import bh_adjust, ols_cluster, wilson

MAX_R2 = 0.75  # a factor explained this well by aim and speed can't be told apart from difficulty

# Accuracy points lost per unit: one miss per 100 objects costs 1 pp, one 100 costs about 0.7 pp
PP = {"acc": 1.0, "miss_pct": 1.0, "ok_pct": 0.7, "completion": 0.0}


@dataclass
class Check:
    key: str
    x: str                  # factor under test
    y: str                  # acc, miss_pct, ok_pct
    bad: int                # sign of x's coefficient that means 'worse'; 0 = either direction can be bad
    label: str              # "AR" -> used in the sentence
    table: str = "att"      # att = logged attempts, full = finished plays (top + logged)
    # Squared terms matter: errors curve upward with difficulty, and factors that track difficulty (AR, OD)
    # would otherwise pick that curve up and look like effects of their own
    controls: tuple = ("aim", "speed", "aim_sq", "speed_sq")
    binary: bool = False
    min_true: int = 8       # binary factors need this many plays on each side
    min_sessions: int = 0   # within-session effects need several sessions to mean anything
    phrase: str = ""        # how the sentence starts, when "Higher <label>" reads wrong


CHECKS = [
    Check("aim_misses", "aim", "miss_pct", 1, "aim difficulty", controls=("speed",)),
    Check("speed_timing", "speed", "ok_pct", 1, "speed difficulty", controls=("aim",)),
    Check("aim_timing", "aim", "ok_pct", 1, "aim difficulty", controls=("speed",)),
    Check("speed_misses", "speed", "miss_pct", 1, "speed difficulty", controls=("aim",)),
    Check("stamina", "log2_len", "acc", -1, "map length", table="full", phrase="Each doubling of map length"),
    Check("ar_reading", "eff_ar", "miss_pct", 0, "AR"),
    Check("od_timing", "eff_od", "ok_pct", 1, "OD"),
    Check("cs_precision", "eff_cs", "miss_pct", 1, "circle size"),
    Check("sliders", "slider_share", "miss_pct", 0, "slider share"),
    Check("streams", "stream_share", "ok_pct", 1, "stream/burst density"),
    Check("dt", "dt", "acc", -1, "DT", binary=True),
    Check("hr", "hr", "acc", -1, "HR", binary=True),
    Check("hd", "hd", "acc", -1, "HD", binary=True),
    Check("fatigue", "session_hours", "acc", -1, "time into a session", min_sessions=5,
          phrase="Each hour into a session"),
    Check("warmup", "warmup", "acc", -1, "the first 5 attempts of a session", binary=True, min_sessions=5,
          phrase="In the first 5 attempts of a session"),
]

CORE = {"aim_misses", "speed_timing", "aim_timing", "speed_misses"}  # these test aim and speed themselves

# Why it matters + what to do. Advice is ordinary osu! practice wisdom; the data only decides which ones show up.
DRILLS = {
    "aim_misses": ("aim", "Jump control", "Jump-heavy maps inside your aim focus range. Judge by miss count, not accuracy, "
                                          "and replay the section with the first miss until it goes clean."),
    "aim_timing": ("aim", "Clean jumps", "Jump maps at comfort level, aiming for 300s on landing rather than just hitting."),
    "speed_timing": ("speed", "Tapping accuracy", "Bursts and short streams inside your speed focus range. Watch the 100s "
                                                  "count; move up only when it falls."),
    "speed_misses": ("speed", "Holding streams", "Stream maps slightly under your speed ceiling. Finish the stream even if "
                                                 "it gets ugly, then replay it."),
    "stamina": ("stamina", "Long maps", "One map over 3 minutes at comfort stars per session, played to the end."),
    "ar_reading": ("reading", "Reading", "Comfort-level maps at the {side} AR end, so reading is the only new thing."),
    "od_timing": ("speed", "Hit windows", "Comfort-level maps with high OD. Slow down and hit on the beat."),
    "cs_precision": ("aim", "Small circles", "Comfort-level maps with high CS, focused on landing in the middle of the circle."),
    "sliders": ("reading", "Sliders", "Slider-heavy maps at comfort stars; follow slider ends instead of dropping them."),
    "streams": ("speed", "Stream density", "Stream-heavy maps at comfort stars, then raise density before star rating."),
    "dt": ("speed", "DT adaptation", "Easy maps with DT, so the speed-up is the only new thing."),
    "hr": ("reading", "HR adaptation", "Comfort-level maps with HR, one step below your normal stars."),
    "hd": ("reading", "HD adaptation", "Comfort-level maps with HD at a BPM you read easily."),
    "fatigue": ("habit", "Shorter sessions", "Your accuracy drops as sessions go on: put the hard block early and stop sooner."),
    "warmup": ("habit", "Warm-up", "Your first attempts are measurably worse: start every session with easy maps."),
    "overreach": ("habit", "Play easier", "Most attempts don't finish. Spend most of a session on maps you can complete; "
                                          "finished maps teach the whole map, restarts only teach the intro."),
    "early_quit": ("habit", "Finish the map", "Many attempts stop very early. Use NF on practice maps or commit to the first "
                                              "minute before restarting."),
    "grinding": ("habit", "Stop grinding", "Long fail streaks on one map. After 3 fails in a row move on and put the map "
                                           "in the revisit queue."),
    "retry_no_gain": ("habit", "Retry less", "Retrying a map right away isn't getting you further. Come back another day instead."),
}


@dataclass
class Finding:
    key: str
    kind: str               # weakness, strength or habit
    text: str               # one readable sentence, written from the numbers
    cost_pp: float = 0.0    # conservative accuracy cost per step (end of the interval nearest zero)
    effect: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    p: float | None = None
    q: float | None = None
    n: int = 0
    area: str = ""
    drill: str = ""
    how: str = ""
    extra: dict = field(default_factory=dict)


def _full_table(ds):
    return ds.cur[ds.cur["passed"]].copy() if len(ds.cur) else ds.cur


def explained_by_difficulty(t, x, controls):
    """R^2 of the factor on the controls."""
    d = t[[x] + list(controls)].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(d) < 10 or d[x].std() == 0:
        return 0.0
    X = np.column_stack([np.ones(len(d))] + [d[c] for c in controls])
    y = d[x].to_numpy()
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())))


def run_checks(att, full):
    """
    Run every regression check. Returns (results, skipped for too little data, collinear with difficulty).
    """
    rows, skipped, collinear = [], [], []
    tables = {"att": att.copy() if att is not None else pd.DataFrame(), "full": full.copy()}
    for t in tables.values():
        if not t.empty:
            t["cluster"] = t["beatmap_id"]
            t["aim_sq"], t["speed_sq"] = t["aim"] ** 2, t["speed"] ** 2
    for c in CHECKS:
        t = tables[c.table]
        if t.empty or c.x not in t or c.y not in t:
            skipped.append(c.key)
            continue
        t = t.copy()
        if c.min_sessions and ("session" not in t or t["session"].nunique() < c.min_sessions):
            skipped.append(c.key)
            continue
        if c.binary:
            t[c.x] = t[c.x].astype(float)
            if t[c.x].sum() < c.min_true or (1 - t[c.x]).sum() < c.min_true:
                skipped.append(c.key)
                continue
        if c.key not in CORE and explained_by_difficulty(t, c.x, c.controls) > MAX_R2:
            collinear.append(c.key)
            continue
        fit = ols_cluster(t, c.y, list(c.controls) + [c.x], "cluster")
        if fit is None:
            skipped.append(c.key)
            continue
        co = fit["coef"][c.x]
        step = 1.0 if c.binary or c.key == "stamina" or c.key == "fatigue" else co["sd"]
        rows.append({"check": c, "b": co["b"] * step, "lo": co["lo"] * step, "hi": co["hi"] * step, "p": co["p"],
                     "n": fit["n"], "step": step})
    return rows, skipped, collinear


def _sentence(c, r):
    unit = {"acc": "pp accuracy", "miss_pct": "misses per 100 objects", "ok_pct": "100s/50s per 100 objects"}[c.y]
    step = "" if c.binary or c.key in ("stamina", "fatigue") else " per typical step"
    ci = f"95% CI {r['lo']:+.1f} to {r['hi']:+.1f}"
    if c.phrase:
        return f"{c.phrase}, difficulty held fixed: {r['b']:+.1f} {unit} ({ci})."
    if c.binary:
        return f"With {c.label}, difficulty held fixed: {r['b']:+.1f} {unit} ({ci})."
    return f"Higher {c.label}, other difficulty held fixed: {r['b']:+.1f} {unit}{step} ({ci})."


def behaviour(att_all, predictor=None):
    """Habits. These describe what you do rather than test an effect, so they use plain thresholds."""
    out = []
    a = att_all
    if a is None or a.empty or len(a) < 30:
        return out
    last = a.sort_values("date").tail(150)
    real = last[~last["nf"]]
    n, k = len(real), int(real["passed"].sum())
    lo, hi = wilson(k, n)
    if n >= 30 and hi < 0.5:
        out.append(Finding("overreach", "habit", f"You finish {100 * k / n:.0f}% of attempts ({k} of your last {n}, "
                           f"95% CI {100 * lo:.0f}-{100 * hi:.0f}%). Median attempt {real['stars'].median():.2f}*.",
                           n=n, extra={"finish": k / n}))
    fails = last[~last["passed"]]
    if len(fails) >= 20:
        early = float((fails["completion"] < 0.25).mean())
        if early > 0.35:
            out.append(Finding("early_quit", "habit", f"{100 * early:.0f}% of unfinished attempts stop in the first quarter "
                               "of the map.", n=len(fails)))
    st = a.groupby("streak").agg(n=("passed", "size"), ok=("passed", "sum"), title=("title", "first"), diff=("diff", "first"))
    bad = st[(st["n"] >= 6) & (st["ok"] == 0)]
    if len(bad):
        w = bad.sort_values("n").iloc[-1]
        word = "streak" if len(bad) == 1 else "streaks"
        out.append(Finding("grinding", "habit", f"{len(bad)} {word} of 6+ tries in a row without a finish; longest "
                           f"{int(w['n'])} on {w['title']} [{w['diff']}].", n=int(bad["n"].sum())))
    # Do immediate retries get further on the same map? Within-map slope of completion on try number
    multi = a[a.groupby("streak")["passed"].transform("size") >= 3].dropna(subset=["completion"]).copy()
    if len(multi) >= 30 and multi["streak"].nunique() >= 8:
        multi["c_dm"] = multi["completion"] - multi.groupby("streak")["completion"].transform("mean")
        multi["t_dm"] = multi["streak_try"] - multi.groupby("streak")["streak_try"].transform("mean")
        fit = ols_cluster(multi, "c_dm", ["t_dm"], "beatmap_id")
        if fit:
            co = fit["coef"]["t_dm"]
            gain = 100 * co["b"]
            if co["hi"] * 100 < 2:
                out.append(Finding("retry_no_gain", "habit", f"Retrying a map straight away gets you {gain:+.1f}% of the map "
                                   f"further per try (95% CI {100 * co['lo']:+.1f} to {100 * co['hi']:+.1f}).",
                                   n=fit["n"], p=co["p"]))
            elif co["p"] < 0.05 and co["b"] > 0:
                out.append(Finding("retry_gain", "strength", f"Retrying pays off: each try gets {gain:+.1f}% of the map further "
                                   f"(95% CI {100 * co['lo']:+.1f} to {100 * co['hi']:+.1f}).", n=fit["n"], p=co["p"]))
    return out


def scan(ds, predictor=None):
    """All findings, worst first, plus what was checked and came back clean or lacked data."""
    full = _full_table(ds)
    rows, skipped, collinear = run_checks(ds.att, full)
    qs = bh_adjust([r["p"] for r in rows])
    weak, strong, clean = [], [], []
    for r, q in zip(rows, qs):
        c = r["check"]
        worse = (r["b"] * c.bad > 0) if c.bad else True
        if q > FDR:
            clean.append(c.key)
            continue
        near_zero = min(abs(r["lo"]), abs(r["hi"]))
        cost = near_zero * PP.get(c.y, 1.0)
        f = Finding(c.key, "weakness" if worse else "strength", _sentence(c, r), cost_pp=cost, effect=r["b"], lo=r["lo"],
                    hi=r["hi"], p=r["p"], q=float(q), n=r["n"])
        if c.bad == 0:
            # A positive slope means the high end is the hard side
            f.extra["side"] = "high" if r["b"] > 0 else "low"
        (weak if worse else strong).append(f)
    habits = behaviour(ds.rall if not ds.rall.empty else None, predictor)
    strong += [h for h in habits if h.kind == "strength"]
    habits = [h for h in habits if h.kind == "habit"]
    for f in weak + habits:
        if f.key in DRILLS:
            f.area, f.drill, f.how = DRILLS[f.key]
            f.how = f.how.replace("{side}", f.extra.get("side", ""))
    weak.sort(key=lambda f: -f.cost_pp)
    return {"findings": weak, "habits": habits, "strengths": strong, "clean": clean, "skipped": skipped,
            "collinear": collinear, "n_tests": len(rows)}


def to_markdown(res):
    md = ["## What's holding you back", "",
          f"{res['n_tests']} factors tested with aim and speed held fixed, p-values clustered by map and corrected for "
          f"testing many things at once (false discovery rate {100 * FDR:.0f}%). Ranked by the cautious end of each interval.", ""]
    if res["findings"]:
        md += ["| # | what | evidence | drill |", "|---|---|---|---|"]
        for i, f in enumerate(res["findings"], 1):
            q = "q<0.001" if f.q < 0.001 else f"q={f.q:.3f}"
            md.append(f"| {i} | {f.drill or f.key} | {f.text} {q}. | {f.how} |")
    else:
        md.append("Nothing passed the test. That means no clear single weakness yet, not that there isn't one.")
    if res["habits"]:
        md += ["", "Habits (what you do, not a tested effect):", "", "| habit | what the data shows | suggestion |", "|---|---|---|"]
        md += [f"| {f.drill} | {f.text} | {f.how} |" for f in res["habits"]]
    if res["strengths"]:
        md += ["", "Going better than expected:", ""] + [f"- {f.text}" for f in res["strengths"]]
    names = {c.key: c.label for c in CHECKS}
    if res["clean"]:
        md += ["", "Checked, no clear effect: " + ", ".join(names.get(k, k) + f" ({k})" for k in res["clean"]) + "."]
    if res["collinear"]:
        md += ["", "On your maps these move almost in step with aim and speed, so they can't be judged separately: "
               + ", ".join(names.get(k, k) for k in res["collinear"]) + "."]
    if res["skipped"]:
        md += ["", "Not enough data to check: " + ", ".join(names.get(k, k) for k in res["skipped"]) + "."]
    md.append("")
    return "\n".join(md)
