"""
Map picking. Each role has a target predicted accuracy; maps score on how close they land, whether they
stress the right skill, how liked they are and how likely you are to finish them. Picks are greedy with a
penalty for repeating an artist or near-identical difficulty, so a list isn't one song's five remixes.
"""
import numpy as np
import pandas as pd

from .constants import COMFORT, PUSH

# role -> (target accuracy, spread, focus direction, mods)
ROLES = {
    "warmup": (COMFORT + 2.0, 3.0, 0, "NM"),
    "focus_aim": (sum(PUSH) / 2, 3.0, -1, "NM"),
    "focus_speed": (sum(PUSH) / 2, 3.0, 1, "NM"),
    "stretch": (PUSH[0] - 5.0, 3.0, 0, "NM"),
    "dt": (sum(PUSH) / 2, 4.0, 0, "DT"),
}
ROLE_TITLES = {"warmup": "Warm-up and accuracy reps", "focus_aim": "Aim focus (jumpier than your usual maps)",
               "focus_speed": "Speed focus (more tapping than your usual maps)", "stretch": "Stretch",
               "dt": "DT practice"}


def fallback_acc(ds):
    """Star-rating fit, for when the proper accuracy model didn't beat the mean on held-out maps."""
    d = ds.cur.dropna(subset=["stars", "acc"])
    if len(d) < 10:
        return lambda df: np.full(len(df), np.nan)
    b, a = np.polyfit(d["stars"], d["acc"], 1)
    return lambda df: np.clip(a + b * df["stars"].to_numpy(float), 0, 100)


def annotate(cands, predictor, ds):
    """Add predictions. Returns (table, method text)."""
    c = cands.copy()
    method = "accuracy model"
    if predictor is not None:
        c = predictor.predict(c)
    else:
        c["p_finish"] = np.nan
    if "acc_pred" not in c:
        c["acc_pred"] = fallback_acc(ds)(c)
        method = "star-rating fit (the accuracy model did not beat a plain average yet)"
    return c, method


def score(c, role, ratio_ref, challenge=0.0, max_len=240):
    target, spread, tilt_sign, mods = ROLES[role]
    d = c[c["mods"] == mods].dropna(subset=["acc_pred"]).copy()
    if d.empty:
        return d
    # A positive challenge means recent plans ran easy: aim lower on accuracy, so harder maps
    zone = np.exp(-0.5 * ((d["acc_pred"] - (target - challenge)) / spread) ** 2)
    pop = np.log1p(d["fav"]) / max(np.log1p(c["fav"].max()), 1e-9)
    finish = d["p_finish"].fillna(0.5)
    if tilt_sign:
        t = np.log((d["speed"] / d["aim"]).clip(lower=1e-3) / ratio_ref)
        tilt = 1 / (1 + np.exp(-tilt_sign * t / 0.12))
        total = 0.45 * zone + 0.3 * tilt + 0.15 * pop + 0.1 * finish
        total = total * np.where(d["length_s"] > max_len, 0.7, 1.0)
    else:
        total = 0.6 * zone + 0.25 * pop + 0.15 * finish
    d["score"] = total
    d["zone"] = zone
    return d


def pick(scored, n, used_sets, min_zone=0.2):
    """Greedy: best score, minus a penalty for artists and difficulty spots already picked."""
    if scored is None or scored.empty:
        return []
    d = scored[(scored["zone"] >= min_zone) & ~scored["set"].isin(used_sets)]
    d = d.sort_values("score", ascending=False).drop_duplicates("set").head(n * 8)
    rows = d.to_dict("records")
    chosen = []
    while rows and len(chosen) < n:
        def adjusted(r):
            same = sum(c["artist"] == r["artist"] for c in chosen)
            near = sum(abs(c["aim"] - r["aim"]) < 0.03 and abs(c["speed"] - r["speed"]) < 0.03 for c in chosen)
            return r["score"] - 0.12 * same - 0.05 * near
        best = max(rows, key=adjusted)
        chosen.append(best)
        rows.remove(best)
    used_sets.update(c["set"] for c in chosen)
    return chosen


def build_lists(c, cfg, ceilings, challenge=0.0):
    ratio = (ceilings or {}).get("ratio") or float((c["speed"] / c["aim"]).median())
    used = set()
    out = {}
    # Focus lists first, they need the most specific maps
    for role in ["focus_aim", "focus_speed", "dt", "warmup", "stretch"]:
        out[role] = pick(score(c, role, ratio, challenge, cfg.drill_max_length), cfg.per_role, used)
    return out


def revisit_queue(rall, predictor=None, now=None, limit=8):
    """
    Spaced repetition for maps you've already tried. Unfinished maps you got far on come back after a day;
    finished ones with weak accuracy come back after 2, 4, 8, 16 days as finishes pile up.
    """
    if rall is None or rall.empty:
        return []
    now = now or pd.Timestamp.now(tz="UTC")
    a = rall[~rall["nf"]].dropna(subset=["aim", "speed"])
    if a.empty:
        return []
    g = a.sort_values("date").groupby("map_key")
    t = g.agg(title=("title", "last"), artist=("artist", "last"), version=("diff", "last"), mods=("mods", "last"),
              id=("beatmap_id", "last"), set=("beatmapset_id", "last"), stars=("stars", "last"), aim=("aim", "last"),
              speed=("speed", "last"), ar=("ar", "last"), bpm=("eff_bpm", "last"), length_s=("length_s", "last"),
              tries=("passed", "size"), finishes=("passed", "sum"), best_comp=("completion", "max"),
              last=("date", "max"), eff_od=("eff_od", "last"))
    t["best_acc"] = a[a["passed"]].groupby("map_key")["acc"].max()
    interval = np.where(t["finishes"] == 0, 1, 2.0 ** np.minimum(t["finishes"], 4))
    t["due"] = t["last"] + pd.to_timedelta(interval, unit="D")
    near = (t["finishes"] == 0) & (t["tries"] >= 2) & (t["best_comp"] >= 0.4)
    weak = (t["finishes"] > 0) & (t["best_acc"].fillna(100) < 90)
    t = t[(near | weak) & (t["due"] <= now)].copy()
    if t.empty:
        return []
    if predictor is not None:
        t = predictor.predict(t)
    t["unfinished"] = t["finishes"] == 0
    t = t.sort_values(["unfinished", "best_comp"], ascending=False)
    if "acc_pred" in t:
        # Keep maps in reach; anything predicted far below the stretch zone is not ready yet
        t = t[t["acc_pred"] >= ROLES["stretch"][0] - 6]
    t = t.drop_duplicates("set").head(limit)
    return t.assign(id=t["id"].astype(int)).to_dict("records")
