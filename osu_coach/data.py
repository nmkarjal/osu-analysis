"""Loading and cleaning: raw pull files -> tidy tables with derived columns."""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import ERA_NEW, ERA_OLD, MIN_HITS, SESSION_GAP_MIN

NUM = ["score_id", "acc", "pp", "combo", "map_max_combo", "n300", "n100", "n50", "miss", "stars_nomod",
       "stars", "aim", "speed", "speed_notes", "slider_factor", "bpm_nomod", "length_s_nomod",
       "drain_s_nomod", "ar", "od", "cs", "hp", "circles", "sliders", "spinners", "beatmap_id"]


def read_csv(p):
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def split_mods(m):
    m = str(m) if m is not None else "NM"
    return {m[i:i + 2] for i in range(0, len(m), 2)}


def _ms_to_ar(ms):
    return np.where(ms > 1200, (1800 - ms) / 120, 5 + (1200 - ms) / 150)


def effective_ar(ar, rate, hr=False, ez=False):
    """AR after mods. HR x1.4 (cap 10), EZ x0.5, then the approach time scales with the clock rate."""
    ar = np.asarray(ar, float)
    ar = np.where(hr, np.minimum(ar * 1.4, 10), np.where(ez, ar * 0.5, ar))
    ms = np.where(ar < 5, 1800 - 120 * ar, 1200 - 150 * (ar - 5)) / rate
    return _ms_to_ar(ms)


def effective_od(od, rate, hr=False, ez=False):
    """OD after mods, from the 300 hit window (80 - 6*OD ms)."""
    od = np.asarray(od, float)
    od = np.where(hr, np.minimum(od * 1.4, 10), np.where(ez, od * 0.5, od))
    return (80 - (80 - 6 * od) / rate) / 6


def prep(df, source, era_ts):
    if df.empty:
        return df
    df = df.copy()
    for c in NUM:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["passed"] = df["passed"].astype(str).str.lower().isin(["true", "1", "1.0"])
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601", errors="coerce")
    modset = df["mods"].fillna("NM").astype(str).map(split_mods)
    df["dt"] = modset.map(lambda s: bool(s & {"DT", "NC"}))
    df["ht"] = modset.map(lambda s: bool(s & {"HT", "DC"}))
    df["hr"] = modset.map(lambda s: "HR" in s)
    df["hd"] = modset.map(lambda s: "HD" in s)
    df["ez"] = modset.map(lambda s: "EZ" in s)
    df["nf"] = modset.map(lambda s: "NF" in s)
    rate = np.where(df["dt"], 1.5, np.where(df["ht"], 0.75, 1.0))
    df["eff_bpm"] = df["bpm_nomod"] * rate
    df["length_s"] = df["length_s_nomod"] / rate
    df["eff_ar"] = effective_ar(df["ar"], rate, df["hr"], df["ez"])
    df["eff_od"] = effective_od(df["od"], rate, df["hr"], df["ez"])
    df["eff_cs"] = np.where(df["hr"], np.minimum(df["cs"] * 1.3, 10), np.where(df["ez"], df["cs"] * 0.5, df["cs"]))
    for c in ["n300", "n100", "n50", "miss", "circles", "sliders", "spinners"]:
        df[c] = df[c].fillna(0)
    hits = df["n300"] + df["n100"] + df["n50"] + df["miss"]
    objects = df["circles"] + df["sliders"] + df["spinners"]
    df["hits"] = hits
    obj = objects.where(objects > 0)
    df["completion"] = hits / obj
    h = hits.where(hits > 0)
    df["miss_pct"] = 100 * df["miss"] / h
    df["ok_pct"] = 100 * (df["n100"] + df["n50"]) / h
    df["n50_pct"] = 100 * df["n50"] / h
    df["ratio"] = df["speed"] / df["aim"]
    df["log2_len"] = np.log2(df["length_s"].clip(lower=10))
    df["slider_share"] = df["sliders"] / obj
    df["stream_share"] = df["speed_notes"] / obj
    df["era"] = np.where(df["date"] < era_ts, ERA_OLD, ERA_NEW)
    df["source"] = source
    df["err_type"] = df.apply(classify, axis=1)
    return add_uid(df)


def classify(r):
    """Error profile of one play. Thresholds are fixed so runs stay comparable."""
    oks = r["n100"] + r["n50"]
    if r["hits"] <= 0:
        return "n/a"
    if r["miss"] <= 1 and r["ok_pct"] < 2:
        return "clean"
    if r["miss"] >= 2 and oks / r["miss"] < 4:
        return "miss-heavy"
    if r["miss"] <= 1 or oks / r["miss"] >= 12:
        return "timing-heavy"
    return "mixed"


def add_uid(df):
    """Failed scores come back with id 0, so key those on map + timestamp instead."""
    sid = pd.to_numeric(df["score_id"], errors="coerce")
    alt = "f" + df["beatmap_id"].astype(str) + "_" + df["date"].astype(str)
    df["uid"] = np.where(sid > 0, sid.fillna(0).astype("int64").astype(str), alt)
    return df


def merge_recent(hist, recent_raw):
    """Append this pull's recent plays to the cumulative log so fail/retry data grows run over run."""
    path = hist / "recent_all.csv"
    old = read_csv(path)
    allr = pd.concat([old, recent_raw], ignore_index=True) if not old.empty else recent_raw.copy()
    if allr.empty:
        return allr
    allr = add_uid(allr).drop_duplicates("uid", keep="last").sort_values("date")
    hist.mkdir(parents=True, exist_ok=True)
    allr.to_csv(path, index=False)
    return allr


def add_sessions(a, gap_min=SESSION_GAP_MIN):
    """Session id, minutes into the session, attempt number in the session, try number in a same-map streak."""
    a = a.sort_values("date").copy()
    if a.empty:
        return a
    gap = a["date"].diff().dt.total_seconds().div(60)
    a["session"] = (gap.isna() | (gap > gap_min)).cumsum()
    a["session_min"] = (a["date"] - a.groupby("session")["date"].transform("min")).dt.total_seconds().div(60)
    a["session_hours"] = a["session_min"] / 60
    a["session_try"] = a.groupby("session").cumcount() + 1
    a["warmup"] = a["session_try"] <= 5
    key = a["beatmap_id"].astype("Int64").astype(str) + " " + a["mods"].astype(str)
    a["map_key"] = key
    a["streak"] = (key != key.shift()).cumsum()
    a["streak_try"] = a.groupby("streak").cumcount() + 1
    return a


@dataclass
class Dataset:
    export: dict
    top: pd.DataFrame        # top plays
    recent_now: pd.DataFrame  # attempts in this pull
    rall: pd.DataFrame       # every attempt ever logged
    passes: pd.DataFrame     # top + logged passes, one row per score
    cur: pd.DataFrame        # passes in the current era
    att: pd.DataFrame        # logged attempts in the current era with enough objects to measure
    cur_label: str
    since: str               # e.g. "since 2023-01"
    old_label: str           # e.g. "before 2023-01"


def load_dataset(cfg):
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.hist.mkdir(parents=True, exist_ok=True)
    path = cfg.data / "export.json"
    if not path.exists():
        raise SystemExit("No data yet. Run: python -m osu_coach pull")
    exp = json.loads(path.read_text(encoding="utf-8"))
    era = cfg.era_ts
    top = prep(read_csv(cfg.data / "top.csv"), "top", era)
    recent_raw = read_csv(cfg.data / "recent.csv")
    recent_now = prep(recent_raw, "recent", era)
    rall = prep(merge_recent(cfg.hist, recent_raw), "recent", era)
    if not rall.empty:
        rall = add_sessions(rall)
    parts = [d for d in [top, rall[rall["passed"]] if not rall.empty else rall] if not d.empty]
    if not parts:
        raise SystemExit("The export has no plays yet. Play a bit and pull again.")
    passes = pd.concat(parts, ignore_index=True).drop_duplicates("uid", keep="first")
    passes = passes[passes["hits"] > 0]
    since, old_label = f"since {era:%Y-%m}", f"before {era:%Y-%m}"
    cur = passes[passes["era"] == ERA_NEW]
    cur_label = f"passed plays {since}"
    if len(cur) < 20:
        cur, cur_label = passes, f"all passed plays (too few {since})"
    att = rall[(rall["era"] == ERA_NEW) & (rall["hits"] >= MIN_HITS)] if not rall.empty else rall
    return Dataset(exp, top, recent_now, rall, passes, cur, att, cur_label, since, old_label)
