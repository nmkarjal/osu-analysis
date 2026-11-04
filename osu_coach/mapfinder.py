"""Builds the pool of candidate maps from the sources in config, filtered per difficulty."""
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from .data import effective_ar, effective_od

POOL_DAYS = 14
GENRE_PAGES = 4
TEXT_PAGES = 2
PER_BAND = 25       # most favourited mapsets per 0.5* band that get difficulty attributes
PER_BAND_DT = 8
DT_STARS = 1.4      # DT raises star rating by very roughly this factor; only used to pick the search range


def star_window(ds):
    """Stars where your accuracy fit crosses 95% and 76%: the range worth searching."""
    d = ds.att if len(ds.att) >= 30 else ds.cur
    d = d.dropna(subset=["stars", "acc"])
    if len(d) < 10:
        return (2.0, 5.0)
    b, a = np.polyfit(d["stars"], d["acc"], 1)
    if b >= 0:
        q = d["stars"].quantile([0.1, 0.9])
        return (float(q.iloc[0]), float(q.iloc[1]))
    lo, hi = (95 - a) / b, (76 - a) / b
    return (float(np.clip(lo, 1.0, 9.0)), float(np.clip(hi, lo + 1.0, 10.0)))


def add_diffs(pool, sets, source):
    for bs in sets:
        for b in bs.get("beatmaps") or []:
            if b.get("mode") != "osu" or b.get("status") != "ranked":
                continue
            d = pool.setdefault(str(b["id"]), {
                "id": b["id"], "set": bs["id"], "artist": bs.get("artist"), "title": bs.get("title"),
                "version": b.get("version"), "stars_nomod": b.get("difficulty_rating") or 0, "ar": b.get("ar") or 0,
                "od": b.get("accuracy") or 0, "cs": b.get("cs") or 0, "bpm": b.get("bpm") or 0,
                "length": b.get("total_length") or 0, "checksum": b.get("checksum"),
                "fav": bs.get("favourite_count") or 0, "plays": bs.get("play_count") or 0,
                "sources": [], "no_dt": False})
            if source.name not in d["sources"]:
                d["sources"].append(source.name)
            d["no_dt"] = d["no_dt"] or source.no_dt


def source_key(cfg, window):
    raw = json.dumps([vars(s) for s in cfg.sources], sort_keys=True) + f"{window[0]:.1f}-{window[1]:.1f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def build_pool(cfg, client, window, refresh=False):
    path = cfg.data / "map_pool.json"
    key = source_key(cfg, window)
    if path.exists() and not refresh:
        saved = json.loads(path.read_text(encoding="utf-8"))
        fresh = time.time() - saved.get("built_at", 0) < POOL_DAYS * 86400
        if client.offline or (fresh and saved.get("key") == key):
            return saved["diffs"], saved.get("source_hits", {})
    if client.offline:
        return {}, {}
    lo = window[0] / DT_STARS
    hi = window[1]
    pool, hits = {}, {}
    for src in cfg.sources:
        rng_q = f"stars>={lo:.2f} stars<={hi:.2f}"
        base = {"g": src.genre} if src.genre is not None else {}
        if not src.queries and not src.artists:
            print(f"Search: {src.name}...")
            sets = client.search(dict(base, q=rng_q), GENRE_PAGES)
            add_diffs(pool, sets, src)
            hits[src.name] = len(sets)
        for q in src.queries:
            print(f"Search: {src.name}, '{q}'...")
            sets = client.search(dict(base, q=q), TEXT_PAGES)
            add_diffs(pool, sets, src)
            hits[f"{src.name}: {q}"] = len(sets)
        for a in src.artists:
            print(f"Search: artist {a}...")
            rx = re.compile(rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])")
            sets = [bs for bs in client.search(dict(base, q=a), TEXT_PAGES) if rx.search((bs.get("artist") or "").lower())]
            add_diffs(pool, sets, src)
            hits[a] = len(sets)
    cfg.data.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"built_at": time.time(), "key": key, "window": window, "diffs": pool, "source_hits": hits}),
                    encoding="utf-8")
    return pool, hits


def _shortlist(rows, lo, hi, per_band):
    """Top mapsets by favourites inside each 0.5* band, so every difficulty level gets candidates."""
    out = []
    edges = np.arange(np.floor(lo * 2) / 2, hi + 0.5, 0.5)
    for a, b in zip(edges[:-1], edges[1:]):
        band = [d for d in rows if a <= d["stars_nomod"] < b and lo <= d["stars_nomod"] <= hi]
        seen = set()
        for d in sorted(band, key=lambda d: (-d["fav"], -d["plays"])):
            if d["set"] not in seen:
                seen.add(d["set"])
                out.append(d)
            if len(seen) >= per_band:
                break
    return out


def candidates(cfg, client, pool, window, exclude_ids):
    """Attributes for a shortlist of nomod and DT candidates. Returns one row per (map, mods)."""
    rows = [d for d in pool.values() if d["id"] not in exclude_ids]
    nm = _shortlist([d for d in rows if d["ar"] >= cfg.min_ar], window[0], window[1], PER_BAND)
    dt = _shortlist([d for d in rows if d["ar"] >= cfg.dt_min_ar and d["bpm"] <= cfg.dt_max_bpm and not d["no_dt"]],
                    window[0] / DT_STARS, window[1] / DT_STARS, PER_BAND_DT)
    print(f"Difficulty attributes for {len(nm)} nomod and {len(dt)} DT candidates (cached ones are free)...")
    out = []
    try:
        for mods, group in [("NM", nm), ("DT", dt)]:
            for d in group:
                a = client.attributes(d["id"], [] if mods == "NM" else ["DT"], d["checksum"])
                if not a.get("aim_difficulty"):
                    continue
                rate = 1.5 if mods == "DT" else 1.0
                out.append(dict(d, mods=mods, stars=a.get("star_rating"), aim=a["aim_difficulty"],
                                speed=a.get("speed_difficulty"), eff_bpm=d["bpm"] * rate, length_s=d["length"] / rate,
                                eff_ar=float(effective_ar(d["ar"], rate)), eff_od=float(effective_od(d["od"], rate))))
    finally:
        client.cache.save()
    return pd.DataFrame(out)
