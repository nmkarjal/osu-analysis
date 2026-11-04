"""
A made-up player, so anyone can try the whole thing without API keys, and so tests have data.
The player has a known aim and speed level, misses more on aim and hits more 100s on speed, and fails a lot
on maps that are too hard, which is what the analysis should be able to find.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .api import attr_key

AIM_SKILL, SPEED_SKILL = 2.35, 1.5


def make_map(rng, i):
    stars = rng.uniform(2.0, 5.6)
    tilt = rng.normal(0, 0.12)
    aim = stars * (0.56 + tilt * 0.3)
    speed = stars * (0.36 - tilt * 0.35)
    objs = int(rng.uniform(120, 900))
    circles = int(objs * rng.uniform(0.4, 0.7))
    return {"id": 1000 + i, "set": 500 + i, "artist": f"Artist {i % 37}", "title": f"Song {i}", "version": "Insane",
            "stars": stars, "aim": aim, "speed": speed, "ar": float(np.clip(5.5 + stars * 0.9, 6, 10)),
            "od": float(np.clip(4 + stars * 0.85, 4, 10)), "cs": 4.0, "bpm": float(rng.uniform(110, 230)),
            "length": int(objs / rng.uniform(1.6, 3.0)), "circles": circles, "sliders": objs - circles - 1, "spinners": 1,
            "speed_notes": objs * rng.uniform(0.2, 0.6), "fav": int(rng.lognormal(5, 1.5)), "plays": int(rng.lognormal(10, 1))}


def play(rng, m, when, skill_gain):
    a_gap = m["aim"] - (AIM_SKILL + skill_gain)
    s_gap = m["speed"] - (SPEED_SKILL + skill_gain * 0.6)
    miss_rate = float(np.clip(0.03 + 0.05 * a_gap + rng.normal(0, 0.008), 0.0, 0.5))
    ok_rate = float(np.clip(0.15 + 0.14 * s_gap + rng.normal(0, 0.02), 0.01, 0.6))
    total = m["circles"] + m["sliders"] + m["spinners"]
    finish_p = 1 / (1 + np.exp(1.1 * (a_gap + s_gap) + 0.6))
    passed = bool(rng.random() < finish_p)
    frac = 1.0 if passed else float(np.clip(rng.beta(1.6, 2.2), 0.05, 0.95))
    hits = max(int(total * frac), 1)
    miss = int(rng.binomial(hits, miss_rate))
    n100 = int(rng.binomial(hits - miss, ok_rate * 0.9))
    n50 = int(rng.binomial(hits - miss - n100, ok_rate * 0.1))
    n300 = hits - miss - n100 - n50
    acc = 100 * (300 * n300 + 100 * n100 + 50 * n50) / (300 * hits)
    return {"score_id": 0 if not passed else int(rng.integers(10 ** 9, 10 ** 10)), "date": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "artist": m["artist"], "title": m["title"], "diff": m["version"], "mapper": "demo", "beatmap_id": m["id"],
            "beatmapset_id": m["set"], "mods": "NM", "rank": "A" if passed else "F", "passed": passed, "acc": round(acc, 2),
            "pp": 40.0 if passed else None, "combo": 100, "map_max_combo": total, "n300": n300, "n100": n100, "n50": n50,
            "miss": miss, "stars_nomod": m["stars"], "stars": m["stars"], "aim": m["aim"], "speed": m["speed"],
            "speed_notes": m["speed_notes"], "slider_factor": 0.95, "bpm_nomod": m["bpm"], "length_s_nomod": m["length"],
            "drain_s_nomod": m["length"], "ar": m["ar"], "od": m["od"], "cs": m["cs"], "hp": 5.0, "circles": m["circles"],
            "sliders": m["sliders"], "spinners": m["spinners"]}


def generate(root, seed=7, sessions=10, per_session=45):
    """Write a config and a full data folder under `root`. Same file formats as a real pull."""
    root = Path(root)
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    maps = [make_map(rng, i) for i in range(420)]
    now = pd.Timestamp.now(tz="UTC").floor("s")
    played, rows = maps[:220], []
    for s in range(sessions):
        t = now - pd.Timedelta(days=2 * (sessions - s)) + pd.Timedelta(hours=float(rng.uniform(0, 3)))
        gain = 0.015 * s
        i = 0
        while i < per_session:
            m = played[int(rng.integers(0, len(played)))]
            for _ in range(int(rng.integers(1, 4))):
                t += pd.Timedelta(seconds=float(rng.uniform(40, 200)))
                rows.append(play(rng, m, t, gain))
                i += 1
    recent = pd.DataFrame(rows)
    top = recent[recent["passed"]].sort_values("acc", ascending=False).drop_duplicates("beatmap_id").head(100)
    old = recent[recent["passed"]].sample(40, random_state=seed).copy()
    old["score_id"] = rng.integers(10 ** 8, 10 ** 9, len(old))
    old["date"] = (now - pd.Timedelta(days=900)).strftime("%Y-%m-%dT%H:%M:%SZ")
    top = pd.concat([top, old], ignore_index=True)
    # Older sessions go straight to the cumulative log, like earlier pulls would have
    hist = root / "osu_analysis" / "history"
    hist.mkdir(parents=True, exist_ok=True)
    recent.to_csv(hist / "recent_all.csv", index=False)
    recent.tail(100).to_csv(data / "recent.csv", index=False)
    top.to_csv(data / "top.csv", index=False)

    months = [{"start_date": f"{d:%Y-%m-01}", "count": int(rng.integers(40, 300))}
              for d in pd.date_range(end=now.tz_localize(None), periods=14, freq="MS")]
    user = {"username": "demo player", "id": 1, "statistics": {"pp": 1234.5, "global_rank": 500000, "hit_accuracy": 88.2,
            "play_count": 4000, "play_time": 400000}, "monthly_playcounts": months,
            "rank_history": {"data": [int(x) for x in np.linspace(700000, 500000, 90)]}}
    (data / "export.json").write_text(json.dumps({
        "exported_at": now.strftime("%Y-%m-%dT%H:%M:%S+0000"), "user": user, "best": [], "recent": [], "most_played": [],
        "most_played_attrs": {}, "lookups": []}), encoding="utf-8")

    diffs, cache = {}, {}
    t0 = time.time()
    for m in maps:
        cs = f"demo{m['id']}"
        diffs[str(m["id"])] = {"id": m["id"], "set": m["set"], "artist": m["artist"], "title": m["title"], "version": m["version"],
                               "stars_nomod": m["stars"], "ar": m["ar"], "od": m["od"], "cs": m["cs"], "bpm": m["bpm"],
                               "length": m["length"], "checksum": cs, "fav": m["fav"], "plays": m["plays"],
                               "sources": ["demo"], "no_dt": False}
        cache[attr_key(m["id"], [], cs)] = {"t": t0, "a": {"star_rating": m["stars"], "aim_difficulty": m["aim"],
                                                           "speed_difficulty": m["speed"]}}
        cache[attr_key(m["id"], ["DT"], cs)] = {"t": t0, "a": {"star_rating": m["stars"] * 1.4, "aim_difficulty": m["aim"] * 1.38,
                                                               "speed_difficulty": m["speed"] * 1.42}}
    (data / "map_pool.json").write_text(json.dumps({"built_at": t0, "key": "demo", "window": [2, 5.5], "diffs": diffs}), encoding="utf-8")
    (data / "attr_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    (root / "config.toml").write_text('[player]\nuser = "demo"\n', encoding="utf-8")
    return root
