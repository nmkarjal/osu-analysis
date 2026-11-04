"""Download a player's data: profile, top plays, last-24h attempts, most played, per-map difficulty."""
import csv
import json
import time

from .api import ApiError, Client

MOST_PLAYED_ATTRS = 60  # most played maps that get nomod aim/speed attributes


def mods_of(s):
    out = []
    for m in s.get("mods") or []:
        out.append(m if isinstance(m, str) else m.get("acronym"))
    return [m for m in out if m]


def stat(s, *keys):
    st = s.get("statistics") or {}
    for k in keys:
        if st.get(k) is not None:
            return st[k]
    return 0


def row(s, a):
    b = s.get("beatmap") or {}
    bs = s.get("beatmapset") or {}
    return {
        "score_id": s.get("id"),
        "date": s.get("created_at") or s.get("ended_at"),
        "artist": bs.get("artist"),
        "title": bs.get("title"),
        "diff": b.get("version"),
        "mapper": bs.get("creator"),
        "beatmap_id": b.get("id"),
        "beatmapset_id": b.get("beatmapset_id") or bs.get("id"),
        "mods": "".join(mods_of(s)) or "NM",
        "rank": s.get("rank"),
        "passed": s.get("passed", True),
        "acc": round((s.get("accuracy") or 0) * 100, 2),
        "pp": s.get("pp"),
        "combo": s.get("max_combo"),
        "map_max_combo": a.get("max_combo") or b.get("max_combo"),
        "n300": stat(s, "count_300", "great"),
        "n100": stat(s, "count_100", "ok"),
        "n50": stat(s, "count_50", "meh"),
        "miss": stat(s, "count_miss", "miss"),
        "stars_nomod": b.get("difficulty_rating"),
        "stars": a.get("star_rating"),
        "aim": a.get("aim_difficulty"),
        "speed": a.get("speed_difficulty"),
        "speed_notes": a.get("speed_note_count"),
        "slider_factor": a.get("slider_factor"),
        "bpm_nomod": b.get("bpm"),
        "length_s_nomod": b.get("total_length"),
        "drain_s_nomod": b.get("hit_length"),
        "ar": b.get("ar"),
        "od": b.get("accuracy"),
        "cs": b.get("cs"),
        "hp": b.get("drain"),
        "circles": b.get("count_circles"),
        "sliders": b.get("count_sliders"),
        "spinners": b.get("count_spinners"),
    }


def write_csv(path, rows):
    if not rows:
        print(f"  {path.name}: no rows")
        # Still truncate so a stale file from an earlier run is not reused
        open(path, "w", encoding="utf-8").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")


def lookup_watch(client, watch):
    """Look up the maps in [[watch]] so the report can show their real difficulty."""
    out = []
    for w in watch:
        for bs in client.search({"q": w.get("query", "")}, 1)[:10]:
            for b in bs.get("beatmaps") or []:
                if b.get("mode") != client.mode:
                    continue
                if w.get("title") and w["title"].lower() not in (bs.get("title") or "").lower():
                    continue
                if w.get("version") and w["version"].lower() != (b.get("version") or "").lower():
                    continue
                a = client.attributes(b["id"], [], b.get("checksum"))
                out.append({"query": w.get("query"), "beatmap_id": b["id"], "beatmapset_id": bs.get("id"),
                            "artist": bs.get("artist"), "title": bs.get("title"), "creator": bs.get("creator"),
                            "version": b.get("version"), "status": b.get("status"),
                            "stars": a.get("star_rating", b.get("difficulty_rating")),
                            "aim": a.get("aim_difficulty"), "speed": a.get("speed_difficulty"),
                            "bpm": b.get("bpm"), "length_s": b.get("total_length")})
    return out


def pull(cfg):
    if not cfg.user:
        raise SystemExit("No player set. Run: python -m osu_coach setup")
    cfg.data.mkdir(parents=True, exist_ok=True)
    client = Client(cfg.data / "attr_cache.json", root=cfg.root, mode=cfg.mode)
    who = cfg.user if cfg.user.isdigit() else f"@{cfg.user}"
    print("Profile...")
    try:
        user = client.get(f"/users/{who}/{cfg.mode}")
    except ApiError as e:
        raise SystemExit(f"Could not load player '{cfg.user}': {e}")
    uid = user["id"]
    print("Top plays...")
    best = client.paged(f"/users/{uid}/scores/best", {"mode": cfg.mode}, cap=200)
    print("Recent plays (24h, incl. fails)...")
    recent = client.paged(f"/users/{uid}/scores/recent", {"mode": cfg.mode, "include_fails": 1}, cap=1000)
    print("Most played...")
    most = client.paged(f"/users/{uid}/beatmapsets/most_played", {}, cap=1000)

    try:
        print("Difficulty attributes (cached ones are free, new ones ~1s each)...")
        top_rows, recent_rows = [], []
        for i, s in enumerate(best + recent):
            b = s.get("beatmap")
            if not b:
                continue
            a = client.attributes(b["id"], mods_of(s), b.get("checksum"))
            (top_rows if i < len(best) else recent_rows).append(row(s, a))
        mp_attrs = {}
        for m in sorted(most, key=lambda m: -(m.get("count") or 0))[:MOST_PLAYED_ATTRS]:
            b = m.get("beatmap") or {}
            bid = m.get("beatmap_id") or b.get("id")
            if bid:
                mp_attrs[str(bid)] = client.attributes(bid, [], b.get("checksum"))
        lookups = lookup_watch(client, cfg.watch)
    finally:
        client.cache.save()
    print(f"  attributes: {client.cache.hits} from cache, {client.cache.fetched} fetched")

    write_csv(cfg.data / "top.csv", top_rows)
    write_csv(cfg.data / "recent.csv", recent_rows)
    with open(cfg.data / "export.json", "w", encoding="utf-8") as f:
        json.dump({"exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "user": user, "best": best, "recent": recent,
                   "most_played": most, "most_played_attrs": mp_attrs, "lookups": lookups},
                  f, ensure_ascii=False, indent=1)
    st = user.get("statistics") or {}
    print(f"Done. {user.get('username')}: {st.get('pp')}pp, {st.get('hit_accuracy')}% acc, {st.get('play_count')} plays.")
