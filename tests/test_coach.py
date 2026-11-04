import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from osu_coach import coach, mapfinder, planner, recommend
from osu_coach.config import MapSource, load_config
from osu_coach.diagnose import Finding


def cands(n=60, seed=0):
    rng = np.random.default_rng(seed)
    aim = rng.uniform(1.2, 3, n)
    speed = aim * rng.uniform(0.45, 0.85, n)
    return pd.DataFrame({"id": range(n), "set": range(n), "artist": [f"a{i % 9}" for i in range(n)], "title": "t", "version": "v",
                         "mods": ["DT" if i % 6 == 0 else "NM" for i in range(n)], "stars": aim + speed, "aim": aim, "speed": speed,
                         "acc_pred": 110 - 12 * aim, "p_finish": 0.5, "fav": rng.integers(1, 5000, n), "plays": 10,
                         "length_s": rng.uniform(60, 300, n), "eff_ar": 9.0, "eff_bpm": 180.0, "sources": [["x"]] * n})


def scan_result(findings=(), habits=()):
    return {"findings": list(findings), "habits": list(habits), "strengths": [], "clean": [], "skipped": [], "n_tests": 12}


def finding(key="aim_misses", area="aim", cost=2.0):
    return Finding(key, "weakness", "text", cost_pp=cost, area=area, drill=key, how="how")


def test_roles_land_near_their_target(demo_cfg):
    c = cands()
    lists = recommend.build_lists(c, demo_cfg, {"ratio": 0.6})
    warm = np.mean([m["acc_pred"] for m in lists["warmup"]])
    focus = np.mean([m["acc_pred"] for m in lists["focus_aim"]])
    assert warm > focus


def test_focus_lists_lean_the_right_way(demo_cfg):
    lists = recommend.build_lists(cands(200), demo_cfg, {"ratio": 0.65})
    r = lambda ms: np.mean([m["speed"] / m["aim"] for m in ms])  # noqa: E731
    assert r(lists["focus_speed"]) > r(lists["focus_aim"])


def test_no_mapset_twice_across_lists(demo_cfg):
    lists = recommend.build_lists(cands(200), demo_cfg, {"ratio": 0.65})
    sets = [m["set"] for v in lists.values() for m in v]
    assert len(sets) == len(set(sets))


def test_dt_list_is_only_dt(demo_cfg):
    lists = recommend.build_lists(cands(200), demo_cfg, {"ratio": 0.65})
    assert lists["dt"] and all(m["mods"] == "DT" for m in lists["dt"])


def test_challenge_makes_picks_harder(demo_cfg):
    c = cands(300)
    easy = recommend.build_lists(c, demo_cfg, {"ratio": 0.65}, challenge=0)["warmup"]
    hard = recommend.build_lists(c, demo_cfg, {"ratio": 0.65}, challenge=5)["warmup"]
    assert np.mean([m["acc_pred"] for m in hard]) < np.mean([m["acc_pred"] for m in easy])


def test_artist_variety(demo_cfg):
    c = cands(200)
    picks = recommend.pick(recommend.score(c, "warmup", 0.6), 8, set())
    assert max(pd.Series([m["artist"] for m in picks]).value_counts()) <= 3


def test_dnb_source_never_reaches_dt(demo_cfg):
    """no_dt and the BPM cap are filters at the candidate stage."""
    class Offline:
        offline = True
        class cache:
            save = staticmethod(lambda: None)
        def attributes(self, bid, mods, checksum=None):
            return {"aim_difficulty": 2.0, "speed_difficulty": 1.2, "star_rating": 3.0}
    pool = {str(i): {"id": i, "set": i, "artist": "a", "title": "t", "version": "v", "stars_nomod": 2.4, "ar": 8, "od": 8, "cs": 4,
                     "bpm": bpm, "length": 100, "checksum": "c", "fav": 10, "plays": 1, "sources": ["s"], "no_dt": no_dt}
            for i, (bpm, no_dt) in enumerate([(150, False), (150, True), (240, False)])}
    c = mapfinder.candidates(demo_cfg, Offline(), pool, (2.8, 3.5), set())
    assert c[c["mods"] == "DT"]["id"].tolist() == [0]


def test_excluded_maps_stay_out(demo_cfg):
    class Offline:
        class cache:
            save = staticmethod(lambda: None)
        def attributes(self, bid, mods, checksum=None):
            return {"aim_difficulty": 2.0, "speed_difficulty": 1.2, "star_rating": 3.0}
    pool = {"1": {"id": 1, "set": 1, "artist": "a", "title": "t", "version": "v", "stars_nomod": 3.2, "ar": 9, "od": 8, "cs": 4,
                  "bpm": 150, "length": 100, "checksum": "c", "fav": 10, "plays": 1, "sources": ["s"], "no_dt": False}}
    assert mapfinder.candidates(demo_cfg, Offline(), pool, (2.8, 3.5), {1}).empty


def test_plan_fills_the_time(demo_cfg):
    lists = recommend.build_lists(cands(200), demo_cfg, {"ratio": 0.65})
    plan = planner.build_plan(demo_cfg, lists, [], scan_result([finding(), finding("speed_timing", "speed", 1.0)]),
                              {"limiter": "aim"}, pd.DataFrame(), 0.0)
    assert sum(b["minutes"] for b in plan["blocks"]) == pytest.approx(demo_cfg.session_minutes)
    names = [b["name"] for b in plan["blocks"]]
    assert names[0] == "Warm-up" and names[-1] == "Cool-down" and "Focus: aim_misses" in names
    for b in plan["blocks"]:
        used = sum((m["length_s"] + planner.OVERHEAD_S) * m["plays"] for m in b["maps"]) / 60
        assert used < b["minutes"] * 1.6 + 5


def test_focus_falls_back_to_the_limiter():
    assert planner.choose_focus(scan_result(), {"limiter": "speed"})[0][0] == "speed"


def test_week_follows_cost():
    w = planner.week_schedule(scan_result([finding(cost=3.0), finding("speed_timing", "speed", 1.0)]), None, 4)
    assert w.count("aim_misses") == 3 and w.count("speed_timing") == 1


def review_case(acc, n=8):
    created = pd.Timestamp("2026-01-01", tz="UTC")
    plan = {"id": "p1", "created": created.isoformat(), "goals": [],
            "blocks": [{"maps": [{"id": 1, "acc_pred": 85.0}, {"id": 2, "acc_pred": 85.0}]}]}
    rall = pd.DataFrame({"date": created + pd.to_timedelta(np.arange(1, n + 1), unit="h"), "beatmap_id": [1, 2] * (n // 2),
                         "passed": True, "acc": acc})
    return plan, rall


def test_review_counts_adherence_and_bias():
    plan, rall = review_case(90.0)
    r = planner.review(plan, rall, None)
    assert r["adherence"] == 1 and r["bias"] == pytest.approx(5)


@pytest.mark.parametrize("acc,step", [(90.0, 1.0), (80.0, -1.0), (85.5, 0.0)])
def test_difficulty_loop(acc, step):
    plan, rall = review_case(acc)
    state = planner.adjust_challenge({"challenge": 0.0, "reviewed": []}, planner.review(plan, rall, None))
    assert state["challenge"] == step


def test_a_plan_only_counts_once():
    plan, rall = review_case(90.0)
    r = planner.review(plan, rall, None)
    s = planner.adjust_challenge({"challenge": 0.0, "reviewed": []}, r)
    s = planner.adjust_challenge(s, r)
    assert s["challenge"] == 1.0


def test_review_needs_new_plays():
    plan, rall = review_case(90.0)
    assert planner.review(plan, rall[rall["date"] < pd.Timestamp("2025-01-01", tz="UTC")], None) is None


def test_revisit_queue_due_dates():
    now = pd.Timestamp("2026-02-01", tz="UTC")
    base = {"nf": False, "aim": 2.0, "speed": 1.2, "title": "t", "artist": "a", "diff": "d", "mods": "NM", "stars": 3.0, "ar": 9,
            "eff_bpm": 170, "length_s": 90, "eff_od": 8, "acc": 80.0}
    rows = [dict(base, map_key="far", beatmap_id=1, beatmapset_id=1, passed=False, completion=0.8, date=now - pd.Timedelta(days=3)),
            dict(base, map_key="far", beatmap_id=1, beatmapset_id=1, passed=False, completion=0.7, date=now - pd.Timedelta(days=3)),
            dict(base, map_key="early", beatmap_id=2, beatmapset_id=2, passed=False, completion=0.1, date=now - pd.Timedelta(days=3)),
            dict(base, map_key="early", beatmap_id=2, beatmapset_id=2, passed=False, completion=0.1, date=now - pd.Timedelta(days=3)),
            dict(base, map_key="today", beatmap_id=3, beatmapset_id=3, passed=False, completion=0.9, date=now - pd.Timedelta(hours=2)),
            dict(base, map_key="today", beatmap_id=3, beatmapset_id=3, passed=False, completion=0.9, date=now - pd.Timedelta(hours=2))]
    q = recommend.revisit_queue(pd.DataFrame(rows), None, now)
    assert [m["id"] for m in q] == [1]


def test_config_defaults_and_sources(tmp_path):
    (tmp_path / "config.toml").write_text('[player]\nuser="x"\n[maps]\ndt_max_bpm=170\n[[maps.source]]\nname="a"\ngenre=4\nno_dt=true\n',
                                          encoding="utf-8")
    c = load_config(tmp_path)
    assert c.user == "x" and c.dt_max_bpm == 170 and c.sources == [MapSource(name="a", genre=4, no_dt=True)]
    assert load_config(tmp_path / "nothing_here").session_minutes == 45


def test_demo_end_to_end(tmp_path):
    """The README's first command: no API keys, no network, every output exists."""
    r = subprocess.run([sys.executable, "-m", "osu_coach", "demo", "--folder", str(tmp_path / "d")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "d" / "osu_analysis"
    for name in ["report.md", "plan.md", "practice_maps.md", "practice_maps.json", "metrics.json", "8_sessions.png",
                 "9_weaknesses.png", "10_recommendations.png", "4_comfort_zones.png"]:
        assert (out / name).exists(), name
    assert json.loads((out / "practice_maps.json").read_text())["sets"]
    assert len(list((out / "plans").glob("*.json"))) == 1
    assert (out / "history" / "runs.csv").exists()


def test_second_run_reviews_the_first(demo_root, demo_cfg, demo_ds):
    res = coach.analyze(demo_ds)
    assert coach.make_plan(demo_cfg, demo_ds, res, offline=True) is not None
    assert planner.latest_plan(demo_cfg)["blocks"]
