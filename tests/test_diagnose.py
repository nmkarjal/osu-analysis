import numpy as np
import pandas as pd
from conftest import fake_attempts

from osu_coach import diagnose
from osu_coach.data import Dataset


def dataset(att):
    a = att.copy()
    rng = np.random.default_rng(1)
    for c, v in [("log2_len", 7.0), ("eff_ar", rng.normal(9, 0.5, len(a))), ("eff_cs", rng.normal(4, 0.3, len(a))),
                 ("slider_share", rng.uniform(0.2, 0.6, len(a))), ("stream_share", rng.uniform(0.1, 0.5, len(a))),
                 ("dt", False), ("hr", False), ("hd", False), ("completion", 1.0), ("title", "t"), ("diff", "d")]:
        a[c] = v
    a["eff_od"] = rng.normal(8, 0.5, len(a))
    a["streak"] = range(len(a))
    a["streak_try"] = 1
    empty = pd.DataFrame()
    return Dataset({}, empty, empty, a, a, a.assign(passed=True), a, "x", "since", "before")


def keys(res):
    return [f.key for f in res["findings"]]


def test_planted_weaknesses_are_found():
    r = diagnose.scan(dataset(fake_attempts(400, aim_miss=6, speed_ok=10)))
    assert "aim_misses" in keys(r) and "speed_timing" in keys(r)


def test_noise_stays_near_the_false_discovery_rate():
    """A dozen factors on pure noise. FDR 10% means about 1 scan in 10 may flag something; measured rate is 12%."""
    scans = [diagnose.scan(dataset(fake_attempts(300, seed=s))) for s in range(120)]
    flagged = sum(bool(r["findings"] or r["strengths"]) for r in scans)
    assert flagged <= 20


def test_every_finding_has_a_drill():
    r = diagnose.scan(dataset(fake_attempts(400, aim_miss=6, speed_ok=10)))
    assert all(f.drill and f.how for f in r["findings"])


def test_session_effects_need_several_sessions():
    a = fake_attempts(300)
    a["session"] = 1
    a["session_hours"] = np.linspace(0, 2, 300)
    a["warmup"] = np.arange(300) < 5
    r = diagnose.scan(dataset(a))
    assert "fatigue" in r["skipped"]


def test_two_sided_check_names_the_hard_side():
    a = fake_attempts(400)
    ds = dataset(a)
    ds.att["eff_ar"] = ds.att["beatmap_id"] % 5 + 7.0
    ds.att["miss_pct"] = ds.att["miss_pct"] - 2.5 * (ds.att["eff_ar"] - 9)
    r = diagnose.scan(ds)
    f = next(f for f in r["findings"] if f.key == "ar_reading")
    assert f.extra["side"] == "low" and "low AR" in f.how


def test_overreach_habit():
    a = fake_attempts(200)
    a["passed"] = np.arange(200) % 10 == 0
    ds = dataset(a)
    ds.rall["passed"] = a["passed"]
    habits = diagnose.behaviour(ds.rall)
    assert any(h.key == "overreach" for h in habits)


def test_markdown_lists_clean_checks():
    md = diagnose.to_markdown(diagnose.scan(dataset(fake_attempts(300))))
    assert "no clear effect" in md


def test_factor_that_tracks_difficulty_is_not_an_effect(demo_ds):
    """The demo player has no AR or OD effect, but both follow star rating and errors curve with difficulty."""
    r = diagnose.scan(demo_ds)
    found = [f.key for f in r["findings"] + r["strengths"]]
    assert "ar_reading" not in found and "od_timing" not in found
    assert "ar_reading" in r["collinear"]
