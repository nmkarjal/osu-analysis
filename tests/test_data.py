import numpy as np
import pandas as pd
import pytest

from osu_coach.data import add_sessions, add_uid, classify, effective_ar, effective_od, prep, split_mods

ERA = pd.Timestamp("2023-01-01", tz="UTC")


def row(**kw):
    r = {"score_id": 1, "date": "2026-10-18T12:00:00Z", "beatmap_id": 5, "mods": "NM", "passed": True, "acc": 90,
         "n300": 90, "n100": 8, "n50": 1, "miss": 1, "aim": 2.0, "speed": 1.5, "stars": 3.5, "bpm_nomod": 180,
         "length_s_nomod": 120, "ar": 9, "od": 8, "cs": 4, "circles": 60, "sliders": 39, "spinners": 1, "speed_notes": 30}
    r.update(kw)
    return r


def test_split_mods():
    assert split_mods("HDDT") == {"HD", "DT"}
    assert split_mods("NM") == {"NM"}


def test_effective_ar_known_values():
    assert float(effective_ar(9, 1.5)) == pytest.approx(10.33, abs=0.01)
    assert float(effective_ar(6, 1.5)) == pytest.approx(8.33, abs=0.01)
    assert float(effective_ar(9, 1.0, hr=True)) == pytest.approx(10.0)
    assert float(effective_ar(8, 1.0, ez=True)) == pytest.approx(4.0)


def test_effective_od_dt():
    # 300 window 32 ms at OD 8, 21.33 ms with DT
    assert float(effective_od(8, 1.5)) == pytest.approx((80 - 32 / 1.5) / 6)


def test_prep_mods_change_bpm_and_length():
    d = prep(pd.DataFrame([row(mods="HDDT"), row(mods="HT", score_id=2), row(score_id=3)]), "top", ERA)
    assert list(d["eff_bpm"]) == [270, 135, 180]
    assert list(d["length_s"]) == [80, 160, 120]
    assert d["hd"].tolist() == [True, False, False]


def test_fail_gets_unique_key():
    """Failed scores all come back with id 0. Collapsing them was a real bug."""
    d = pd.DataFrame([row(score_id=0, date="2026-10-18T12:00:00Z"), row(score_id=0, date="2026-10-18T12:01:00Z"),
                      row(score_id=7)])
    assert add_uid(d)["uid"].nunique() == 3


def test_completion_uses_all_objects():
    d = prep(pd.DataFrame([row(n300=40, n100=8, n50=1, miss=1, passed=False)]), "recent", ERA)
    assert d["completion"].iloc[0] == pytest.approx(0.5)


@pytest.mark.parametrize("miss,oks,expected", [(0, 1, "clean"), (5, 8, "miss-heavy"), (1, 20, "timing-heavy"), (3, 20, "mixed")])
def test_classify(miss, oks, expected):
    r = {"n100": oks, "n50": 0, "miss": miss, "hits": 300, "ok_pct": 100 * oks / 300}
    assert classify(r) == expected


def test_sessions_split_on_gap():
    t = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:05", "2026-01-01 12:00"], utc=True)
    d = pd.DataFrame({"date": t, "beatmap_id": [1, 1, 2], "mods": "NM"})
    s = add_sessions(d)
    assert s["session"].tolist() == [1, 1, 2]
    assert s["streak_try"].tolist() == [1, 2, 1]
    assert s["session_min"].iloc[1] == 5


def test_era_split_is_configurable():
    d = prep(pd.DataFrame([row(date="2022-01-01T00:00:00Z"), row(score_id=2)]), "top", ERA)
    assert d["era"].tolist() == ["old", "new"]
    assert np.all(d["ratio"] == 0.75)
