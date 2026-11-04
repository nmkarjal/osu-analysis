import numpy as np
import pandas as pd
import pytest
from conftest import fake_attempts

from osu_coach import model


def finish_data(n=400, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    d = fake_attempts(n, seed)
    z = 2.5 - 1.5 * d["aim"] + (0 if signal else 1.5 * d["aim"].mean())
    p = 1 / (1 + np.exp(-z)) if signal else np.full(n, 0.5)
    d["passed"] = rng.random(n) < p
    return model.finish_table(d)


def test_logit_recovers_direction():
    d = finish_data()
    m = model.Logit(["aim", "speed"]).fit(d)
    assert m.b[1] < -0.3
    assert abs(m.b[2]) < abs(m.b[1])


def test_model_beats_base_rate_when_there_is_signal():
    t = model.compare_models(model.walk_forward(finish_data()))
    assert t["aim+speed"]["gain_lo"] > 0


def test_model_not_trusted_on_noise():
    """A coin-flip player must not get a 'trusted' model."""
    trusted = 0
    for s in range(6):
        t = model.compare_models(model.walk_forward(finish_data(seed=s, signal=False)), n_boot=300)
        trusted += t["aim+speed"]["gain_lo"] > 0
    assert trusted <= 1


def test_walk_forward_never_trains_on_test_maps_or_future():
    d = finish_data()
    wf = model.walk_forward(d)
    assert len(wf) < len(d)  # the first stretch is training only


def test_nf_is_not_a_finish():
    d = fake_attempts(50)
    d.loc[:9, "nf"] = True
    assert len(model.finish_table(d)) == 40


def test_ceilings_recover_planted_levels():
    d = fake_attempts(500, aim_miss=8, speed_ok=14)
    # miss line 3 sits at aim 2.0 and the 100s line 15 at speed 1.5, by construction
    c = model.clean_ceilings(d, n_boot=60)
    assert c["aim"] == pytest.approx(2.0, abs=0.08)
    assert c["speed"] == pytest.approx(1.5, abs=0.08)
    assert c["aim_ci"][0] < c["aim"] < c["aim_ci"][1]


def test_ceiling_missing_when_no_slope():
    c = model.clean_ceilings(fake_attempts(300, aim_miss=-5, speed_ok=14), n_boot=20)
    assert c["aim"] is None


def test_trend_waits_for_enough_sessions():
    s = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=4, tz="UTC"), "adj_miss_pct": [6, 5, 4, 3.0]})
    assert model.trend(s, "adj_miss_pct") is None
    s = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=8, freq="2D", tz="UTC"),
                      "adj_miss_pct": [6, 5.8, 5.5, 5.4, 5.1, 5, 4.7, 4.6]})
    t = model.trend(s, "adj_miss_pct")
    assert t["per_week"] < 0 and t["clear"]


def test_error_model_flags_uselessness():
    d = fake_attempts(200, noise=4)
    _, st = model.fit_error_model(d, "miss_pct")
    assert not st["useful"]
    d = fake_attempts(200, aim_miss=10)
    _, st = model.fit_error_model(d, "miss_pct")
    assert st["useful"]


def test_demo_player_is_recovered(demo_ds):
    from osu_coach.demo import AIM_SKILL, SPEED_SKILL
    c = model.clean_ceilings(demo_ds.att, n_boot=40)
    assert AIM_SKILL - 0.1 < c["aim"] < AIM_SKILL + 0.3
    assert SPEED_SKILL - 0.1 < c["speed"] < SPEED_SKILL + 0.3


def test_prediction_stays_in_range(demo_ds):
    res = model.build(demo_ds.rall, demo_ds.att)
    p = res["predictor"].predict(demo_ds.att.head(50))
    assert p["p_finish"].between(0, 1).all()
    assert p["acc_pred"].between(0, 100).all()
    assert np.isfinite(p["miss_pct_pred"]).all()
