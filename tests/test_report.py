"""Wording bugs that actually happened: sentences must follow the numbers, whatever direction they go."""
import json

import pandas as pd
import pytest

from osu_coach import report
from osu_coach.history import compare


@pytest.fixture(scope="module")
def metrics(demo_cfg, demo_ds):
    M, _, _ = report.analyze(demo_cfg, demo_ds)
    return M


def line(M, start):
    return next(ln for ln in report.findings(M).splitlines() if ln.startswith(start))


@pytest.mark.parametrize("old,new,word", [(-1.26, 0.67, "1.9 pp above"), (0.95, -0.79, "1.7 pp below"), (0.1, 0.3, "about level with")])
def test_old_vs_new_direction(metrics, old, new, word):
    assert word in line(dict(metrics, resid_old=old, resid_new=new, acc_old=80.0, acc_new=81.0, stars_old=3.0, stars_new=3.0,
                             n_old=10, n_new=10), "**Old vs new")


def test_positive_effect_is_not_called_a_cost(metrics):
    t = line(dict(metrics, acc_sd_speed=2.0), "**Aim vs speed")
    assert "speed difficulty adds 2.0 pp" in t


def test_error_cost_follows_the_numbers(metrics):
    assert "misses cost you more" in line(dict(metrics, loss_ok=3.0, loss_miss=7.0), "**Error type")
    assert "100s/50s cost you more" in line(dict(metrics, loss_ok=7.0, loss_miss=3.0), "**Error type")


def test_trend_wording_needs_a_test(metrics):
    assert "no significant trend" in line(dict(metrics, slope_3_5=-4.0, p_3_5=0.2), "**Accuracy by difficulty")
    assert "falling 4.0" in line(dict(metrics, slope_3_5=-4.0, p_3_5=0.01), "**Accuracy by difficulty")


def test_no_breakpoint_means_no_bpm_in_the_bands(demo_cfg):
    text = (demo_cfg.out / "report.md").read_text(encoding="utf-8")
    bands = text[text.index("## Difficulty bands"):text.index("## Since last run")]
    if "BPM is left out" in bands:
        assert "eff BPM" not in bands


def test_report_is_generic(demo_cfg, metrics):
    """Nothing about one specific player may be baked into the output."""
    text = (demo_cfg.out / "report.md").read_text(encoding="utf-8")
    assert "demo player" in text
    for word in ["2018-22", "since 2023"]:
        assert word not in text


def test_metrics_are_valid_json(demo_cfg, metrics):
    assert json.loads((demo_cfg.out / "metrics.json").read_text(encoding="utf-8"))["username"] == "demo player"


def test_compare_first_run_and_quit_estimate():
    M = {"exported_at": "2026-01-02T00:00:00+0000", "play_count": 148, "pp": 11.0}
    assert "First run" in compare(None, M, pd.DataFrame())
    prev = {"_dir": "x", "exported_at": "2026-01-01T12:00:00+0000", "play_count": 100, "pp": 10.0}
    rall = pd.DataFrame({"date": pd.to_datetime(["2026-01-01T18:00:00Z"] * 48, utc=True)})
    assert "Every play was submitted" in compare(prev, M, rall)
    rall = pd.DataFrame({"date": pd.to_datetime(["2026-01-01T18:00:00Z"] * 20, utc=True)})
    assert "about 28 plays" in compare(prev, M, rall)
