import numpy as np
import pandas as pd
import pytest

from osu_coach.stats import bh_adjust, binned, hinge, joint, ols_cluster, simple, wilson, zones


def line_data(n=120, seed=0, k=None, drop=0.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(100, 260, n)
    y = 85 + rng.normal(0, 3, n)
    if k:
        y = y - drop * np.maximum(0, x - k)
    return pd.DataFrame({"x": x, "y": y})


def test_hinge_finds_a_planted_break():
    h = hinge(line_data(k=190, drop=0.3), "x", "y", n_boot=300)
    assert h["significant"]
    assert 170 <= h["k"] <= 210


def test_hinge_rejects_a_straight_line():
    """The breakpoint search always finds something. It must not count unless it beats a line."""
    falses = sum(hinge(line_data(seed=s), "x", "y", n_boot=200)["significant"] for s in range(12))
    assert falses <= 1


def test_hinge_needs_a_drop_not_just_a_kink():
    d = line_data(seed=2)
    d["y"] = d["y"] + 0.3 * np.maximum(0, d["x"] - 190)  # accuracy rising after the kink is no drop-off
    assert not hinge(d, "x", "y", n_boot=200)["significant"]


def test_joint_separates_correlated_predictors():
    rng = np.random.default_rng(1)
    a = rng.normal(size=400)
    s = 0.7 * a + rng.normal(0, 0.7, 400)
    d = pd.DataFrame({"a": a, "s": s, "y": -2 * s + rng.normal(0, 0.5, 400)})
    j = joint(d, "y", ["a", "s"])
    assert j["coef"]["s"]["b"] == pytest.approx(-2, abs=0.15)
    assert abs(j["coef"]["a"]["b"]) < 0.15


def test_cluster_se_is_wider_with_repeats():
    """Copying each map ten times adds no information, so the clustered p-value must not shrink."""
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"g": range(40), "x": rng.normal(size=40)})
    base["y"] = 0.25 * base["x"] + rng.normal(size=40)
    many = pd.concat([base] * 10, ignore_index=True)
    naive = simple(many, "x", "y")["p"]
    clustered = ols_cluster(many, "y", ["x"], "g")["coef"]["x"]["p"]
    assert clustered > naive * 5


def test_bh_adjust():
    q = bh_adjust([0.01, 0.04, 0.03, 0.5])
    assert q[0] == pytest.approx(0.04)
    assert q[3] == pytest.approx(0.5)
    assert np.all(q >= [0.01, 0.04, 0.03, 0.5])


def test_wilson_bounds():
    lo, hi = wilson(0, 10)
    assert lo == 0 and 0.2 < hi < 0.4
    assert wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]


def test_zones():
    d = pd.DataFrame({"x": [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3], "acc": [97, 96, 95, 88, 89, 87, 70, 72, 71]})
    z = zones(binned(d, "x", 1.0))
    assert z["comfort"] == (1.0, 2.0)
    assert z["push"] == (2.0, 3.0)
    assert z["first_below_93"] == 2.0
