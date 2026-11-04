import numpy as np
import pandas as pd
import pytest

from osu_coach.config import load_config
from osu_coach.data import load_dataset
from osu_coach.demo import generate


@pytest.fixture(scope="session")
def demo_root(tmp_path_factory):
    return generate(tmp_path_factory.mktemp("demo"))


@pytest.fixture(scope="session")
def demo_cfg(demo_root):
    return load_config(demo_root)


@pytest.fixture(scope="session")
def demo_ds(demo_cfg):
    return load_dataset(demo_cfg)


def fake_attempts(n=300, seed=0, aim_miss=0.0, speed_ok=0.0, noise=1.0, maps=80):
    """A plain attempts table with planted effects, for testing the statistics."""
    rng = np.random.default_rng(seed)
    bid = rng.integers(0, maps, n)
    base_aim, base_speed = rng.uniform(1.5, 3, maps), rng.uniform(1, 2.2, maps)
    aim, speed = base_aim[bid], base_speed[bid]
    return pd.DataFrame({
        "beatmap_id": bid, "aim": aim, "speed": speed, "stars": aim + speed,
        "miss_pct": 3 + aim_miss * (aim - 2) + rng.normal(0, noise, n),
        "ok_pct": 15 + speed_ok * (speed - 1.5) + rng.normal(0, noise, n),
        "acc": 90 - aim_miss * (aim - 2) - speed_ok * 0.7 * (speed - 1.5) + rng.normal(0, noise, n),
        "date": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(np.arange(n), unit="min"),
        "passed": rng.random(n) < 0.5, "nf": False, "length_s": 120.0, "eff_od": 8.0, "hits": 300,
    })
