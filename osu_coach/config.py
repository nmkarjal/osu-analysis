"""Settings. Everything personal lives in config.toml and .env, which git ignores."""
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_TAGS = {"nightcore/sped-up": r"nightcore|sped up|speed up"}


@dataclass
class MapSource:
    """Where candidate maps come from. Empty lists plus no genre means 'most favourited maps overall'."""
    name: str = "popular"
    genre: int | None = None
    queries: list = field(default_factory=list)
    artists: list = field(default_factory=list)
    no_dt: bool = False  # never suggest these with DT


@dataclass
class Config:
    root: Path = field(default_factory=Path.cwd)
    user: str = ""                  # username or numeric id
    mode: str = "osu"
    data_dir: str = "data"
    out_dir: str = "osu_analysis"
    era_split: str = ""             # plays before this date count as 'old'; empty = one year ago
    session_minutes: int = 45
    sessions_per_week: int = 4
    per_role: int = 8               # maps per list in practice_maps.md
    min_ar: float = 7.5
    dt_min_ar: float = 6.0          # nomod AR; 6 becomes about 8.3 with DT
    dt_max_bpm: float = 180.0       # nomod BPM cap for DT suggestions
    drill_max_length: int = 240     # focus maps longer than this (seconds) score lower
    sources: list = field(default_factory=lambda: [MapSource()])
    tags: dict = field(default_factory=lambda: dict(DEFAULT_TAGS))
    watch: list = field(default_factory=list)  # [{query, title, version}] maps to look up in the report

    @property
    def data(self):
        return self.root / self.data_dir

    @property
    def out(self):
        return self.root / self.out_dir

    @property
    def hist(self):
        return self.out / "history"

    @property
    def era_ts(self):
        if self.era_split:
            return pd.Timestamp(self.era_split, tz="UTC")
        return pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=365)


def load_config(root=None, path=None):
    root = Path(root or Path.cwd()).resolve()
    cfg = Config(root=root)
    p = Path(path) if path else root / "config.toml"
    if not p.exists():
        return cfg
    with open(p, "rb") as fh:
        raw = tomllib.load(fh)
    player, coach, maps = raw.get("player", {}), raw.get("coach", {}), raw.get("maps", {})
    cfg.user = str(player.get("user", cfg.user))
    cfg.era_split = str(player.get("era_split", cfg.era_split))
    cfg.session_minutes = int(coach.get("session_minutes", cfg.session_minutes))
    cfg.sessions_per_week = int(coach.get("sessions_per_week", cfg.sessions_per_week))
    for k in ["per_role", "min_ar", "dt_min_ar", "dt_max_bpm", "drill_max_length"]:
        if k in maps:
            setattr(cfg, k, type(getattr(cfg, k))(maps[k]))
    if maps.get("source"):
        cfg.sources = [MapSource(**s) for s in maps["source"]]
    if "tags" in raw:
        cfg.tags = dict(raw["tags"])
    cfg.watch = list(raw.get("watch", []))
    return cfg
