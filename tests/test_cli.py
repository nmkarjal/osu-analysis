import shutil

import pytest

from osu_coach import cli, report
from osu_coach.config import load_config
from osu_coach.data import load_dataset


def test_setup_saves_locally_and_never_prints_the_secret(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.example.toml").write_text('[player]\nuser = ""\n', encoding="utf-8")
    answers = iter(["123", "someplayer"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr("getpass.getpass", lambda *_: "s3cret-value")
    monkeypatch.setattr("osu_coach.api.get_token", lambda cid, sec: "token")
    monkeypatch.setattr("osu_coach.api.http", lambda *a, **k: {"id": 42, "username": "someplayer"})
    cli.main(["--root", str(tmp_path), "setup"])
    assert "OSU_CLIENT_SECRET=s3cret-value" in (tmp_path / ".env").read_text(encoding="utf-8")
    assert load_config(tmp_path).user == "42"
    assert "s3cret-value" not in capsys.readouterr().out


def test_credential_reads_dotenv(tmp_path, monkeypatch):
    from osu_coach.api import credential
    monkeypatch.delenv("OSU_CLIENT_ID", raising=False)
    (tmp_path / ".env").write_text("OSU_CLIENT_ID=777\n", encoding="utf-8")
    assert credential("OSU_CLIENT_ID", tmp_path) == "777"


def test_pull_without_a_player_explains_itself(tmp_path):
    with pytest.raises(SystemExit, match="setup"):
        cli.main(["--root", str(tmp_path), "pull"])


def test_analyze_before_any_pull_explains_itself(tmp_path):
    with pytest.raises(SystemExit, match="pull"):
        cli.main(["--root", str(tmp_path), "analyze"])


def test_coming_back_after_a_break(demo_root, tmp_path, capsys):
    """Nothing played in the last 24 hours: the pull is empty, the report still has to work."""
    root = tmp_path / "copy"
    shutil.copytree(demo_root, root)
    (root / "data" / "recent.csv").write_text("", encoding="utf-8")
    cfg = load_config(root)
    M, prev, _ = report.analyze(cfg, load_dataset(cfg))
    assert M["fail_rate_now"] is None
    cli.summary(cfg, M, prev)
    assert "nan" not in capsys.readouterr().out
