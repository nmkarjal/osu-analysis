"""Command line: python -m osu_coach <command>"""
import argparse
import getpass
import sys
from pathlib import Path

from . import __version__
from .config import load_config


def cmd_setup(cfg, a):
    from .api import API, ApiError, get_token, http, read_env_file
    print("You need your own osu! API client: osu.ppy.sh -> Settings -> OAuth -> New OAuth Application.")
    print("Name it anything and leave the callback URL blank. Your details stay in .env and config.toml in this folder.\n")
    env = cfg.root / ".env"
    have = read_env_file(env)
    cid = input(f"Client ID{' (enter keeps the saved one)' if have.get('OSU_CLIENT_ID') else ''}: ").strip() or have.get("OSU_CLIENT_ID", "")
    sec = getpass.getpass("Client secret (hidden): ").strip() or have.get("OSU_CLIENT_SECRET", "")
    user = input("osu! username or id: ").strip()
    try:
        tok = get_token(cid, sec)
        who = user if user.isdigit() else f"@{user}"
        found = http("GET", f"{API}/users/{who}/{cfg.mode}", tok)
    except ApiError as e:
        raise SystemExit(f"Couldn't find that player: {e}")
    env.write_text(f"OSU_CLIENT_ID={cid}\nOSU_CLIENT_SECRET={sec}\n", encoding="utf-8")
    conf = cfg.root / "config.toml"
    if conf.exists():
        print(f"Keeping your existing config.toml. Set user = \"{found['id']}\" there if it changed.")
    else:
        example = cfg.root / "config.example.toml"
        text = example.read_text(encoding="utf-8") if example.exists() else '[player]\nuser = ""\n'
        conf.write_text(text.replace('user = ""', f'user = "{found["id"]}"', 1), encoding="utf-8")
    print(f"All set for {found['username']}. Next: python -m osu_coach run")


def run_analysis(cfg, plan=True, offline=False, refresh=False, snapshot=True):
    from . import coach, history, report
    from .data import load_dataset
    ds = load_dataset(cfg)
    M, prev, res = report.analyze(cfg, ds)
    if plan:
        coach.make_plan(cfg, ds, res, offline=offline, refresh=refresh)
    if snapshot:
        history.snap_run(cfg, M)
    return M, prev, res


def summary(cfg, M, prev):
    from .history import KEYS
    print(f"\nReport: {cfg.out / 'report.md'}")
    if (cfg.out / "plan.md").exists():
        print(f"Plan:   {cfg.out / 'plan.md'}")
    print(f"Aim vs speed: {M['verdict']}")
    show = {"pp", "global_rank", "hit_accuracy", "ceil_aim", "ceil_speed", "adj_miss_last", "adj_ok_last", "fail_rate_now"}
    for k, lab, nd in [x for x in KEYS if x[0] in show]:
        v = M.get(k)
        if v is None or v != v:
            continue
        was = prev.get(k) if prev else None
        d = f"  ({v - was:+.{nd}f})" if was is not None and was == was else ""
        print(f"  {lab:<42} {v:.{nd}f}{d}")


def cmd_pull(cfg, a):
    from .pull import pull
    pull(cfg)


def cmd_run(cfg, a):
    from .pull import pull
    if not a.no_pull:
        pull(cfg)
    M, prev, _ = run_analysis(cfg, plan=not a.no_plan, refresh=a.refresh)
    summary(cfg, M, prev)


def cmd_analyze(cfg, a):
    M, prev, _ = run_analysis(cfg, plan=False, snapshot=False)
    summary(cfg, M, prev)


def cmd_plan(cfg, a):
    from . import coach
    from .data import load_dataset
    if a.minutes:
        cfg.session_minutes = a.minutes
    ds = load_dataset(cfg)
    res = coach.analyze(ds)
    coach.make_plan(cfg, ds, res, refresh=a.refresh)
    print(f"Plan: {cfg.out / 'plan.md'}\nMaps: {cfg.out / 'practice_maps.md'}")


def cmd_download(cfg, a):
    from .download import download
    download(cfg, test=a.test, list_only=a.list)


def cmd_demo(cfg, a):
    from .config import load_config as lc
    from .demo import generate
    root = Path(a.folder).resolve()
    generate(root, seed=a.seed)
    c = lc(root)
    M, prev, _ = run_analysis(c, plan=True, offline=True)
    summary(c, M, prev)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="osu-coach", description="osu! stats to weaknesses, map picks and practice plans.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--root", default=None, help="project folder (default: current folder)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="save your API client and username").set_defaults(fn=cmd_setup)
    p = sub.add_parser("run", help="pull, analyse, plan, save a snapshot. Run this after each session")
    p.add_argument("--no-pull", action="store_true", help="skip the API pull")
    p.add_argument("--no-plan", action="store_true", help="skip the plan and map search")
    p.add_argument("--refresh", action="store_true", help="rebuild the map pool")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("pull", help="only download data").set_defaults(fn=cmd_pull)
    sub.add_parser("analyze", help="only rebuild the report from the last pull").set_defaults(fn=cmd_analyze)
    p = sub.add_parser("plan", aliases=["maps"], help="new session plan and map lists from the last pull")
    p.add_argument("--minutes", type=int, help="session length")
    p.add_argument("--refresh", action="store_true", help="rebuild the map pool")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("download", help="queue the recommended maps in osu!direct (stable, supporter)")
    p.add_argument("--test", action="store_true", help="only the first map")
    p.add_argument("--list", action="store_true", help="show the list, queue nothing")
    p.set_defaults(fn=cmd_download)
    p = sub.add_parser("demo", help="try everything on a made-up player, no API keys needed")
    p.add_argument("--folder", default="demo_output")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    cfg = load_config(a.root)
    a.fn(cfg, a)


if __name__ == "__main__":
    sys.exit(main())
