"""Queue the recommended mapsets in osu! stable through osu!direct (needs supporter)."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def stable_dir():
    return Path(os.environ.get("LOCALAPPDATA", "")) / "osu!"


def installed(songs):
    if not songs.exists():
        return set()
    return {int(m.group(1)) for d in os.listdir(songs) if (m := re.match(r"(\d+)\s", d))}


def download(cfg, test=False, list_only=False, delay=4.0):
    path = cfg.out / "practice_maps.json"
    if not path.exists():
        raise SystemExit("No map list yet. Run: python -m osu_coach maps")
    sets = json.loads(path.read_text(encoding="utf-8"))["sets"]
    have = installed(stable_dir() / "Songs")
    todo = [s for s in sets if s not in have]
    print(f"{len(sets)} mapsets in the list, {len(sets) - len(todo)} already installed, {len(todo)} to queue.")
    if test:
        todo = todo[:1]
    exe = stable_dir() / "osu!.exe"
    if not list_only and (sys.platform != "win32" or not exe.exists()):
        print("osu! stable not found, so here are the links to open yourself:")
        list_only = True
    for i, sid in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] https://osu.ppy.sh/beatmapsets/{sid}")
        if list_only:
            continue
        # Straight to osu!.exe: with stable and lazer both installed, the shell shows a "choose an app" dialog
        subprocess.Popen([str(exe), f"osu://dl/{sid}"])
        if i < len(todo):
            time.sleep(delay)
    if todo and not list_only:
        print("Queued. Downloads import by themselves.")
