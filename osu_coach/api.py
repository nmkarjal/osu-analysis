"""Small osu! API v2 client: auth, rate limiting, paging and an on-disk attribute cache (stdlib only)."""
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://osu.ppy.sh/api/v2"
SLEEP = 1.0  # osu! asks for <= 60 requests/min

# Mods that change aim/speed attributes; everything else is ignored for the attributes lookup
DIFF_MODS = {"EZ", "HR", "DT", "NC", "HT", "DC", "FL"}
CACHE_DAYS = 30  # refetch after this long, in case star rating is reworked


class ApiError(Exception):
    pass


def http(method, url, token=None, data=None, form=False):
    headers = {"Accept": "application/json", "User-Agent": "osu-coach"}
    body = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        if form:
            body = urllib.parse.urlencode(data).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            body = json.dumps(data).encode()
            headers["Content-Type"] = "application/json"
    for attempt in range(5):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if e.code >= 500:
                print(f"  server error {e.code}, retrying")
                time.sleep(5)
                continue
            raise ApiError(f"HTTP {e.code} on {url.split('?')[0]}: {e.read().decode(errors='replace')[:300]}")
        except urllib.error.URLError as e:
            print(f"  network error ({e.reason}), retrying")
            time.sleep(5)
    raise ApiError(f"Gave up on {url.split('?')[0]}")


def read_env_file(path):
    """Minimal KEY=value reader for the git-ignored .env file."""
    out = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def credential(name, root=None):
    """Env var, then .env next to the project, then the Windows user env (for setx), then a prompt."""
    v = os.environ.get(name) or read_env_file(Path(root or Path.cwd()) / ".env").get(name)
    if v:
        return v.strip()
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v, _ = winreg.QueryValueEx(k, name)
                if v:
                    return str(v).strip()
        except OSError:
            pass
    if sys.stdin and sys.stdin.isatty():
        return (getpass.getpass(f"{name}: ") if "SECRET" in name else input(f"{name}: ")).strip()
    raise SystemExit(f"{name} is not set. Run: python -m osu_coach setup")


def get_token(cid, secret):
    try:
        r = http("POST", "https://osu.ppy.sh/oauth/token", form=True,
                 data={"client_id": cid, "client_secret": secret, "grant_type": "client_credentials", "scope": "public"})
    except ApiError as e:
        raise SystemExit(f"Authentication failed: {e}")
    return r["access_token"]


class AttrCache:
    """Difficulty attributes on disk, keyed by map + mods + map checksum."""

    def __init__(self, path):
        self.path = str(path)
        self.data = {}
        self.hits = self.fetched = 0
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except (OSError, ValueError):
                self.data = {}

    def get(self, key):
        e = self.data.get(key)
        if e and time.time() - e.get("t", 0) < CACHE_DAYS * 86400:
            self.hits += 1
            return e["a"]
        return None

    def put(self, key, a):
        self.fetched += 1
        self.data[key] = {"t": time.time(), "a": a}

    def save(self):
        # Merge with whatever another process saved meanwhile, so concurrent runs don't drop entries
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    disk = json.load(f)
                for k, v in disk.items():
                    if k not in self.data or v.get("t", 0) > self.data[k].get("t", 0):
                        self.data[k] = v
            except (OSError, ValueError):
                pass
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)


def attr_key(bid, mods, checksum=None):
    dm = sorted(m for m in mods if m in DIFF_MODS)
    return f"{bid}|{''.join(dm)}|{checksum or ''}"


class Client:
    """Token + cache. With offline=True it only ever reads the cache, which is how the demo runs."""

    def __init__(self, cache_path, root=None, offline=False, mode="osu"):
        self.cache = AttrCache(cache_path)
        self.offline = offline
        self.mode = mode
        self.token = None if offline else get_token(credential("OSU_CLIENT_ID", root),
                                                    credential("OSU_CLIENT_SECRET", root))

    def get(self, path, **params):
        if self.offline:
            raise ApiError("offline")
        q = urllib.parse.urlencode(params, doseq=True)
        return http("GET", f"{API}{path}" + (f"?{q}" if q else ""), self.token)

    def paged(self, path, params, cap):
        out, offset = [], 0
        while offset < cap:
            page = self.get(path, **dict(params, limit=100, offset=offset))
            if not page:
                break
            out += page
            if len(page) < 100:
                break
            offset += 100
            time.sleep(SLEEP)
        return out

    def search(self, params, pages):
        sets, cursor = [], None
        for _ in range(pages):
            q = dict(params, m=0, s="ranked", sort="favourites_desc")
            if cursor:
                q["cursor_string"] = cursor
            time.sleep(SLEEP)
            try:
                r = self.get("/beatmapsets/search", **q)
            except ApiError as e:
                print(f"  search {params} failed: {e}")
                break
            sets += r.get("beatmapsets") or []
            cursor = r.get("cursor_string")
            if not cursor:
                break
        return sets

    def attributes(self, bid, mods, checksum=None):
        key = attr_key(bid, mods, checksum)
        a = self.cache.get(key)
        if a is not None:
            return a
        if self.offline:
            return {}
        dm = sorted(m for m in mods if m in DIFF_MODS)
        time.sleep(SLEEP)
        try:
            r = http("POST", f"{API}/beatmaps/{bid}/attributes", self.token, {"mods": dm, "ruleset": self.mode})
            a = r.get("attributes") or {}
            self.cache.put(key, a)
        except ApiError as e:
            # Not cached, so a transient failure is retried next run
            print(f"  attributes failed for {bid} {dm}: {e}")
            a = {}
        return a
