"""
Skill model.

Finish chance: three models compete on attempts they never saw (train on the past, skip the test map,
predict the future): the base rate, a star-rating logistic, and an aim+speed logistic. A model only counts
as trusted if it beats the base rate with a bootstrap interval that excludes zero.

Accuracy, misses and 100s: ridge regressions on aim, speed, OD and length, checked on held-out maps.

Ceilings: the aim level where you average MISS_LINE misses per 100 objects, and the speed level where you
average OK_LINE 100s per 100 objects. They come from the two error types the data can actually separate.
A two-ceiling "weakest link" finish model was tried first and dropped: aim and speed are too correlated in
real plays to identify it, and it predicted no better than plain logistic regression.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-6
MISS_LINE = 3.0   # misses per 100 objects
OK_LINE = 15.0    # 100s/50s per 100 objects; together with MISS_LINE that is roughly 86% accuracy


def finish_table(att):
    """Attempts usable for the finish model. NF passes say nothing about finishing, so they are dropped."""
    d = att[~att["nf"]].dropna(subset=["aim", "speed", "stars", "length_s", "date"]).copy()
    d["y"] = d["passed"].astype(float)
    d["log_len"] = np.log(d["length_s"].clip(lower=10))
    return d.sort_values("date")


def _prep_x(df, cols):
    out = df.copy()
    if "log_len" in cols and "log_len" not in out:
        out["log_len"] = np.log(out["length_s"].clip(lower=10))
    return out[cols].to_numpy(float)


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class BaseRate:
    name = "base rate"

    def fit(self, d):
        self.p = float((d["y"].sum() + 1) / (len(d) + 2))
        return self

    def predict(self, df):
        return np.full(len(df), self.p)


class Logit:
    """Ridge logistic regression by Newton steps, on standardised features."""

    def __init__(self, cols, l2=1.0, name=None):
        self.cols, self.l2 = list(cols), l2
        self.name = name or "+".join(cols)

    def fit(self, d):
        X = _prep_x(d, self.cols)
        y = d["y"].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        b = np.zeros(Z.shape[1])
        pen = np.eye(Z.shape[1]) * self.l2
        pen[0, 0] = 0
        for _ in range(50):
            p = 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
            g = Z.T @ (y - p) - pen @ b
            H = (Z * (p * (1 - p) + 1e-9)[:, None]).T @ Z + pen
            step = np.linalg.solve(H, g)
            b = b + step
            if np.abs(step).max() < 1e-6:
                break
        self.b = b
        return self

    def predict(self, df):
        Z = np.column_stack([np.ones(len(df)), (_prep_x(df, self.cols) - self.mu) / self.sd])
        return 1 / (1 + np.exp(-np.clip(Z @ self.b, -30, 30)))


def builders():
    return [BaseRate, lambda: Logit(["stars"], name="stars"),
            lambda: Logit(["aim", "speed", "log_len"], name="aim+speed")]


def walk_forward(d, min_train=60, blocks=6):
    """Train on the past, skip attempts on the test maps, predict each later block. Returns per-attempt losses."""
    n = len(d)
    if n < min_train + 20 or d["y"].nunique() < 2:
        return None
    start = max(min_train, int(n * 0.4))
    edges = np.linspace(start, n, blocks + 1).astype(int)
    preds = {}
    rows = []
    for i in range(blocks):
        test = d.iloc[edges[i]:edges[i + 1]]
        if test.empty:
            continue
        train = d.iloc[:edges[i]]
        train = train[~train["beatmap_id"].isin(test["beatmap_id"])]
        if len(train) < 30 or train["y"].nunique() < 2:
            continue
        rows.append(test.index.to_numpy())
        for b in builders():
            m = b().fit(train)
            preds.setdefault(m.name, []).append(m.predict(test))
    if not rows:
        return None
    idx = np.concatenate(rows)
    t = d.loc[idx]
    out = pd.DataFrame({"y": t["y"].to_numpy(), "map": t["beatmap_id"].to_numpy()})
    for k, v in preds.items():
        out[k] = np.concatenate(v)
    return out


def compare_models(wf, n_boot=1000, seed=0):
    """Mean log loss per model, plus a bootstrap over maps of each model's gain against the base rate."""
    y = wf["y"].to_numpy()
    names = [c for c in wf.columns if c not in ("y", "map")]
    loss = {k: log_loss(y, wf[k].to_numpy()) for k in names}
    base = loss["base rate"]
    uniq = wf["map"].unique()
    idx_by_map = [np.where(wf["map"].to_numpy() == m)[0] for m in uniq]
    rng = np.random.default_rng(seed)
    table = {}
    samples = [np.concatenate([idx_by_map[j] for j in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]
    for k in names:
        diff = loss[k] - base
        boots = np.array([diff[s].mean() for s in samples])
        table[k] = {"log_loss": float(loss[k].mean()), "brier": float(((wf[k] - y) ** 2).mean()),
                    "gain": float(-diff.mean()), "gain_lo": float(-np.percentile(boots, 97.5)),
                    "gain_hi": float(-np.percentile(boots, 2.5))}
    return table


def calibration(y, p, bins=5):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            rows.append({"predicted": float(p[m].mean()), "actual": float(y[m].mean()), "n": int(m.sum())})
    return rows


def fit_error_model(att, target, cols=("aim", "speed", "eff_od", "log_len"), l2=2.0):
    """Ridge regression for accuracy, misses or 100s per 100 objects. Returns (predict, out-of-sample stats)."""
    d = att.dropna(subset=[target, "aim", "speed", "eff_od", "length_s"]).copy()
    if len(d) < 30:
        return None, None
    d["log_len"] = np.log(d["length_s"].clip(lower=10))
    cols = [c for c in cols if d[c].nunique() > 1]

    def fit(t):
        X = t[cols].to_numpy(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        pen = np.eye(Z.shape[1]) * l2
        pen[0, 0] = 0
        b = np.linalg.solve(Z.T @ Z + pen, Z.T @ t[target].to_numpy(float))

        def predict(df):
            q = df.copy()
            if "log_len" not in q:
                q["log_len"] = np.log(q["length_s"].clip(lower=10))
            X2 = q[cols].fillna(dict(zip(cols, mu))).to_numpy(float)
            return np.column_stack([np.ones(len(X2)), (X2 - mu) / sd]) @ b
        return predict

    # 5-fold by map, so the check is about maps the model has not seen
    maps = d["beatmap_id"].unique()
    rng = np.random.default_rng(0)
    fold = dict(zip(rng.permutation(maps), np.arange(len(maps)) % 5))
    f = d["beatmap_id"].map(fold)
    err, base = [], []
    for i in range(5):
        tr, te = d[f != i], d[f == i]
        if len(tr) < 15 or te.empty:
            continue
        err.append(np.abs(fit(tr)(te) - te[target].to_numpy()))
        base.append(np.abs(tr[target].mean() - te[target].to_numpy()))
    mae, mae0 = float(np.concatenate(err).mean()), float(np.concatenate(base).mean())
    return fit(d), {"mae": mae, "mae_base": mae0, "n": len(d), "useful": mae < mae0 * 0.97}


MIN_ATTEMPTS = 60


@dataclass
class Predictor:
    """What the rest of the app uses. `trusted` is False when no finish model beat the base rate."""
    finish: object
    name: str
    trusted: bool
    errors: dict  # target -> predict function, only the ones that beat the mean on held-out maps

    def predict(self, df):
        out = df.copy()
        out["p_finish"] = self.finish.predict(out)
        for t, fn in self.errors.items():
            out[f"{t}_pred"] = fn(out)
        for t, lo, hi in [("acc", 0, 100), ("miss_pct", 0, 100), ("ok_pct", 0, 100)]:
            if f"{t}_pred" in out:
                out[f"{t}_pred"] = out[f"{t}_pred"].clip(lo, hi)
        return out


def _line_point(fit, xs, line, ref):
    """Where a fitted plane reaches `line`, moving along the first variable with the second fixed at ref."""
    b0, b1, b2 = fit
    return (line - b0 - b2 * ref) / b1 if b1 > 0 else None


def _plane(d, y):
    X = np.column_stack([np.ones(len(d)), d["aim"], d["speed"]])
    return np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)[0]


def clean_ceilings(att, n_boot=300, seed=0, last=250):
    """
    aim: the aim level where misses reach MISS_LINE, with speed at your median.
    speed: the speed level where 100s reach OK_LINE, with aim at your median.
    limiter: moving up along your typical map balance, which line do you hit first?
    Intervals come from resampling whole maps.
    """
    d = att.dropna(subset=["aim", "speed", "miss_pct", "ok_pct"]).sort_values("date").tail(last)
    if len(d) < 40 or d["beatmap_id"].nunique() < 15:
        return None
    a_ref, s_ref = float(d["aim"].median()), float(d["speed"].median())
    ratio = float((d["speed"] / d["aim"]).median())

    def point(t):
        pm, po = _plane(t, "miss_pct"), _plane(t, "ok_pct")
        aim = (MISS_LINE - pm[0] - pm[2] * s_ref) / pm[1] if pm[1] > 0 else np.nan
        speed = (OK_LINE - po[0] - po[1] * a_ref) / po[2] if po[2] > 0 else np.nan
        # Along speed = ratio * aim, the aim value where each line is reached
        rm, ro = pm[1] + pm[2] * ratio, po[1] + po[2] * ratio
        hit_m = (MISS_LINE - pm[0]) / rm if rm > 0 else np.inf
        hit_o = (OK_LINE - po[0]) / ro if ro > 0 else np.inf
        return aim, speed, hit_m, hit_o

    aim, speed, hit_m, hit_o = point(d)
    groups = [g.index.to_numpy() for _, g in d.groupby("beatmap_id")]
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        pts.append(point(d.loc[np.concatenate([groups[i] for i in pick])]))
    pts = np.array(pts, float)

    def ci(col):
        v = pts[:, col]
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (None, None)

    first = "aim" if hit_m < hit_o else "speed"
    share = float(np.mean(pts[:, 2] < pts[:, 3]))
    return {"aim": None if not np.isfinite(aim) else float(aim), "aim_ci": ci(0),
            "speed": None if not np.isfinite(speed) else float(speed), "speed_ci": ci(1),
            "limiter": first, "limiter_conf": share if first == "aim" else 1 - share,
            "ratio": ratio, "aim_ref": a_ref, "speed_ref": s_ref, "n": len(d)}


def session_effects(att, min_n=8):
    """Each session's misses, 100s and accuracy at equal difficulty, so harder maps don't look like getting worse."""
    d = att.dropna(subset=["aim", "speed", "miss_pct", "ok_pct", "acc"])
    if len(d) < 30 or "session" not in d:
        return pd.DataFrame()
    X = np.column_stack([np.ones(len(d)), d["aim"], d["speed"]])
    out = {"date": d.groupby("session")["date"].min(), "n": d.groupby("session").size()}
    for y in ["miss_pct", "ok_pct", "acc"]:
        b = np.linalg.lstsq(X, d[y].to_numpy(float), rcond=None)[0]
        resid = pd.Series(d[y].to_numpy(float) - X @ b, index=d.index)
        g = resid.groupby(d["session"])
        out[f"adj_{y}"] = g.mean() + d[y].mean()
        out[f"se_{y}"] = g.std() / np.sqrt(g.size())
    t = pd.DataFrame(out)
    t["finish"] = d.groupby("session")["passed"].mean()
    return t[t["n"] >= min_n].reset_index()


def trend(sessions, col):
    """Change per week of a session series. None until there are 5+ sessions over 3+ days."""
    if len(sessions) < 5:
        return None
    days = (sessions["date"] - sessions["date"].min()).dt.total_seconds() / 86400
    if days.max() < 3:
        return None
    d = pd.DataFrame({"t": days / 7, "y": sessions[col], "g": range(len(sessions))})
    x = d["t"] - d["t"].mean()
    b = float((x * (d["y"] - d["y"].mean())).sum() / (x ** 2).sum())
    resid = d["y"] - d["y"].mean() - b * x
    se = float(np.sqrt((resid ** 2).sum() / (len(d) - 2) / (x ** 2).sum()))
    return {"per_week": b, "se": se, "clear": abs(b) > 2 * se}


def build(rall_new, att):
    """Fit everything. `rall_new` = every attempt in the era, `att` = those with enough objects to measure."""
    res = {"n": 0, "table": None, "chosen": None, "trusted": False, "predictor": None, "ceilings": None,
           "sessions": pd.DataFrame(), "calibration": [], "errors": {}, "tested": 0, "finish_rate": None}
    if rall_new is None or rall_new.empty:
        return res
    d = finish_table(rall_new)
    res["n"] = len(d)
    res["finish_rate"] = float(d["y"].mean()) if len(d) else None
    res["sessions"] = session_effects(att)
    res["ceilings"] = clean_ceilings(att)
    errors = {}
    for t in ["acc", "miss_pct", "ok_pct"]:
        fn, st = fit_error_model(att, t)
        if fn:
            res["errors"][t] = st
            if st["useful"]:
                errors[t] = fn
    if len(d) < MIN_ATTEMPTS or d["y"].nunique() < 2:
        return res
    wf = walk_forward(d)
    if wf is not None:
        res["tested"] = len(wf)
        table = compare_models(wf)
        res["table"] = table
        best = "aim+speed" if table["aim+speed"]["gain"] >= table["stars"]["gain"] else "stars"
        res["chosen"] = best
        res["trusted"] = table[best]["gain_lo"] > 0
        res["calibration"] = calibration(wf["y"].to_numpy(), wf[best].to_numpy())
    name = res["chosen"] or "aim+speed"
    finish = next(m for m in (b() for b in builders()) if m.name == name).fit(d)
    res["predictor"] = Predictor(finish, name, res["trusted"], errors)
    return res
