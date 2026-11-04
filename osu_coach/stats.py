"""Statistics helpers: regressions, breakpoint test, binning, zones, multiple-testing correction."""
import numpy as np
import pandas as pd
from scipy import stats

from .constants import COMFORT, MIN_N, PUSH


def simple(df, x, y):
    d = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 6 or d[x].nunique() < 3:
        return None
    lr = stats.linregress(d[x], d[y])
    return {"n": len(d), "slope": lr.slope, "intercept": lr.intercept, "r": lr.rvalue, "p": lr.pvalue}


def joint(df, y, xs):
    d = df[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < len(xs) + 8:
        return None
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy() for x in xs])
    Y = d[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = len(d) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    ss = (Y - Y.mean()) @ (Y - Y.mean())
    out = {"n": len(d), "r2": 1 - resid @ resid / ss if ss else np.nan, "coef": {}}
    for i, x in enumerate(xs, start=1):
        out["coef"][x] = {"b": beta[i], "se": se[i], "p": p[i], "per_sd": beta[i] * d[x].std(), "sd": d[x].std()}
    if len(xs) == 2:
        out["x_corr"] = d[xs[0]].corr(d[xs[1]])
    return out


def hinge(df, x, y, min_side=5, n_boot=1000):
    """
    Continuous two-segment fit with the breakpoint searched in 5-unit steps.

    Significance accounts for that search: residual bootstrap under the straight-line model, refitting the
    breakpoint search each time, p = share of simulations whose best breakpoint beats the line by as much as
    the real data. 'significant' also requires a drop-off (slope after the break lower than before).
    """
    d = df[[x, y]].dropna()
    if len(d) < 2 * min_side + 2:
        return None
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    lo, hi = np.percentile(xv, 10), np.percentile(xv, 90)
    ks = [k for k in np.arange(np.floor(lo / 5) * 5, hi + 5, 5)
          if (xv < k).sum() >= min_side and (xv >= k).sum() >= min_side]
    if not ks:
        return None
    X0 = np.column_stack([np.ones_like(xv), xv])
    H0 = X0 @ np.linalg.pinv(X0)
    Xs = [np.column_stack([np.ones_like(xv), xv, np.maximum(0, xv - k)]) for k in ks]
    Hs = np.stack([X @ np.linalg.pinv(X) for X in Xs])  # hat matrix per candidate breakpoint

    def gain(yy):
        sse_lin = ((yy - H0 @ yy) ** 2).sum()
        sse = ((yy[None, :] - Hs @ yy) ** 2).sum(axis=1)
        i = int(np.argmin(sse))
        return (1 - sse[i] / sse_lin if sse_lin else 0.0), i

    g_obs, i = gain(yv)
    b, *_ = np.linalg.lstsq(Xs[i], yv, rcond=None)
    fitted, resid = H0 @ yv, yv - H0 @ yv
    rng_ = np.random.default_rng(0)
    g_null = np.array([gain(fitted + rng_.choice(resid, len(resid), replace=True))[0] for _ in range(n_boot)])
    p = (np.sum(g_null >= g_obs) + 1) / (n_boot + 1)
    out = {"k": float(ks[i]), "before": b[1], "after": b[1] + b[2], "b": b, "sse_gain_pct": 100 * g_obs,
           "p": float(p), "n": len(d)}
    out["significant"] = p < 0.05 and out["after"] < out["before"]
    lin = stats.linregress(xv, yv)
    out["lin"] = {"slope": lin.slope, "intercept": lin.intercept, "r": lin.rvalue, "p": lin.pvalue}
    return out


def binned(df, x, step, y="acc", lo=None):
    d = df[[x, y]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["lo", "hi", "n", "mean", "median"])
    start = np.floor((d[x].min() if lo is None else lo) / step) * step
    edges = np.arange(start, d[x].max() + step, step)
    if len(edges) < 2:
        edges = np.array([start, start + step])
    cut = pd.cut(d[x], edges, right=False)
    g = d.groupby(cut, observed=False)[y].agg(["count", "mean", "median"]).reset_index()
    g["lo"] = [i.left for i in g[x]]
    g["hi"] = [i.right for i in g[x]]
    return g.rename(columns={"count": "n"})[["lo", "hi", "n", "mean", "median"]]


def zones(b):
    """Comfort = contiguous low bins with mean >= 93; push = bins with mean in 85-92 above that."""
    b = b[b["n"] >= MIN_N]
    comfort = b[b["mean"] >= COMFORT]
    push = b[(b["mean"] >= PUSH[0]) & (b["mean"] < COMFORT)]
    c = (float(comfort["lo"].min()), float(comfort["hi"].max())) if len(comfort) else None
    p = (float(push["lo"].min()), float(push["hi"].max())) if len(push) else None
    first_below = b[b["mean"] < COMFORT]
    edge = float(first_below["lo"].min()) if len(first_below) else None
    return {"comfort": c, "push": p, "first_below_93": edge}


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def rng(t, nd=2):
    return f"{t[0]:.{nd}f}-{t[1]:.{nd}f}" if t else "none"


def ols_cluster(df, y, xs, cluster):
    """
    OLS with cluster-robust standard errors. Retries of one map are not independent plays, so plain
    p-values would be too small; clustering on the map fixes that.
    """
    d = df[[y] + xs + [cluster]].replace([np.inf, -np.inf], np.nan).dropna()
    groups = d[cluster].to_numpy()
    uniq = np.unique(groups)
    k = len(xs) + 1
    if len(d) < k + 12 or len(uniq) < 8 or any(d[x].nunique() < 2 for x in xs):
        return None
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    Y = d[y].to_numpy(float)
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ Y
    e = Y - X @ beta
    meat = np.zeros((k, k))
    for g in uniq:
        m = groups == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    n, G = len(d), len(uniq)
    cov = (G / (G - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    dof = G - 1
    crit = stats.t.ppf(0.975, dof)
    out = {"n": n, "groups": G, "coef": {}}
    for i, x in enumerate(xs, start=1):
        t = beta[i] / se[i] if se[i] > 0 else 0.0
        sd = float(d[x].astype(float).std())
        out["coef"][x] = {"b": float(beta[i]), "se": float(se[i]), "p": float(2 * stats.t.sf(abs(t), dof)),
                          "lo": float(beta[i] - crit * se[i]), "hi": float(beta[i] + crit * se[i]), "sd": sd}
    return out


def bh_adjust(pvals):
    """Benjamini-Hochberg q-values: scanning many factors would otherwise 'find' noise."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def wilson(k, n, z=1.96):
    """Wilson interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))
