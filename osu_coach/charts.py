"""All matplotlib charts. Every function takes data plus an output folder and writes one PNG."""
import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .constants import COMFORT, ERA_NEW, ERA_OLD, MIN_N, PUSH  # noqa: E402
from .stats import binned, simple  # noqa: E402

# Chart style: light surface, recessive axes, fixed categorical order
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
ZONE_OK, ZONE_PUSH = "#e3f1ea", "#fbeee2"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "axes.axisbelow": True,
})


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)


def scatter_fit(ax, df, x, y, label_x):
    groups = [("top", C1, "top plays"), ("recent", C2, "recent passes")]
    for src, col, lab in groups:
        d = df[df["source"] == src]
        if len(d):
            ax.scatter(d[x], d[y], s=22, color=col, alpha=0.65, edgecolor=SURF, linewidth=0.6, label=lab)
    fit = simple(df, x, y)
    if fit:
        xs = np.linspace(df[x].min(), df[x].max(), 50)
        ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=INK, linewidth=1.5)
        ax.set_title(f"{label_x}: slope {fit['slope']:+.2f}, r = {fit['r']:+.2f} (n={fit['n']})", fontsize=10)
    ax.set_xlabel(label_x)


def chart_hypothesis(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for j, (x, lab) in enumerate([("aim", "aim difficulty"), ("speed", "speed difficulty"), ("ratio", "speed / aim")]):
        scatter_fit(axes[0, j], df, x, "acc", lab)
        axes[0, j].set_ylabel("accuracy %")
        scatter_fit(axes[1, j], df, x, "miss_pct", lab)
        axes[1, j].set_ylabel("misses per 100 objects")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("Accuracy and misses vs aim / speed difficulty (passed plays)", fontweight="bold")
    save(fig, out, "1_acc_miss_vs_aim_speed.png")


def chart_errors(df, out):
    order = ["clean", "timing-heavy", "mixed", "miss-heavy"]
    cols = {"clean": C3, "timing-heavy": C1, "mixed": "#a9a8a2", "miss-heavy": C2}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    srcs = [s for s in ["top", "recent"] if (df["source"] == s).any()]
    left = np.zeros(len(srcs))
    for t in order:
        vals = np.array([(df[df["source"] == s]["err_type"] == t).mean() * 100 for s in srcs])
        a1.barh(srcs, vals, left=left, color=cols[t], edgecolor=SURF, linewidth=2, label=t)
        for i, v in enumerate(vals):
            if v >= 8:
                a1.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += vals
    a1.set_xlabel("% of passed plays")
    a1.set_title("Error profile by source")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    a1.grid(axis="y", visible=False)
    for t in order:
        d = df[df["err_type"] == t]
        a2.scatter(d["aim"], d["speed"], s=26, color=cols[t], alpha=0.75, edgecolor=SURF, linewidth=0.6, label=t)
    a2.set_xlabel("aim difficulty")
    a2.set_ylabel("speed difficulty")
    a2.set_title("Where each error type happens")
    a2.legend(loc="upper left")
    save(fig, out, "2_error_types.png")


def chart_bpm(df, fails, h, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.scatter(df["eff_bpm"], df["acc"], s=22, color=C1, alpha=0.6, edgecolor=SURF, linewidth=0.6, label="passed play")
    b = binned(df, "eff_bpm", 20)
    b = b[b["n"] >= MIN_N]
    a1.plot((b["lo"] + b["hi"]) / 2, b["mean"], color=INK, marker="o", markersize=5, label="20-BPM bin mean")
    if h:
        xs = np.linspace(df["eff_bpm"].min(), df["eff_bpm"].max(), 100)
        if h["significant"]:
            ys = h["b"][0] + h["b"][1] * xs + h["b"][2] * np.maximum(0, xs - h["k"])
            a1.plot(xs, ys, color=C2, linestyle="--", label=f"breakpoint at {h['k']:.0f} BPM (p={h['p']:.3f})")
            a1.axvline(h["k"], color=C2, linewidth=1, alpha=0.6)
        else:
            ln = h["lin"]
            a1.plot(xs, ln["intercept"] + ln["slope"] * xs, color=C2, linestyle="--",
                    label=f"straight line (no breakpoint, p={h['p']:.2f})")
    a1.axhline(COMFORT, color=INK2, linewidth=1, linestyle=":")
    a1.set_xlabel("effective BPM (DT x1.5, HT x0.75)")
    a1.set_ylabel("accuracy %")
    a1.set_title("Accuracy vs effective BPM")
    a1.legend(loc="lower left")
    if not fails.empty:
        fb = fails.copy()
        fb["bin"] = (fb["eff_bpm"] // 20) * 20
        g = fb.groupby("bin").agg(n=("passed", "size"), fail=("passed", lambda s: 100 * (1 - s.mean()))).reset_index()
        g = g[g["n"] >= MIN_N]
        a2.bar(g["bin"] + 10, g["fail"], width=16, color=C2, edgecolor=SURF)
        for _, r in g.iterrows():
            a2.text(r["bin"] + 10, r["fail"] + 1, f"n={int(r['n'])}", ha="center", fontsize=8, color=INK2)
    a2.set_xlabel("effective BPM")
    a2.set_ylabel("fail rate %")
    a2.set_title("Fail rate by effective BPM (all recent attempts)")
    save(fig, out, "3_bpm_dropoff.png")


def chart_zones(df, out):
    # Dots, not bars: the axis does not start at 0, which bars would misrepresent
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (x, step, lab) in zip(axes, [("stars", 0.5, "star rating (with mods)"),
                                         ("aim", 0.25, "aim difficulty"),
                                         ("speed", 0.25, "speed difficulty")]):
        b = binned(df, x, step)
        b = b[b["n"] >= 1].copy()
        b["mid"] = b["lo"] + step / 2
        good = b[b["n"] >= MIN_N]
        thin = b[b["n"] < MIN_N]
        lo = float(np.nanmin(b["mean"])) if len(b) else 80.0
        ymin = np.floor(min(lo, PUSH[0]) / 5) * 5 - 5
        ax.set_ylim(ymin, 102)
        ax.axhspan(COMFORT, 102, color=ZONE_OK, zorder=0)
        ax.axhspan(PUSH[0], COMFORT, color=ZONE_PUSH, zorder=0)
        ax.plot(good["mid"], good["mean"], color=C1, linewidth=1.5, zorder=2)
        ax.scatter(good["mid"], good["mean"], s=70, color=C1, edgecolor=SURF, linewidth=1.5, zorder=3,
                   label=f"{MIN_N}+ plays")
        ax.scatter(thin["mid"], thin["mean"], s=70, facecolor=SURF, edgecolor="#a9a8a2", linewidth=1.5, zorder=3,
                   label=f"fewer than {MIN_N}")
        for _, r in b.iterrows():
            ax.annotate(f"{int(r['n'])}", (r["mid"], r["mean"]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2, annotation_clip=True)
        ax.set_xticks(np.arange(b["lo"].min(), b["hi"].max() + step / 2, step * (2 if x == "stars" else 4)))
        ax.set_xlabel(lab)
        ax.set_ylabel("mean accuracy %")
        ax.set_title(f"Accuracy by {x}", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Comfort zone (green, 93%+) and push zone (orange, 85-93%); number = plays in the bin",
                 fontweight="bold")
    save(fig, out, "4_comfort_zones.png")


def chart_fails(rall, out):
    if rall.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fl = rall[~rall["passed"]]
    if len(fl):
        a1.hist(fl["completion"].dropna() * 100, bins=np.arange(0, 105, 10), color=C2, edgecolor=SURF, linewidth=2)
    a1.set_xlabel("% of map completed when failing")
    a1.set_ylabel("fails")
    a1.set_title(f"Where fails happen (n={len(fl)})")
    b = rall.copy()
    b["bin"] = (b["stars"] // 0.5) * 0.5
    g = b.groupby("bin").agg(n=("passed", "size"), fail=("passed", lambda s: 100 * (1 - s.mean()))).reset_index()
    a2.bar(g["bin"] + 0.25, g["fail"], width=0.42, color=C2, edgecolor=SURF)
    for _, r in g.iterrows():
        a2.text(r["bin"] + 0.25, r["fail"] + 1, f"n={int(r['n'])}", ha="center", fontsize=8, color=INK2)
    a2.set_xlabel("star rating (with mods)")
    a2.set_ylabel("fail rate %")
    a2.set_title("Fail rate by star rating (all recent attempts)")
    save(fig, out, "5_fails.png")


def chart_activity(monthly, rank, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    if not monthly.empty:
        a1.bar(monthly["month"], monthly["count"], width=25, color=C1)
        a1.set_title("Plays per month")
        a1.set_ylabel("plays")
    if rank:
        a2.plot(range(-len(rank) + 1, 1), rank, color=C1)
        a2.invert_yaxis()
        a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a2.set_title("Global rank, last 90 days")
        a2.set_xlabel("days ago")
        a2.set_ylabel("rank (lower is better)")
    save(fig, out, "6_activity.png")


def chart_eras(top, out, labels):
    eras = [e for e in [ERA_OLD, ERA_NEW] if (top["era"] == e).any()]
    if len(eras) < 2:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (x, lab) in zip(axes, [("stars", "stars"), ("aim", "aim"), ("speed", "speed"), ("eff_bpm", "effective BPM")]):
        data = [top[top["era"] == e][x].dropna() for e in eras]
        bp = ax.boxplot(data, tick_labels=[labels[e] for e in eras], patch_artist=True, widths=0.5,
                        medianprops={"color": INK}, flierprops={"markersize": 3})
        for patch, col in zip(bp["boxes"], [C4, C1]):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
        ax.set_title(lab)
    fig.suptitle("Top plays: old vs new", fontweight="bold")
    save(fig, out, "7_old_vs_new.png")


def chart_sessions(sessions, out):
    """Per-session misses and 100s at equal difficulty, as two panels so each keeps its own scale."""
    if sessions is None or len(sessions) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = range(1, len(sessions) + 1)
    for ax, col, se, lab, c in [(axes[0], "adj_miss_pct", "se_miss_pct", "misses per 100 objects", C2),
                                (axes[1], "adj_ok_pct", "se_ok_pct", "100s/50s per 100 objects", C1)]:
        ax.errorbar(x, sessions[col], yerr=2 * sessions[se], color=c, marker="o", markersize=7, capsize=3, linewidth=1.5)
        ax.set_xticks(list(x))
        ax.set_xlabel("session")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab}, at equal difficulty")
    fig.suptitle("Sessions compared at equal difficulty (lower is better, bars are about 95%)", fontweight="bold")
    save(fig, out, "8_sessions.png")


def chart_findings(scan, out):
    """Tested weaknesses with their intervals, in each check's own unit, worst first."""
    fs = scan["findings"]
    if not fs:
        return
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(fs) + 1.6))
    y = list(range(len(fs)))[::-1]
    for yi, f in zip(y, fs):
        ax.plot([f.lo, f.hi], [yi, yi], color=C1, linewidth=2)
        ax.scatter([f.effect], [yi], s=60, color=C1, edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f.drill or f.key for f in fs])
    ax.set_xlabel("effect per typical step (accuracy pp, or errors per 100 objects)")
    ax.set_title("What the scan found (95% intervals)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "9_weaknesses.png")


def chart_targets(c, lists, ceilings, att, out):
    """Where recommended maps sit against what you actually play."""
    fig, ax = plt.subplots(figsize=(9, 6))
    if att is not None and len(att):
        ax.scatter(att["aim"], att["speed"], s=16, color="#cfcdc6", label="your attempts", zorder=1)
    style = [("warmup", C3, "warm-up"), ("focus_aim", C2, "aim focus"), ("focus_speed", C1, "speed focus")]
    for role, col, lab in style:
        maps = lists.get(role) or []
        if maps:
            ax.scatter([m["aim"] for m in maps], [m["speed"] for m in maps], s=70, color=col, edgecolor=SURF,
                       linewidth=1.5, label=lab, zorder=3)
    if ceilings:
        if ceilings.get("aim"):
            ax.axvline(ceilings["aim"], color=C2, linestyle="--", linewidth=1.2, label="aim ceiling")
        if ceilings.get("speed"):
            ax.axhline(ceilings["speed"], color=C1, linestyle="--", linewidth=1.2, label="speed ceiling")
    ax.set_xlabel("aim difficulty")
    ax.set_ylabel("speed difficulty")
    ax.set_title("Recommended maps against what you play")
    ax.legend(loc="upper left")
    save(fig, out, "10_recommendations.png")
