"""Renders the README figures (SVG) into ../docs/figures from the result
folders of final.sh and the archive cache of archive_all.py.

Usage: python figures.py OUT [DIM:OUT ...]
Reads exdata/OUT/{P,U}{dim} (development and validation instances) for
every dimension; an optional DIM:OUT takes that dimension from another run.
The ECDF reference curves are computed from cocopp's archive and cached in
.fig_cache/ (the first run downloads six datasets).
"""
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import cocopp  # noqa: E402

from archive_all import CACHE, DIMS, TARGETS, table  # noqa: E402

cocopp.genericsettings.verbose = False
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")
FIG_CACHE = os.path.join(HERE, ".fig_cache")

# Light-mode palette, fixed slot order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
NEUTRAL = "#f0efec"
# ECDF comparison algorithms: label, archive name, colour.
REFS = [
    ("HCMA (2013)", "2013/HCMA_loshchilov_noiseless", ORANGE),
    ("BIPOP-aCMA-STEP (2013)", "2013/BIPOP-aCMA-STEP_loshchilov_noiseless", AQUA),
    ("lq-CMA-ES (2020)", "2020/lq-CMA-ES_Hansen", YELLOW),
    ("SLSQP+lq-CMA-ES (2020)", "2020/SLSQP+lq-CMA-ES_Hansen", VIOLET),
    ("NELDERDOERR (2009)", "2009/NELDERDOERR_doerr_noiseless", MAGENTA),
    ("MCS (2009, MATLAB)", "2009/MCS_huyer_noiseless", GREEN),
]
GRIDB = np.logspace(0, 5, 101)  # budgets in units of D
DEV, VAL = "this work, development instances", "this work, validation instances"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
    "svg.fonttype": "none",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.12)
    if os.environ.get("FIG_PREVIEW"):
        os.makedirs(os.environ["FIG_PREVIEW"], exist_ok=True)
        fig.savefig(os.path.join(os.environ["FIG_PREVIEW"], name[:-4] + ".png"), dpi=130, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", name)


def load(out, override, s):
    """All dimensions of one instance set (P or U) as one DataSetList."""
    dsl = None
    for d in DIMS:
        path = os.path.join(HERE, "exdata", override.get(d, out), f"{s}{d}")
        if os.path.isdir(path):
            part = cocopp.load(path)
            dsl = part if dsl is None else dsl + part
    return dsl


def ecdf_curves(dsl):
    """Fraction of (function, run, target) reached within b*D, per dimension,
    on GRIDB; only dimensions with all 24 functions."""
    out = {}
    for d in DIMS:
        per_f = []
        for ds in dsl:
            if ds.dim == d and 1 <= ds.funcId <= 24:
                e = np.concatenate([np.asarray(x, float) for x in ds.detEvals(TARGETS)])
                e = np.where(np.isfinite(e), e, np.inf) / d
                per_f.append((ds.funcId, np.mean(e[None, :] <= GRIDB[:, None], axis=1)))
        if len({f for f, _ in per_f}) == 24:
            out[d] = np.mean([c for _, c in per_f], axis=0)
    return out


def archived_curves(name):
    os.makedirs(FIG_CACHE, exist_ok=True)
    path = os.path.join(FIG_CACHE, name.replace("/", "__") + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        return {int(k): z[k] for k in z.files}
    # find() matches substrings ("lq-CMA-ES_Hansen" also finds
    # "SLSQP+lq-CMA-ES_Hansen"), so keep the exact name only.
    hit = [n for n in cocopp.archives.bbob.find(name.split("/")[-1]) if n == name or n.startswith(name + ".")]
    assert len(hit) == 1, (name, hit)
    curves = ecdf_curves(cocopp.load(cocopp.archives.bbob.get(hit[0])))
    np.savez(path, **{str(k): v for k, v in curves.items()})
    return curves


def load_archive_tables():
    arch = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npy"))):
        if not p.endswith(".budget.npy"):
            arch[os.path.basename(p)[:-4].replace("__", "/")] = np.load(p)
    return arch


def complete(arch, di):
    return {n: t[di] for n, t in arch.items() if not np.isnan(t[di, :, 0, 0]).any()}


def fig_ranking(dev, val, arch):
    """Figure 1: mean score over the five budgets per dimension, with the
    validation rank among the archived datasets covering all 24 functions."""
    rows = []
    for di, d in enumerate(DIMS):
        if np.isnan(val[di, :, 0, 0]).any():
            continue
        comp = complete(arch, di)
        scores = {n: t.mean() for n, t in comp.items()}
        best = max(scores, key=scores.get)
        vbs = np.nanmax(np.stack(list(comp.values())), axis=0).mean()
        v = val[di].mean()
        rank = 1 + sum(s > v + 1e-12 for s in scores.values())
        rows.append((d, dev[di].mean(), v, scores[best], best.split("/")[-1].split("_")[0], vbs, rank, len(comp)))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(rows))
    ax.plot(x, [r[5] * 100 for r in rows], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="virtual best of the archive")
    ax.plot(x, [r[3] * 100 for r in rows], color=ORANGE, marker="o", ms=6, mec=SURFACE, mew=1.5, label="best archived algorithm")
    ax.plot(x, [r[1] * 100 for r in rows], color=BLUE, lw=1.2, ls=(0, (2, 2)), marker="o", ms=5, mfc=SURFACE, mec=BLUE, label=DEV)
    ax.plot(x, [r[2] * 100 for r in rows], color=BLUE, marker="o", ms=6, mec=SURFACE, mew=1.5, label=VAL)
    for i, r in enumerate(rows):
        ax.annotate(f"#{r[6]}/{r[7]}", (i, r[2] * 100), textcoords="offset points", xytext=(0, 9), ha="center", color=INK, fontsize=8.5, weight="semibold")
        ax.annotate(r[4], (i, r[3] * 100), textcoords="offset points", xytext=(0, -15), ha="center", va="top", color=INK2, fontsize=7.5)
    ax.set_xticks(x, [f"{r[0]}-D" for r in rows])
    ax.set_ylabel("targets reached, mean over budgets (%)")
    ax.set_title("Rank among archived BBOB datasets, per dimension", loc="left")
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    ax.set_ylim(max(0, min(r[2] for r in rows) * 100 - 18), 90)
    save(fig, "fig1_ranking.svg")
    return rows


def fig_generalization(dev, val):
    """Figure 2: every (dimension, budget) cell, development against
    validation instances."""
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=0.9)
    cols = {2: BLUE, 3: ORANGE, 5: AQUA, 10: YELLOW, 20: MAGENTA, 40: GREEN}
    gap = 0.0
    for di, d in enumerate(DIMS):
        if np.isnan(dev[di, :, 0, 0]).any() or np.isnan(val[di, :, 0, 0]).any():
            continue
        a, b = dev[di].mean(axis=(0, 1)) * 100, val[di].mean(axis=(0, 1)) * 100
        gap = max(gap, np.abs(a - b).max())
        ax.scatter(a, b, s=36, color=cols[d], edgecolor=SURFACE, linewidth=1.2, label=f"{d}-D", zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("development instances 1–5, 71–80 (%)")
    ax.set_ylabel("validation instances 101–115 (%)")
    ax.set_title("Development against validation instances", loc="left")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "fig2_generalization.svg")
    return gap


def fig_ecdf(dev_dsl, val_dsl):
    """Figure 3: runtime ECDFs per dimension."""
    dev, val = ecdf_curves(dev_dsl), ecdf_curves(val_dsl)
    refs = [(lab, archived_curves(name), col) for lab, name, col in REFS]
    dims = [d for d in DIMS if d in val]
    cols = 3
    rows_n = (len(dims) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(9.6, 3.0 * rows_n), sharey=True, squeeze=False)
    for k, d in enumerate(dims):
        ax = axes[k // cols][k % cols]
        for lab, cur, col in refs:
            if d in cur:
                ax.plot(GRIDB, cur[d] * 100, color=col, lw=1.3, label=lab)
        if d in dev:
            ax.plot(GRIDB, dev[d] * 100, color=BLUE, lw=1.2, ls=(0, (2, 2)), label=DEV)
        ax.plot(GRIDB, val[d] * 100, color=BLUE, lw=2.2, label=VAL)
        ax.set_xscale("log")
        ax.set_xlim(1, 1e5)
        ax.set_ylim(0, 100)
        ax.set_title(f"{d}-D", loc="left")
        ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5], ["1", "10", "10²", "10³", "10⁴", "10⁵"])
        if k % cols == 0:
            ax.set_ylabel("targets reached (%)")
        if k // cols == rows_n - 1:
            ax.set_xlabel("evaluations / dimension")
    for k in range(len(dims), rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.suptitle("Runtime ECDF over all 24 BBOB functions and 51 targets (10² … 10⁻⁸)", x=0.01, ha="left", y=1.11, fontsize=10.5, weight="semibold", color=INK)
    fig.tight_layout()
    save(fig, "fig3_ecdf.svg")


def fig_heatmap(val, arch):
    """Figure 4: our validation score minus that of the strongest archived
    algorithm of each dimension (best mean), per function, mean over the
    budgets: blue where we lead, red where we trail."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    dims = [di for di in range(len(DIMS)) if not np.isnan(val[di, :, 0, 0]).any()]
    gap = np.full((24, len(dims)), np.nan)
    leaders = []
    for j, di in enumerate(dims):
        comp = complete(arch, di)
        lead = max(comp, key=lambda n: comp[n].mean())
        leaders.append(lead.split("/")[-1].split("_")[0])
        gap[:, j] = (val[di].mean(axis=(1, 2)) - comp[lead].mean(axis=(1, 2))) * 100
    cmap = LinearSegmentedColormap.from_list("div", ["#c43a3a", "#f2a3a0", NEUTRAL, "#86b6ef", "#1c5cab"])
    lim = max(5, np.nanmax(np.abs(gap)))
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(dims), 7.6))
    im = ax.imshow(gap, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    names = ["sphere", "ellipsoid", "Rastrigin", "Büche-Rastrigin", "linear slope", "attractive sector", "step ellipsoid",
             "Rosenbrock", "Rosenbrock rot.", "ellipsoid rot.", "discus", "bent cigar", "sharp ridge", "different powers",
             "Rastrigin rot.", "Weierstrass", "Schaffer F7", "Schaffer F7 ill-c.", "Griewank-Rosenbrock", "Schwefel",
             "Gallagher 101", "Gallagher 21", "Katsuura", "Lunacek bi-Rastrigin"]
    ax.set_yticks(range(24), [f"f{i + 1} {names[i]}" for i in range(24)], fontsize=7.5)
    ax.set_xticks(range(len(dims)), [f"{DIMS[di]}-D\nvs {lead}" for di, lead in zip(dims, leaders)], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.grid(False)
    for i in range(24):
        for j in range(len(dims)):
            v = gap[i, j]
            ax.text(j, i, "0" if round(v) == 0 else f"{v:+.0f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.6 * lim else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("points vs the dimension's strongest archived algorithm", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Per-function difference to the strongest archived algorithm\nof each dimension (validation instances, mean over 5 budgets)", loc="left", pad=34)
    save(fig, "fig4_function_gap.svg")


def main():
    out = sys.argv[1]
    override = {int(k): v for k, v in (a.split(":") for a in sys.argv[2:])}
    arch = load_archive_tables()
    dev_dsl, val_dsl = load(out, override, "P"), load(out, override, "U")
    dev, val = table(dev_dsl), table(val_dsl)
    for r in fig_ranking(dev, val, arch):
        print(f"{r[0]}-D dev {100 * r[1]:.1f} val {100 * r[2]:.1f} rank {r[6]}/{r[7]} best {r[4]} {100 * r[3]:.1f} VBS {100 * r[5]:.1f}")
    print(f"largest |dev - val| over (dimension, budget): {fig_generalization(dev, val):.1f}")
    fig_ecdf(dev_dsl, val_dsl)
    fig_heatmap(val, arch)


if __name__ == "__main__":
    main()
