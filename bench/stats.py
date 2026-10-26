"""The README's statistics that are not in rank_all.py's tables: bootstrap
intervals of the validation mean, the mean rank among datasets that ran
every budget, and the f24 results at 10^5·D.

Usage: python stats.py OUT [DIM:OUT ...]   (reads exdata/OUT/{P,U}{dim})
"""
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import cocopp  # noqa: E402

from archive_all import BUDGETS, CACHE, DIMS, TARGETS, table  # noqa: E402

cocopp.genericsettings.verbose = False
HERE = os.path.dirname(os.path.abspath(__file__))
RESAMPLES = 1000


def runs(dsl, d):
    """Evaluations to each target per function: {f: array [target, run]}."""
    e = {}
    for ds in dsl:
        if ds.dim == d and 1 <= ds.funcId <= 24:
            e.setdefault(ds.funcId, []).append(np.array([np.asarray(x, float) for x in ds.detEvals(TARGETS)]))
    return {f: np.concatenate(v, axis=1) for f, v in e.items()}


def mean_score(e, d, idx):
    s = [np.mean([np.mean(np.isfinite(x) & (x <= b * d)) for b in BUDGETS]) for x in (e[f][:, idx[f]] for f in range(1, 25))]
    return 100 * np.mean(s)


def main():
    out = sys.argv[1]
    override = {int(k): v for k, v in (a.split(":") for a in sys.argv[2:])}
    arch, budget = {}, {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npy"))):
        if p.endswith(".budget.npy"):
            continue
        name = os.path.basename(p)[:-4].replace("__", "/")
        arch[name] = np.load(p)
        if os.path.exists(p[:-4] + ".budget.npy"):
            budget[name] = np.load(p[:-4] + ".budget.npy")
    print(f"{len(arch)} archived datasets")
    bi = BUDGETS.index(100000)
    for di, d in enumerate(DIMS):
        path = os.path.join(HERE, "exdata", override.get(d, out), f"U{d}")
        if not os.path.isdir(path):
            continue
        dsl = cocopp.load(path)
        rng = np.random.default_rng(d)  # per dimension, so a subset reproduces it
        e = runs(dsl, d)
        full = {f: np.arange(x.shape[1]) for f, x in e.items()}
        mine = mean_score(e, d, full)
        boot = [mean_score(e, d, {f: rng.integers(0, x.shape[1], x.shape[1]) for f, x in e.items()}) for _ in range(RESAMPLES)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        comp = {n: t[di] for n, t in arch.items() if not np.isnan(t[di, :, 0, 0]).any()}
        allb = [n for n in comp if n in budget and budget[n][di] >= BUDGETS[-1]]
        rank_all = 1 + sum(100 * comp[n].mean() > mine + 1e-9 for n in comp)
        rank_full = 1 + sum(100 * comp[n].mean() > mine + 1e-9 for n in allb)
        solved = int(np.sum(np.isfinite(e[24][-1]) & (e[24][-1] <= BUDGETS[-1] * d)))
        f24 = 100 * table(dsl)[di, 23, :, bi].mean()
        best24 = sorted(((100 * t[23, :, bi].mean(), n.split("/")[-1]) for n, t in comp.items()), reverse=True)[:2]
        print(f"{d}-D validation mean {mine:.1f}, 95% bootstrap [{lo:.1f}, {hi:.1f}]; mean rank {rank_all}/{len(comp)} "
              f"(among the {len(allb)} that ran every budget: {rank_full}); "
              f"f24 at 1e5·D: {f24:.0f}% of targets, {solved}/{e[24].shape[1]} solved to 1e-8, "
              f"best archived {best24[0][0]:.0f}% ({best24[0][1]}, {best24[1][1]} {best24[1][0]:.0f}%)")


if __name__ == "__main__":
    main()
