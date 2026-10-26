"""Ranks result folders against every archived bbob dataset (cached by
archive_all.py) and against the virtual best solver (VBS: for every
dimension, function, target and budget, the best fraction any archived
algorithm reached).

Metric: fraction of (function, run, target) reached within B*D, averaged
over f1..f24 and 51 targets; only datasets covering all 24 functions in a
dimension are ranked there. Usage:
    .venv/bin/python rank_all.py NAME=FOLDER [NAME=FOLDER ...] [--top 5] [--eligible]

With --eligible, each budget's rank and leaders count only datasets whose
unsolved runs all ran at least that long (archive_budgets.py).
"""
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import cocopp  # noqa: E402

from archive_all import BUDGETS, CACHE, DIMS, table  # noqa: E402

cocopp.genericsettings.verbose = False


def main():
    args = sys.argv[1:]
    top = 5
    eligible = "--eligible" in args
    if eligible:
        args.remove("--eligible")
    if "--top" in args:
        i = args.index("--top")
        top = int(args[i + 1])
        del args[i : i + 2]
    arch, budget = {}, {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npy"))):
        if p.endswith(".budget.npy"):
            continue
        name = os.path.basename(p)[:-4].replace("__", "/")
        arch[name] = np.load(p)
        bp = p[:-4] + ".budget.npy"
        if os.path.exists(bp):
            budget[name] = np.load(bp)
    ours = {}
    for a in args:
        name, folder = a.split("=", 1)
        ours[name] = table(cocopp.load(folder))
    print(f"{len(arch)} archived datasets")
    for di, d in enumerate(DIMS):
        complete = {n: t for n, t in arch.items() if not np.isnan(t[di, :, 0, 0]).any()}
        if not complete:
            continue
        stack = np.stack([t[di] for t in complete.values()])  # alg, f, target, budget
        vbs = np.nanmax(stack, axis=0).mean(axis=(0, 1))
        scores = {n: t[di].mean(axis=(0, 1)) for n, t in complete.items()}
        # Budget-eligible competitors per budget: every unsolved run of the
        # dataset ran at least B·D evaluations in this dimension.
        elig = [{n for n in complete if n in budget and budget[n][di] >= b} for b in BUDGETS]
        mine = {n: t[di].mean(axis=(0, 1)) for n, t in ours.items() if not np.isnan(t[di, :, 0, 0]).any()}
        if not mine:
            continue
        print(f"\n### {d}-D ({len(complete)} archived algorithms with all 24 functions)")
        print("| | " + " | ".join(f"{b:g}·D" for b in BUDGETS) + " | mean |")
        print("|---|" + "---|" * (len(BUDGETS) + 1))
        print("| virtual best (all archived) | " + " | ".join(f"{100 * x:.1f}" for x in vbs) + f" | {100 * vbs.mean():.1f} |")
        if eligible:
            print("| eligible datasets | " + " | ".join(str(len(e)) for e in elig) + " | |")
        for n, s in mine.items():
            if eligible:
                ranks = [1 + sum(scores[m][bi] > s[bi] + 1e-12 for m in elig[bi]) for bi in range(len(BUDGETS))]
            else:
                ranks = [1 + sum(v[bi] > s[bi] + 1e-12 for v in scores.values()) for bi in range(len(BUDGETS))]
            mrank = 1 + sum(v.mean() > s.mean() + 1e-12 for v in scores.values())
            print(f"| **{n}** | " + " | ".join(f"{100 * s[bi]:.1f} (#{ranks[bi]})" for bi in range(len(BUDGETS))) + f" | {100 * s.mean():.1f} (#{mrank}) |")
        for bi, b in enumerate(BUDGETS):
            pool = {n: scores[n] for n in elig[bi]} if eligible else scores
            best = sorted(pool.items(), key=lambda kv: -kv[1][bi])[:top]
            print(f"| best at {b:g}·D | " + "; ".join(f"{n.split('/')[-1][:28]} {100 * v[bi]:.1f}" for n, v in best) + " |")
        best = sorted(scores.items(), key=lambda kv: -kv[1].mean())[:top]
        print("| best mean | " + "; ".join(f"{n.split('/')[-1][:28]} {100 * v.mean():.1f}" for n, v in best) + " |")


if __name__ == "__main__":
    main()
