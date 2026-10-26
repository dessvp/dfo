"""Records, for every archived bbob dataset, how long its runs really ran:
per dimension, the smallest budget (in evaluations / D) of any run that
ended without reaching the final target f_opt + 1e-8 (infinite when every
run solved every function). A dataset is "budget-eligible" at budget B·D
in a dimension when this value is at least B: its unsolved runs all ran at
least that long, so a failure there is the algorithm's, not its budget's.

Writes .archive_cache/<name>.budget.npy (shape [len(DIMS)], NaN where the
dimension is missing). Safe to rerun; finished datasets are skipped.
Usage: .venv/bin/python archive_budgets.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import cocopp  # noqa: E402

from archive_all import CACHE, DIMS  # noqa: E402

cocopp.genericsettings.verbose = False


def effective_budget(dsl):
    out = np.full(len(DIMS), np.nan)
    for di, d in enumerate(DIMS):
        sets = [ds for ds in dsl if ds.dim == d and 1 <= ds.funcId <= 24]
        if not sets:
            continue
        b = np.inf
        for ds in sets:
            solved = np.isfinite(np.asarray(ds.detEvals([1e-8])[0], dtype=float))
            maxe = np.asarray(ds.maxevals, dtype=float)
            # detEvals and maxevals are both per run, in the same order.
            if len(maxe) == len(solved):
                unsolved = maxe[~solved]
            else:
                unsolved = maxe if not solved.all() else np.array([])
            if unsolved.size:
                b = min(b, unsolved.min() / d)
        out[di] = b
    return out


def main():
    names = cocopp.archives.bbob.find("")
    for k, name in enumerate(names, 1):
        out = os.path.join(CACHE, name.replace("/", "__").replace(".tgz", "") + ".budget.npy")
        if os.path.exists(out):
            continue
        try:
            path = cocopp.archives.bbob.get(name)
            np.save(out, effective_budget(cocopp.load(path)))
            print(f"{k}/{len(names)} {name}", flush=True)
        except Exception as exc:
            print(f"{k}/{len(names)} {name} FAILED: {exc}", flush=True, file=sys.stderr)


if __name__ == "__main__":
    main()
