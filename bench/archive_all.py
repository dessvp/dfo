"""Downloads every bbob dataset in the cocopp archive and caches, per
dataset, the fraction of runs reaching each of COCO's 51 targets within
B*D evaluations: array [dim, function, target, budget] (NaN if missing),
in .archive_cache/<name>.npy. Safe to rerun; finished datasets are skipped.

Usage: .venv/bin/python archive_all.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import cocopp  # noqa: E402

cocopp.genericsettings.verbose = False

DIMS = [2, 3, 5, 10, 20, 40]
BUDGETS = [10, 100, 1000, 10000, 100000]
TARGETS = 10.0 ** np.linspace(2, -8, 51)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".archive_cache")


def table(dsl):
    t = np.full((len(DIMS), 24, len(TARGETS), len(BUDGETS)), np.nan)
    for ds in dsl:
        if ds.dim not in DIMS or not 1 <= ds.funcId <= 24:
            continue
        di, fi = DIMS.index(ds.dim), ds.funcId - 1
        for ti, e in enumerate(ds.detEvals(TARGETS)):
            e = np.asarray(e, dtype=float)
            for bi, b in enumerate(BUDGETS):
                t[di, fi, ti, bi] = np.mean(np.isfinite(e) & (e <= b * ds.dim))
    return t


def main():
    os.makedirs(CACHE, exist_ok=True)
    names = cocopp.archives.bbob.find("")
    for k, name in enumerate(names, 1):
        out = os.path.join(CACHE, name.replace("/", "__").replace(".tgz", "") + ".npy")
        if os.path.exists(out):
            continue
        try:
            path = cocopp.archives.bbob.get(name)
            dsl = cocopp.load(path)
            np.save(out, table(dsl))
            print(f"{k}/{len(names)} {name}", flush=True)
            del dsl
        except Exception as exc:
            print(f"{k}/{len(names)} {name} FAILED: {exc}", flush=True, file=sys.stderr)


if __name__ == "__main__":
    main()
