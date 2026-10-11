# -*- coding: utf-8 -*-
"""
paths.py - shared CSV finder for the analysis scripts
======================================================
Purpose: pick the right measurement CSV from results/csv/ (relative to the
current directory, so run scripts from the repo root).

Files are ordered by the YYYYMMDD_HHMM timestamp in their file name and the
newest one is used.

For the tree (timing) measurement, CPU files are EXCLUDED: the timing model
belongs to the GPU, and accidentally reading a CPU run would make the whole
peak-point calculation meaningless.

Inputs : results/csv/union_<ts>.csv      (from 01_union.py)
         results/csv/tree_<device>_<ts>.csv (from 02_tree_size.py)
Outputs: file paths (nothing is written).
Run    : python paths.py   (prints which files would be picked)
         In other scripts: from paths import union_csv, tree_csv
"""
import glob, os
import re


def _find(pattern, label, exclude=()):
    paths = sorted(glob.glob(f"results/csv/{pattern}"), key=lambda p: (re.findall(r"\d{8}_\d{4}", os.path.basename(p)) or [""])[-1])
    excluded = [p for p in paths if any(e.lower() in os.path.basename(p).lower()
                                        for e in exclude)]
    remaining = [p for p in paths if p not in excluded]

    if not remaining:
        if excluded:
            raise FileNotFoundError(
                f"Only excluded files were found for {label} "
                f"({', '.join(os.path.basename(p) for p in excluded)}).\n"
                f"This measurement must be done on a GPU - rerun 02_tree_size.py "
                f"in a CUDA environment.")
        raise FileNotFoundError(
            f"{label} CSV not found (searched: results/csv/{pattern}).\n"
            f"Run the corresponding measurement script first.")

    if excluded:
        print(f"  ({len(excluded)} CPU file(s) skipped)")
    print(f"reading {label:<10}: {remaining[-1]}")
    return remaining[-1]


def union_csv():
    # "union_[0-9]*" so that union_large_n_*.csv (from 03) is never matched.
    return _find("union_[0-9]*.csv", "union")


def tree_csv(file=None):
    """Pick the timing CSV.
    1) If file is given, use it directly (e.g. "tree_..._1941.csv").
    2) Otherwise drop CPU runs and, among the files with the MOST distinct N,
       pick the LEAST NOISY one (smallest median std_ms/ms).
    Incomplete and noisy runs used to produce a wrong timing model."""
    import pandas as pd
    if file:
        path = file if os.path.exists(file) else os.path.join("results/csv", file)
        print(f"reading {'tree':<10}: {path}  (selected manually)")
        return path
    paths = [p for p in sorted(glob.glob("results/csv/tree_*.csv"))
             if "cpu" not in os.path.basename(p).lower()]
    if not paths:
        return _find("tree_*.csv", "tree", exclude=("CPU",))
    candidates = []
    for p in paths:
        try:
            d = pd.read_csv(p)
            candidates.append((d.N.nunique(), -float((d.std_ms / d.ms).median()), p))
        except Exception:
            continue
    most_n = max(c[0] for c in candidates)
    n, neg_noise, best = max(c for c in candidates if c[0] == most_n)
    dropped = len(candidates) - 1
    print(f"reading {'tree':<10}: {best}  ({n} distinct N, median relative std "
          f"{-neg_noise:.3f}, {dropped} file(s) dropped)")
    if -neg_noise > 0.12:
        print("  ! WARNING: this file is noisy (>0.12). Rerun 02 with the charger "
              "plugged in, in high-performance mode and with no other programs open.")
    return best


if __name__ == "__main__":
    print("file test\n" + "-"*40)
    for f in (union_csv, tree_csv):
        try:
            f()
        except FileNotFoundError as e:
            print(f"MISSING: {e}")
