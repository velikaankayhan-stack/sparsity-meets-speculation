# -*- coding: utf-8 -*-
"""
10_speed_model.py - Speed model (no GPU needed) and comparison with real measurements.

Inputs (all in results/csv):
  * union, quality, acceptance : from 09_multi_model.py / 07_acceptance.py
                                 (measurements that do not depend on timing)
  * T(N) and draft time        : from 13_timing.py (timing_*.csv)
        -> both are measured in the SAME session; an older version mixed
           timings from different days.
  * real measurement (if any)  : from 12_real_speed.py (speed_summary.csv)

Model:
  T(N)   : non-decreasing curve in N fitted to the timings from 13 (isotonic
           regression); linear interpolation between measured N.
  One verification round processes k draft tokens + 1 correction = N = k+1 positions.
  Sparsity reduces the MLP weight reads (fraction F) to the union fraction U:
           T(k,p) = T(k+1) - T(1)*F*(1 - U(k+1, p))
  The draft runs k times:   round = k*t_draft + T(k,p)
  Speed = E(k) / round      E: from the measured acceptance sequences (non-repetitive prompts)
  Baseline: plain decoding, 1 token / T(1).

  This is an UPPER BOUND: on the sparse side it assumes weight reads shrink in
  proportion to time and that the predictor is free. The real measurement from
  12 (16-bit 1.5B) shows how well that assumption holds.

Run:     python 10_speed_model.py
Output:  results/csv/speed_model.csv
         (report figures are drawn from this CSV by 15_report_figures.py)
"""
import glob
import re
import os
import sys

import numpy as np
import pandas as pd

CSV_DIR = "results/csv" if os.path.isdir("results/csv") else "."
SPARSITY = [0.10, 0.20, 0.30, 0.50]
K_SIM = [1, 2, 3, 4, 6, 8, 12, 16, 21, 32, 48, 64, 128]
KS = np.arange(1, 129)
PS = np.round(np.linspace(0.10, 1.0, 91), 3)
FREE_FACTOR = 1.25     # verification with T(N) <= 1.25*T(1) counts as "free"

# name, quant, F_MLP, union/quality/acceptance file pattern, acceptance override,
# model short name, name used in 12
MODELS = [
    ("1.5B", "fp16", 0.749, "mm_Qwen2.5-1.5B_*_20261005_0126.csv", None,
     "Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("1.5B", "4bit", 0.515, "mm_Qwen2.5-1.5B_*_20261005_0207.csv", None,
     "Qwen2.5-1.5B", "Qwen2.5-1.5B (4-bit)"),
    ("3B",   "4bit", 0.606, "mm_Qwen2.5-3B_*_20261005_0142.csv", None,
     "Qwen2.5-3B", "Qwen2.5-3B (4-bit)"),
    ("7B",   "4bit", 0.807, "b7_*_20261005_0014.csv", ("acceptance_20261005_0100.csv", "7B"),
     "Qwen2.5-7B", "Qwen2.5-7B (4-bit)"),
    ("14B",  "4bit", 0.728, "mm_Qwen2.5-14B_*_20261004_2242.csv", None,
     "Qwen2.5-14B", "Qwen2.5-14B (4-bit)"),
]


def read_measure(pattern, kind):
    d = sorted(glob.glob(os.path.join(CSV_DIR, pattern.replace("*", kind))))
    if not d:
        raise FileNotFoundError(pattern.replace("*", kind))
    return pd.read_csv(d[0])


# 13_timing.py measurements used in the report (same session, with KV cache).
# Fixed files instead of "newest file", so that re-running 13 does not silently
# change the report results. Update this on purpose to use new measurements.
TIMING_FILES = {
    ("Qwen2.5-1.5B", "fp16"): "timing_Qwen2.5-1.5B_fp16_20261010_0135.csv",
    ("Qwen2.5-1.5B", "4bit"): "timing_Qwen2.5-1.5B_4bit_20261010_1350.csv",
    ("Qwen2.5-3B", "4bit"):   "timing_Qwen2.5-3B_4bit_20261010_0135.csv",
    ("Qwen2.5-7B", "4bit"):   "timing_Qwen2.5-7B_4bit_20261010_1350.csv",
    ("Qwen2.5-14B", "4bit"):  "timing_Qwen2.5-14B_4bit_20261009_2326.csv",
}


def read_timing(short, quant):
    """The 13_timing.py file used in the report for this model."""
    path = os.path.join(CSV_DIR, TIMING_FILES.get((short, quant), ""))
    return pd.read_csv(path) if os.path.isfile(path) else None


def read_real(name12, device):
    """NEWEST row of 12_real_speed.py output for this model and device."""
    path = os.path.join(CSV_DIR, "speed_summary.csv")
    if not os.path.exists(path):
        return None
    g = pd.read_csv(path)
    g = g[(g.model == name12) & (g.device == device)]
    return None if g.empty else g.sort_values("timestamp").iloc[-1]


def fit_alpha(df):
    par = {}
    for p in SPARSITY:
        a = df[(df.active_frac == p) & (df.k > 1) & (df.k <= 8)].groupby("k")["union"].mean()
        keff = np.log(1 - a.values) / np.log(1 - p)
        al, lc = np.polyfit(np.log(a.index.values), np.log(keff), 1)
        par[p] = (float(al), float(np.exp(lc)))
    return par


def quality_wall(qual, limit):
    qual = qual.groupby("active_frac", as_index=False).degradation_pct.mean()
    u = qual[qual.degradation_pct <= limit]
    return float(u.active_frac.min()) if len(u) else 1.0


def timing_curve(timing):
    """T(N): closest non-decreasing (monotone) curve in N to the measured timings,
    via isotonic regression (pool adjacent violators). The old "lower envelope"
    (smallest time to the right) picked the luckiest measurement at each N, which
    pulled T(1) down, while the draft time is a median, so c came out too large."""
    s = timing.sort_values("N")
    N, ms = s.N.values.astype(float), s.ms.values.astype(float)
    blocks = [[m, 1.0] for m in ms]            # [mean, weight]
    lengths = [1] * len(ms)
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            w = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [(blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / w, w]
            lengths[i] += lengths[i + 1]
            del blocks[i + 1], lengths[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    curve = np.concatenate([[b[0]] * n for b, n in zip(blocks, lengths)])
    return lambda n: float(np.interp(n, N, curve))


def compute(union, qual, timing, acc, F, td):
    par = fit_alpha(union)
    PL = np.array(SPARSITY)
    al_ = np.array([par[p][0] for p in PL]); c_ = np.array([par[p][1] for p in PL])

    def U(n, p):
        if p >= 1.0:
            return 1.0
        return 1 - (1 - p) ** (np.interp(p, PL, c_) * n ** np.interp(p, PL, al_))

    T = timing_curve(timing)
    T1 = T(1)
    d = acc[acc.distinct_4gram >= 0.5]
    ey = [d[f"tokens_k{k}"].sum() / d[f"rounds_k{k}"].sum() for k in K_SIM]
    E = lambda k: float(np.interp(np.log(k), np.log(K_SIM), ey))

    def speed(k, p):
        return E(k) / (k * td + T(k + 1) - T1 * F * (1 - U(k + 1, p))) * 1000

    def best(pmin):
        result = (0.0, None, None)
        for p in PS[PS >= pmin - 1e-9][::-1]:          # on ties, prefer less sparsity
            h = np.array([speed(int(k), p) for k in KS]); i = int(h.argmax())
            if h[i] > result[0] * 1.001:
                result = (float(h[i]), int(KS[i]), float(p))
        return result

    W1, W5 = quality_wall(qual, 1.0), quality_wall(qual, 5.0)
    dense, b5, b1 = best(1.0), best(W5), best(W1)
    base = 1000 / T1                                     # plain decoding
    sparse5 = 1000 / (T1 * (1 - F * (1 - W5)))           # sparsity only, 5% wall
    Ns = sorted(timing.N.unique())
    free_N = max([n for n in Ns if T(n) <= FREE_FACTOR * T1] or [1])
    return dict(a_nonrep=d.a_greedy.mean(), wall_5=W5, wall_1=W1,
                alpha_p20=par[0.2][0], T1_ms=T1, draft_ms=td, c=td / T1,
                free_N=free_N, T_N9_ratio=T(9) / T1,
                best_k=dense[1], speed=dense[0],
                gain_5=b5[0] / dense[0], k_5=b5[1], active_5=b5[2],
                gain_1=b1[0] / dense[0],
                spec_only=dense[0] / base, sparse_only=sparse5 / base, combined=b5[0] / base,
                combined_k=b5[1])


def main():
    rows = []
    missing = []
    for name, quant, F, pattern, acc_override, short, name12 in MODELS:
        timing = read_timing(short, quant)
        if timing is None:
            missing.append(f"{short}:{quant}")
            continue
        device = timing.device.iloc[0]
        td = float(timing.draft_ms.iloc[0])
        union, qual = read_measure(pattern, "union"), read_measure(pattern, "quality")
        if acc_override:
            acc = pd.read_csv(os.path.join(CSV_DIR, acc_override[0]))
            acc = acc[acc.target.str.contains(acc_override[1])]
        else:
            acc = read_measure(pattern, "acceptance")
        r = compute(union, qual, timing, acc, F, td)
        r0 = compute(union, qual, timing, acc, F, 0.0)     # as if the draft were free
        r.update(free_draft_k=r0["best_k"], free_draft_gain_5=r0["gain_5"])
        g = read_real(name12, device)
        # r_spec: measured end to end; r_spec_timing / r_sparse / r_combined: computed from
        # the round times measured by 12 (sparse MLP included), measured E(k) and draft time
        rr = dict(r_spec=np.nan, r_spec_k=np.nan, r_spec_timing=np.nan, r_sparse=np.nan,
                  r_combined=np.nan, r_lossless=np.nan, r_draft_ms=np.nan)
        if g is not None:
            rr.update(r_spec=g.spec_real, r_spec_k=g.spec_k, r_lossless=g.lossless,
                      r_draft_ms=g.draft_ms)
            if "ideal_sparse" in g and pd.notna(g.get("ideal_sparse")):
                rr.update(r_spec_timing=g.ideal_spec, r_sparse=g.ideal_sparse,
                          r_combined=g.ideal_combined)
        rows.append(dict(model=name, quant=quant, device=device, F_MLP=F,
                         draft_quant=timing.draft_quant.iloc[0], **r, **rr))
    if missing:
        print("WARNING - not measured with 13_timing.py, NOT included: " + ", ".join(missing))
        print("        (Older timings are not used because they come from different sessions.)\n")
    if not rows:
        sys.exit("No timing_*.csv found. Run first: python 13_timing.py")

    df = pd.DataFrame(rows)
    print("SPEED MODEL - draft and target timings from the same session\n")
    for _, r in df.iterrows():
        print(f"{r.model:>5} {r.quant:<4} {r.device}")
        print(f"      T(1)={r.T1_ms:.1f} ms  draft({r.draft_quant})={r.draft_ms:.1f} ms  "
              f"c={r.c:.2f}  free verification N<={r.free_N}  a={r.a_nonrep:.3f}  "
              f"wall@5%={r.wall_5:.0%}")
        print(f"      MODEL  (baseline=1): speculative {r.spec_only:.2f}x (k={r.best_k})  "
              f"sparsity {r.sparse_only:.2f}x  combined {r.combined:.2f}x (k={r.combined_k})  "
              f"| sparsity gain on top of speculative {r.gain_5:.2f}x")
        print(f"      IF THE DRAFT WERE FREE: best k={int(r.free_draft_k)}  "
              f"sparsity gain {r.free_draft_gain_5:.2f}x")
        if pd.notna(r.r_spec):
            s = (f"| from measured round times: speculative {r.r_spec_timing:.2f}x  "
                 f"sparsity {r.r_sparse:.2f}x  combined {r.r_combined:.2f}x"
                 if pd.notna(r.r_sparse) else "sparsity not measured (4-bit)")
            print(f"      REAL (baseline=1): speculative end to end {r.r_spec:.2f}x (k={int(r.r_spec_k)})  {s}"
                  f"  | lossless {r.r_lossless:.3f}")
        else:
            print("      REAL: not measured (12_real_speed.py)")
        print()
    path = os.path.join(CSV_DIR, "speed_model.csv")
    df.to_csv(path, index=False)
    print("CSV:", path)
    print("Figures (fig06, fig09a/b): python 15_report_figures.py")



if __name__ == "__main__":
    main()
