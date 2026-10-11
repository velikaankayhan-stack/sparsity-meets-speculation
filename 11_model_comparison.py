# -*- coding: utf-8 -*-
"""
11_model_comparison.py - Model size comparison (no GPU needed).

Puts the device-INDEPENDENT measurements of five model sizes (0.5B-14B) side by side:
  union law (alpha), quality wall, acceptance rate.
For speed/gain (device dependent) see 10_speed_model.py.

Outputs:
  results/csv/model_comparison.csv
  results/figures/fig07_model_size.png   (4 panels, plus fig07_panelA..D.png)

Run: python 11_model_comparison.py
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = "results/csv"
SPARSITY = [0.10, 0.20, 0.30, 0.50]

# name, quant, parameters, union, quality, acceptance (file, target filter) - None: not available
RUNS = [
    ("0.5B", "fp16", 0.49e9, ("union_20260922_2332.csv", "Qwen2.5-0.5B"),
     ("quality_20260922_2343.csv", "Qwen2.5-0.5B"), None),
    ("1.5B", "fp16", 1.54e9, "mm_Qwen2.5-1.5B_union_20261005_0126.csv",
     "mm_Qwen2.5-1.5B_quality_20261005_0126.csv", "mm_Qwen2.5-1.5B_acceptance_20261005_0126.csv"),
    ("1.5B", "4bit", 1.54e9, "mm_Qwen2.5-1.5B_union_20261005_0207.csv",
     "mm_Qwen2.5-1.5B_quality_20261005_0207.csv", "mm_Qwen2.5-1.5B_acceptance_20261005_0207.csv"),
    ("3B", "4bit", 3.09e9, "mm_Qwen2.5-3B_union_20261005_0142.csv",
     "mm_Qwen2.5-3B_quality_20261005_0142.csv", "mm_Qwen2.5-3B_acceptance_20261005_0142.csv"),
    ("7B", "4bit", 7.62e9, "b7_union_20261005_0014.csv",
     "b7_quality_20261005_0014.csv", ("acceptance_20261005_0100.csv", "7B")),
    ("14B", "4bit", 14.77e9, "mm_Qwen2.5-14B_union_20261004_2242.csv",
     "mm_Qwen2.5-14B_quality_20261004_2242.csv", "mm_Qwen2.5-14B_acceptance_20261004_2242.csv"),
]
# categorical colours in fixed order (dataviz palette, CVD-checked); fp16 uses a dashed line
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def read(source):
    if source is None:
        return None
    if isinstance(source, tuple):
        df = pd.read_csv(os.path.join(CSV_DIR, source[0]))
        col = "target" if "target" in df.columns else "model"
        return df[df[col].astype(str).str.contains(source[1], regex=False)]
    return pd.read_csv(os.path.join(CSV_DIR, source))


def alpha(union, p):
    a = union[(union.active_frac == p) & (union.k > 1) & (union.k <= 8)].groupby("k")["union"].mean()
    keff = np.log(1 - a.values) / np.log(1 - p)
    return float(np.polyfit(np.log(a.index.values), np.log(keff), 1)[0])


def quality_wall(qual, limit):
    u = qual[qual.degradation_pct <= limit]
    return float(u.active_frac.min()) if len(u) else 1.0


def main():
    rows, data = [], []
    for name, quant, npar, fu, fq, fa in RUNS:
        union, qual, acc = read(fu), read(fq), read(fa)
        qual = qual.groupby("active_frac", as_index=False).agg(
            degradation_pct=("degradation_pct", "mean"), ppl=("ppl", "mean"))
        r = dict(model=name, quant=quant, params_billion=npar / 1e9,
                 **{f"alpha_p{int(p * 100)}": alpha(union, p) for p in SPARSITY},
                 U_p20_N4=float(union[(union.active_frac == 0.2) & (union.k == 4)]["union"].mean()),
                 U_p20_N8=float(union[(union.active_frac == 0.2) & (union.k == 8)]["union"].mean()),
                 ppl_base=float(qual[qual.active_frac == 1.0].ppl.iloc[0]),
                 degradation_active50=float(qual[np.isclose(qual.active_frac, 0.5)].degradation_pct.iloc[0]),
                 wall_1=quality_wall(qual, 1.0), wall_5=quality_wall(qual, 5.0))
        if acc is not None:
            d = acc[acc.distinct_4gram >= 0.5]
            r.update(a_all=float(acc.a_greedy.mean()), a_nonrep=float(d.a_greedy.mean()),
                     nonrep_prompts=len(d))
        rows.append(r)
        data.append((f"{name} {quant}", quant, npar, union, qual, acc))

    df = pd.DataFrame(rows)
    os.makedirs("results/figures", exist_ok=True)
    df.to_csv(os.path.join(CSV_DIR, "model_comparison.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nMODEL SIZE COMPARISON (device-independent measurements)\n")
    print(df[["model", "quant", "alpha_p20", "U_p20_N4", "U_p20_N8", "ppl_base",
              "degradation_active50", "wall_5", "wall_1", "a_nonrep"]]
          .round(3).to_string(index=False))

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    A, B, C, D = ax.ravel()
    for (label, quant, npar, union, qual, acc), c in zip(data, COLORS):
        ls = "--" if quant == "fp16" else "-"
        g = union[union.active_frac == 0.2].groupby("k")["union"].mean()
        A.plot(g.index, g.values * 100, ls, marker="o", ms=5, lw=2, color=c, label=label)
        k = qual.sort_values("active_frac")
        B.plot(k.active_frac * 100, k.degradation_pct, ls, marker="o", ms=5, lw=2,
               color=c, label=label)
    kk = np.arange(1, 9)
    A.plot(kk, (1 - 0.8 ** kk) * 100, ":", color="#777", lw=1.5, label="independent neurons")
    A.set(xlabel="tokens processed in the same round (N)", ylabel="neurons to fetch (%)",
          title="A — Union (20% active)")
    B.set_yscale("symlog", linthresh=1)
    for s in (1, 5):
        B.axhline(s, color="#555", lw=0.9, ls=":")
        B.text(101, s, f"{s}%", va="center", fontsize=8, color="#333")
    B.set(xlabel="active neuron fraction (%)", ylabel="perplexity degradation (%)",
          title="B — Quality wall")

    x = df.params_billion.values
    label = [f"{m}\n{n}" for m, n in zip(df.model, df.quant)]
    def xpos(r):        # keep the two runs of the same size from overlapping
        return r.params_billion * (0.93 if (r.model == "1.5B" and r.quant == "fp16") else
                                   1.07 if r.model == "1.5B" else 1.0)

    for i, r in df.iterrows():
        C.plot(xpos(r), r.alpha_p20, "o" if r.quant == "4bit" else "s",
               ms=9, color=COLORS[i], mec="white", mew=1.5)
        C.annotate(f"{r.alpha_p20:.2f}", (xpos(r), r.alpha_p20), textcoords="offset points",
                   xytext=(-14 if r.quant == "fp16" and r.model == "1.5B" else
                           14 if r.model == "1.5B" else 0, 9), ha="center", fontsize=8)
    C.set_xscale("log"); C.set_ylim(0.6, 0.9)
    C.set(xlabel="model size (log scale)", ylabel="α (20% active)",
          title="C — Union exponent α  (1 = independent, lower = more sharing)")
    m = df.a_nonrep.notna()
    for i, r in df[m].iterrows():
        D.plot(xpos(r), r.a_nonrep, "o" if r.quant == "4bit" else "s",
               ms=9, color=COLORS[i], mec="white", mew=1.5)
        D.annotate(f"{r.a_nonrep:.3f}", (xpos(r), r.a_nonrep),
                   textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    D.set_xscale("log"); D.margins(y=0.15)
    D.set(xlabel="model size (log scale)", ylabel="acceptance rate (non-repetitive prompts)",
          title="D — Acceptance rate (draft: 0.5B)")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    for a in (C, D):
        a.xaxis.set_major_locator(FixedLocator([0.5, 1.5, 3, 7, 14]))
        a.xaxis.set_major_formatter(FixedFormatter(["0.5B", "1.5B", "3B", "7B", "14B"]))
        a.xaxis.set_minor_locator(NullLocator())
        a.set_xlim(0.38, 19)
    for a in (A, B, C, D):
        a.grid(alpha=0.25)
    A.legend(fontsize=8.5, loc="lower right")
    fig.text(0.5, 0.005, "Dashed line / square = 16-bit, solid line / circle = 4-bit.  "
             "Union, quality and acceptance rate are device-independent measurements.",
             ha="center", fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = "results/figures/fig07_model_size.png"
    fig.savefig(path, dpi=150)
    # the report uses the panels separately: fig07_panelA..D.png
    fig.canvas.draw()
    for letter, axis in zip("ABCD", (A, B, C, D)):
        box = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"results/figures/fig07_panel{letter}.png", dpi=150, bbox_inches=box.expanded(1.03, 1.03))
    print(f"\nCSV   : {CSV_DIR}/model_comparison.csv\nFIGURE: {path} (+ fig07_panelA..D.png)")


if __name__ == "__main__":
    main()
