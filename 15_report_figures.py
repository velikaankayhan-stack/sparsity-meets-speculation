# -*- coding: utf-8 -*-
"""
15_report_figures.py - Report figures that the measurement scripts do not draw themselves.

No GPU needed; only reads files in results/csv.

Colour-blind friendly: Okabe-Ito palette plus a distinct marker/line style/hatch
per series, single-row legend below the plot, large fonts.

Figures (results/figures/)                          Source CSV
  fig02_verify_cost.png          Tree size          timing_* (TIMING_FILES in 10)
  fig03a_cpu_gpu.png             Tree size          tree_CPU_20260919_0257, tree_..._20260919_1941
  fig03b_specexec_trees.png      Tree size          tree_..._20260919_1941
  fig04_acceptance.png           Acceptance rate    acceptance_20261004_2312 (target 1.5B)
  fig05_sparsity_vs_N.png        Surface            union_large_n_20260923_0003
  fig06_draft_cost.png           Small-model cost   computed with 10_speed_model.py
  fig09a_speculative.png         Real speed         speed_model.csv (10) + speed_summary.csv (12)
  fig09b_four_configs.png        Real speed         speed_model.csv (10)

Other report figures: 05_plot_union.py (fig01_panel1/2), 08_decision_surface.py
(fig05_panel1/2), 11_model_comparison.py (fig07_panelA..D).

Run:  python 10_speed_model.py   (first; writes speed_model.csv)
      python 15_report_figures.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

CSV, FIG_DIR = "results/csv", "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
                     "hatch.linewidth": 1.2})
BLUE, ORANGE, GREEN, PINK, RED, SKYBLUE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
                                           "#D55E00", "#56B4E9")
STYLE = [(BLUE, "o", "-"), (ORANGE, "s", "--"), (GREEN, "^", "-."), (RED, "D", ":"),
         (PINK, "v", "-")]
TICKS = FuncFormatter(lambda v, _: f"{v:g}")


def fmt(x, n=2):
    return f"{x:.{n}f}"


def legend_below(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, mode="expand", frameon=False,
               bbox_to_anchor=(0.02, 0.0, 0.96, 0.06), handlelength=2.2, columnspacing=1.0)


def save(fig, name):
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{FIG_DIR}/{name}", dpi=170)
    plt.close(fig)
    print("FIGURE:", f"{FIG_DIR}/{name}")


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_locator(NullLocator())


# reuse the functions of 10_speed_model.py (same computation, same files)
_spec = importlib.util.spec_from_file_location("speed_model", "10_speed_model.py")
K = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(K)
K.CSV_DIR = CSV
SHORT = {("Qwen2.5-1.5B", "fp16"): "1.5B 16-bit", ("Qwen2.5-1.5B", "4bit"): "1.5B 4-bit",
         ("Qwen2.5-3B", "4bit"): "3B 4-bit", ("Qwen2.5-7B", "4bit"): "7B 4-bit",
         ("Qwen2.5-14B", "4bit"): "14B 4-bit (T4)"}

# ------------------------------------------------------------------ fig02
fig, ax = plt.subplots(figsize=(11, 7.4))
for (name, quant, F, pattern, acc_override, short, name12), (col, mk, ls) in zip(K.MODELS, STYLE):
    df = K.read_timing(short, quant).sort_values("N")
    T = K.timing_curve(df); T1 = T(1)
    ax.plot(df.N, [T(n) / T1 for n in df.N], ls=ls, marker=mk, ms=7, lw=2.6, color=col,
            label=SHORT[(short, quant)])
ax.axhline(1.25, color="black", lw=1.4, ls=(0, (2, 2)))
ax.text(1.05, 1.28, "1.25× threshold", fontsize=12)
log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128]); ax.set_yscale("log")
ax.set_yticks([1, 1.25, 1.5, 2, 3, 5, 8]); ax.set_yticklabels(["1", "1.25", "1.5", "2", "3", "5", "8"])
ax.yaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlabel("positions verified in one round (N)")
ax.set_ylabel("round time / single-position round time")
ax.set_title("Cost of a verification round (same session, with KV cache)\n14B on Tesla T4, others on RTX 3050")
ax.grid(alpha=0.3); legend_below(fig, ax, 5); save(fig, "fig02_verify_cost.png")

# ------------------------------------------------------------------ fig03a / 03b (first measurement, no KV cache)
gpu = pd.read_csv(f"{CSV}/tree_GeForce_RTX_3050_6GB_Laptop_GPU_20260919_1941.csv")
cpu = pd.read_csv(f"{CSV}/tree_CPU_20260919_0257.csv")
fig, ax = plt.subplots(figsize=(11, 7.4))
for d, label, (col, mk, ls) in ((gpu[gpu.model == "Qwen2.5-0.5B"], "GPU (RTX 3050 6 GB)", STYLE[0]),
                                (cpu[cpu.model == "Qwen2.5-0.5B"], "CPU", STYLE[3])):
    d = d.sort_values("N"); ratio = d.ms / d.ms.iloc[0]
    ax.plot(d.N, ratio, ls=ls, marker=mk, ms=8, lw=2.6, color=col, label=label)
ax.axhline(1.25, color="black", lw=1.4, ls=(0, (2, 2)))
ax.text(1.1, 1.33, "1.25× threshold", fontsize=12)
log2_axis(ax, [1, 4, 16, 64, 256, 1024, 2048]); ax.set_yscale("log")
ax.yaxis.set_major_formatter(TICKS)
ax.set_xlabel("positions verified in one round (N)")
ax.set_ylabel("round time / single-position round time")
ax.set_title("Same model, two devices (Qwen2.5-0.5B, first measurement without KV cache)\n"
             "flat up to 256 positions on GPU, time grows with N on CPU")
ax.grid(alpha=0.3, which="both"); legend_below(fig, ax, 2); save(fig, "fig03a_cpu_gpu.png")

fig, ax = plt.subplots(figsize=(11, 7.0))
NS = [128, 256, 512, 1024, 2048]; x = np.arange(len(NS)); w = 0.38
for j, (m, label, style) in enumerate((("Qwen2.5-0.5B", "Qwen2.5-0.5B", dict(color=GREEN, edgecolor="black")),
                                       ("Qwen2.5-1.5B", "Qwen2.5-1.5B", dict(color="white", edgecolor=RED,
                                                                             hatch="///", lw=2)))):
    d = gpu[gpu.model == m].set_index("N").ms; ratio = [d[n] / d[1] for n in NS]
    ax.bar(x + (j - 0.5) * w, ratio, w, label=label, **style)
    for xi, o in zip(x + (j - 0.5) * w, ratio):
        ax.text(xi, o * 1.06, f"{fmt(o, 1)}×", ha="center", fontsize=12)
ax.axhline(1.25, color="black", lw=1.4, ls=(0, (2, 2)))
ax.text(2.55, 1.32, "1.25× threshold", fontsize=12)
ax.set_yscale("log"); ax.yaxis.set_major_formatter(TICKS)
ax.set_xticks(x); ax.set_xticklabels([str(n) for n in NS])
ax.set_xlabel("tree size (positions verified in one round, N)")
ax.set_ylabel("round time / single-position round time")
ax.set_title("SpecExec tree sizes on this GPU (RTX 3050 6 GB)\n128 and 256 are free, larger ones are expensive")
ax.grid(axis="y", alpha=0.3, which="both"); legend_below(fig, ax, 2); save(fig, "fig03b_specexec_trees.png")

# ------------------------------------------------------------------ fig04: acceptance rate (target 1.5B)
acc = pd.read_csv(f"{CSV}/acceptance_20261004_2312.csv"); acc = acc[acc.target == "Qwen2.5-1.5B"]
clean = acc[acc.distinct_4gram >= 0.5]; a = clean.a_greedy.mean()
fig, (A, B) = plt.subplots(1, 2, figsize=(15, 7.4))
s = acc.sort_values("a_greedy").reset_index(drop=True)
colors = [BLUE if t >= 0.5 else "white" for t in s.distinct_4gram]
hatches = ["" if t >= 0.5 else "xx" for t in s.distinct_4gram]
for i, r in s.iterrows():
    A.bar(i, r.a_greedy, 0.85, color=colors[i], hatch=hatches[i], edgecolor="#555" if hatches[i] else BLUE,
          label=None)
from matplotlib.patches import Patch
A.add_artist(A.legend(handles=[Patch(facecolor=BLUE, edgecolor=BLUE, label="non-repetitive text"),
                               Patch(facecolor="white", hatch="xx", edgecolor="#555",
                                     label="repetitive text (excluded)")],
                      loc="upper left", fontsize=12, framealpha=0.95))
A.axhline(a, color="black", lw=2, label=f"non-repetitive mean a = {fmt(a)}")
A.axhline(0.6, color=RED, lw=2, ls="--", label="threshold where the main finding holds (a = 0.6)")
A.set_ylim(0.5, 1.0); A.yaxis.set_major_formatter(TICKS)
A.set_xlabel("text sample (sorted by acceptance rate)"); A.set_ylabel("acceptance rate a")
A.set_title("1 — Acceptance rate per text\nQwen2.5-0.5B → Qwen2.5-1.5B, 40 texts × 128 tokens")
A.grid(axis="y", alpha=0.3)
ks = sorted(int(c[8:]) for c in acc.columns if c.startswith("tokens_k"))
E_sim = [clean[f"tokens_k{k}"].sum() / clean[f"rounds_k{k}"].sum() for k in ks]
E_formula = [(1 - a ** (k + 1)) / (1 - a) for k in ks]
B.plot(ks, E_formula, ls="--", lw=2.6, color=ORANGE, label=f"formula (assumes independence), a = {fmt(a)}")
B.plot(ks, E_sim, ls="-", marker="o", ms=7, lw=2.6, color=BLUE, label="simulation (real acceptance sequences)")
log2_axis(B, [1, 2, 4, 8, 16, 32, 64, 128]); B.yaxis.set_major_formatter(TICKS)
B.set_xlabel("chain length k"); B.set_ylabel("tokens gained per round E(k)")
B.set_title("2 — Tokens gained per round\nsimulation stays below the formula")
B.grid(alpha=0.3)
h1, l1 = A.get_legend_handles_labels(); h2, l2 = B.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, frameon=False, fontsize=13)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{FIG_DIR}/fig04_acceptance.png", dpi=170); plt.close(fig)
print("FIGURE:", f"{FIG_DIR}/fig04_acceptance.png")

# ------------------------------------------------------------------ fig05_sparsity_vs_N (surface, large N)
ln = pd.read_csv(f"{CSV}/union_large_n_20260923_0003.csv"); ln = ln[ln.model == "Qwen2.5-0.5B"]
fig, ax = plt.subplots(figsize=(11, 7.4))
for p, (col, mk, ls) in zip((0.1, 0.2, 0.3), STYLE):
    g = ln[ln.active_frac == p].groupby("N")["union"].mean()
    ax.plot(g.index, g.values * 100, ls=ls, marker=mk, ms=8, lw=2.6, color=col,
            label=f"measured, {int(p * 100)}% active neurons")
    ax.plot(g.index, [(1 - (1 - p) ** n) * 100 for n in g.index], ls=(0, (1, 2)), lw=1.6, color=col)
ax.plot([], [], ls=(0, (1, 2)), lw=1.6, color="#555", label="if neurons were independent")
ax.axhline(100, color="black", lw=1.2)
ax.axvspan(128, 256, color="#999", alpha=0.18)
ax.text(132, 12, "SpecExec\ntree size", fontsize=12)
log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128, 256]); ax.set_ylim(0, 105)
ax.set_xlabel("positions verified in one round (N)")
ax.set_ylabel("fraction of neurons to fetch (%)")
ax.set_title("As N grows, the fraction of neurons to fetch approaches 100%\nQwen2.5-0.5B")
ax.grid(alpha=0.3); legend_below(fig, ax, 4); save(fig, "fig05_sparsity_vs_N.png")

# ------------------------------------------------------------------ fig06: draft cost
RATIO = np.round(np.arange(0, 1.01, 0.05), 2)
fig, ax = plt.subplots(figsize=(11, 7.4))
for (name, quant, F, pattern, acc_override, short, name12), (col, mk, ls) in zip(K.MODELS, STYLE):
    timing = K.read_timing(short, quant)
    union, qual = K.read_measure(pattern, "union"), K.read_measure(pattern, "quality")
    if acc_override:
        acc = pd.read_csv(os.path.join(CSV, acc_override[0])); acc = acc[acc.target.str.contains(acc_override[1])]
    else:
        acc = K.read_measure(pattern, "acceptance")
    T1 = K.compute(union, qual, timing, acc, F, 0.0)["T1_ms"]
    gain = [K.compute(union, qual, timing, acc, F, o * T1)["gain_5"] for o in RATIO]
    cc = float(timing.draft_ms.iloc[0]) / T1
    ax.plot(RATIO, gain, ls=ls, marker=mk, ms=6, lw=2.4, color=col, label=SHORT[(short, quant)])
    ax.plot(cc, np.interp(cc, RATIO, gain), marker="*", ms=24, color=col, mec="black", mew=1.4,
            ls="none", zorder=5)
ax.axhline(1, color="black", lw=1.2)
ax.xaxis.set_major_formatter(TICKS); ax.yaxis.set_major_formatter(TICKS)
ax.set_xlabel("draft cost c (draft time per token / target round time)")
ax.set_ylabel("sparsity gain on top of speculative (×)")
ax.set_title("Sparsity gain on top of speculative decoding, 5% quality budget (model, upper bound)\n"
             "★ = c measured in this work   (14B on Tesla T4, others on RTX 3050)")
ax.grid(alpha=0.3); legend_below(fig, ax, 5); save(fig, "fig06_draft_cost.png")

# ------------------------------------------------------------------ fig09a / 09b
sm = pd.read_csv(f"{CSV}/speed_model.csv")
label = {("1.5B", "fp16"): "1.5B 16-bit", ("1.5B", "4bit"): "1.5B 4-bit", ("3B", "4bit"): "3B 4-bit",
         ("7B", "4bit"): "7B 4-bit", ("14B", "4bit"): "14B 4-bit\n(Tesla T4)"}
fig, ax = plt.subplots(figsize=(11, 6.6))
x = np.arange(len(sm)); w = 0.36
ax.bar(x - w / 2, sm.spec_only, w, color="white", edgecolor=BLUE, hatch="///", lw=2, label="computed")
ax.bar(x + w / 2, sm.r_spec, w, color=ORANGE, edgecolor="black", lw=1.2, label="measured")
for xi, h_, g_ in zip(x, sm.spec_only, sm.r_spec):
    ax.text(xi - w / 2, h_ + 0.015, fmt(h_), ha="center", va="bottom", fontsize=12)
    ax.text(xi + w / 2, g_ + 0.015, fmt(g_), ha="center", va="bottom", fontsize=12, weight="bold")
ax.axhline(1, color="black", lw=1.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{label[(r.model, r.quant)]}\nc = {fmt(r.c)}, a = {fmt(r.a_nonrep)}"
                    for _, r in sm.iterrows()], fontsize=12)
ax.set_ylim(0.6, 1.4); ax.yaxis.set_major_formatter(TICKS); ax.set_ylabel("speed (baseline decoding = 1)")
ax.set_title("Speculative decoding: computed vs. measured speed\n"
             "(speeds up models with c < a, slows down those with c > a)")
ax.grid(axis="y", alpha=0.3); legend_below(fig, ax, 2); save(fig, "fig09a_speculative.png")

r = sm[(sm.model == "1.5B") & (sm.quant == "fp16")].iloc[0]
configs = ["baseline\ndecoding", "speculative\nonly", "sparsity\nonly", "both\ncombined"]
series = [("computed (upper bound for sparsity)", [1.0, r.spec_only, r.sparse_only, r.combined],
           dict(color="white", edgecolor=BLUE, hatch="///", lw=2), False),
          ("from measured round times", [1.0, r.r_spec_timing, r.r_sparse, r.r_combined],
           dict(color=ORANGE, edgecolor="black", lw=1.2), True),
          ("measured end to end", [1.0, r.r_spec, np.nan, np.nan],
           dict(color=GREEN, edgecolor="black", hatch="..", lw=1.2), True)]
fig, ax = plt.subplots(figsize=(11, 6.6))
x = np.arange(4); w = 0.26
for j, (lbl, vals, style, bold) in enumerate(series):
    xs = x + (j - 1) * w
    ax.bar(xs, vals, w, label=lbl, **style)
    for xi, d_ in zip(xs, vals):
        if np.isnan(d_):
            ax.text(xi, 0.04, "not measured", rotation=90, ha="center", va="bottom", fontsize=11, color="#444")
        else:
            ax.text(xi, d_ + 0.02, fmt(d_), ha="center", va="bottom", fontsize=11.5,
                    weight="bold" if bold else None)
ax.axhline(1, color="black", lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(configs)
ax.set_ylim(0, 1.85); ax.yaxis.set_major_formatter(TICKS); ax.set_ylabel("speed (baseline decoding = 1)")
ax.set_title("Qwen2.5-1.5B (16-bit, RTX 3050): four configurations\n"
             "sparsity: assumes a perfect predictor, 50% active neurons")
ax.grid(axis="y", alpha=0.3); legend_below(fig, ax, 3); save(fig, "fig09b_four_configs.png")
