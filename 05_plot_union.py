# -*- coding: utf-8 -*-
"""
05_plot_union.py - Figures for the union finding

Inputs : results/csv/union_*.csv   (output of 01_union.py)
         results/csv/tree_*.csv    (output of 02_tree_size.py)
Outputs: results/figures/fig01_union_full.png (4 panels)
         results/figures/fig01_panel1..4.png  (panels 1 and 2 are used in the report)
         Prints a numeric summary (sparsity gain vs. N per active fraction).
Run    : python 05_plot_union.py   (no GPU needed)

NOTE - the 'active_frac' column in the CSV is the fraction of ACTIVE neurons.
10% = only one tenth of the neurons is computed (most aggressive).
"""
import os

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import union_csv, tree_csv
d = pd.read_csv(union_csv())
g = d.groupby(["model", "active_frac", "k"]).union.mean().reset_index()

MODELS = ["Qwen2.5-0.5B", "SmolLM2-360M", "GPT-2"]
COLORS = {"Qwen2.5-0.5B": "#1f77b4", "SmolLM2-360M": "#d62728", "GPT-2": "#7f7f7f"}
P_LEVELS = [0.1, 0.2, 0.3, 0.5]
CM = dict(zip(P_LEVELS, plt.cm.viridis(np.linspace(.12, .82, 4))))
MLP_SHARE = 0.64        # measured MLP share for Qwen2.5-0.5B
FREE_N = 256            # 02_tree_size.py, RTX 3050 6GB

def fit(m, p):
    """U(k) = 1-(1-p)^(c·k^α)"""
    a = g[(g.model == m) & (g.active_frac == p) & (g.k > 1)]
    keff = np.log(1 - a.union) / np.log(1 - p)
    al, lc = np.polyfit(np.log(a.k.values), np.log(keff.values), 1)
    return al, np.exp(lc)

def predict(k, p, al, c):
    return 1 - (1 - p) ** (c * np.asarray(k, float) ** al)

def gain(u, f=MLP_SHARE):
    return 1.0 / (f * u + (1 - f))

fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
kk = np.array([1, 2, 3, 4, 6, 8])

# --- 1 -------------------------------------------------------------
A = ax[0, 0]
for m in MODELS:
    a = g[(g.model == m) & (g.active_frac == 0.2)].sort_values("k")
    A.plot(a.k, a.union, "o-", lw=2.2, ms=7, color=COLORS[m], label=m)
A.plot(kk, 1 - 0.8 ** kk, "k--", lw=1.6, label="if the sets were independent")
A.axhline(0.2, color="green", ls=":", lw=1.8, label="if the sets were identical (ideal)")
A.set_xlabel("tokens verified in the same round  (k)")
A.set_ylabel("fraction of neurons that must be loaded")
A.set_title("1 - Union dilutes sparsity\n"
            "20% active neurons - the fraction that is 0.20 for a single token grows with k")
A.set_ylim(0, 1); A.grid(alpha=.3); A.legend(fontsize=9, loc="lower right")
_u4 = float(g[(g.model == "Qwen2.5-0.5B") & (g.active_frac == 0.2) & (g.k == 4)].union.iloc[0])
A.annotate(f"{_u4:.3f} at k=4\n{_u4/0.2:.2f}x dilution", xy=(4, _u4), xytext=(4.7, _u4-0.22),
           arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=9.5)
A.text(1.05, .90, "the three models nearly overlap:\nthis is structural, not model-specific",
       fontsize=9, style="italic", color="#444")

# --- 2 -------------------------------------------------------------
B = ax[0, 1]
for p in P_LEVELS:
    a = g[(g.model == "Qwen2.5-0.5B") & (g.active_frac == p)].sort_values("k")
    B.plot(a.k, a.union, "o-", lw=2.2, ms=7, color=CM[p],
           label=f"{int(p*100)}% active neurons")
    B.plot(kk, 1 - (1 - p) ** kk, ls="--", lw=1.1, color=CM[p], alpha=.5)
B.set_xlabel("k"); B.set_ylabel("fraction of neurons that must be loaded")
B.set_title("2 - Not one level but FOUR levels tested (Qwen2.5-0.5B)\n"
            "dashed = independence assumption; measured is always below it")
B.set_ylim(0, 1); B.grid(alpha=.3); B.legend(fontsize=9, loc="lower right")

# --- 3 -------------------------------------------------------------
C = ax[1, 0]
for p in P_LEVELS:
    a = g[(g.model == "Qwen2.5-0.5B") & (g.active_frac == p)].sort_values("k")
    C.plot(a.k, gain(a.union.values), "o-", lw=2.2, ms=7, color=CM[p],
           label=f"{int(p*100)}% active neurons")
C.axhline(1.0, color="red", lw=1.6)
C.text(6.3, 1.025, "no gain", color="red", fontsize=9.5)
C.set_xlabel("k"); C.set_ylabel("gain from reduced weight reads (x)")
C.set_title("3 - The gain from sparsity melts away as k grows\n"
            f"MLP share {int(MLP_SHARE*100)}% (measured for Qwen2.5-0.5B)")
C.grid(alpha=.3); C.legend(fontsize=9)

# --- 4 -------------------------------------------------------------
D = ax[1, 1]
N = np.unique(np.round(np.logspace(0, np.log10(FREE_N), 70)).astype(int))
D.axvspan(1, FREE_N, color="gold", alpha=.15)
for p in P_LEVELS:
    al, c = fit("Qwen2.5-0.5B", p)
    D.plot(N, gain(predict(N, p, al, c)), lw=2.3, color=CM[p],
           label=f"{int(p*100)}% active neurons  (α={al:.2f})")
    a = g[(g.model == "Qwen2.5-0.5B") & (g.active_frac == p)].sort_values("k")
    D.plot(a.k, gain(a.union.values), "o", ms=7.5, color=CM[p],
           mec="k", mew=.7, zorder=5)
D.axhline(1.0, color="red", lw=1.6)
D.axvline(FREE_N, color="k", ls="--", lw=1.6)
D.set_xscale("log", base=2)
D.set_xlabel("positions verified in one round  (N)")
D.set_ylabel("gain from reduced weight reads (x)")
D.set_title("4 - COLLISION\n"
            "filled circle = measured (k ≤ 8) · line = power-law extrapolation")
D.grid(alpha=.3, which="both"); D.legend(fontsize=9)
D.text(1.15, 2.25, f"RTX 3050 free verification region  (N ≤ {FREE_N})",
       fontsize=10, color="#7a5f00", weight="bold")
al, c = fit("Qwen2.5-0.5B", 0.2)
D.annotate(f"N={FREE_N}\n{gain(predict(FREE_N,0.2,al,c)):.2f}×",
           xy=(FREE_N, gain(predict(FREE_N, .2, al, c))), xytext=(96, 1.45),
           fontsize=10, arrowprops=dict(arrowstyle="->", lw=1.1))

fig.suptitle("Finding 01 - Activation sparsity and speculative verification "
             "compete for the same N", fontsize=14, y=0.985)
plt.tight_layout(rect=[0, 0, 1, 0.962])
os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/fig01_union_full.png", dpi=150, bbox_inches="tight")
fig.canvas.draw()
for i, axis in enumerate(ax.flat, 1):
    box = axis.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"results/figures/fig01_panel{i}.png", dpi=150, bbox_inches=box.expanded(1.03, 1.03))
plt.close()

# ---------------- numeric summary ----------------
print(f"{'active':<13}{'α':>7}{'N=1':>8}{'N=4':>8}{'N=8':>8}"
      f"{'N=32':>8}{'N=64':>8}{'N=256':>8}")
for p in P_LEVELS:
    al, c = fit("Qwen2.5-0.5B", p)
    v = [gain(predict(n, p, al, c)) for n in (1, 4, 8, 32, 64, 256)]
    print(f"{str(int(p*100)) + '%':<13}{al:>7.3f}" + "".join(f"{x:>8.2f}" for x in v))
