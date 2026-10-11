# -*- coding: utf-8 -*-
"""
08_decision_surface.py — First decision surface (Qwen2.5-0.5B, RTX 3050; "Surface" section of the report)

No GPU needed; only reads CSVs.

Combines four measurements:
  01_union.py      -> union model           U(k, p) = 1 - (1-p)^(b·k^α)
  02_tree_size.py  -> round time model      T(N)    = M + c2·N²   (first, uncached measurement)
  04_quality.py    -> quality wall          lowest allowed active fraction
  07_acceptance.py -> tokens per round      E(k)    (simulated from measured acceptance sequences)
then applies quality as a constraint and finds the fastest point.

SPEED MODEL
    T(k, p) = k·t_draft + M · (F_MLP·U(k+1,p) + (1 - F_MLP)) + c2·(k+1)²
  One verification round processes N = k + 1 positions (k draft tokens + the last accepted token).
    speed   = E(k) / T(k, p)
  The report's Surface section treats the draft as free (DRAFT_RATIO = 0). The effect of
  draft cost is scanned separately in the "Sensitivity — draft cost" table printed below;
  the full draft cost calculation is in 10_speed_model.py.

  M      : time of a single-position round (shortest measured time)
  F_MLP  : share of the weights read per round that belong to the MLP (sparsity only shrinks this)
  E(k)   : from non-repetitive prompts; without an acceptance CSV, E = (1 - a^(k+1)) / (1 - a)

NOTE — all of M is counted as memory reads (an upper bound in favour of sparsity).
The "call overhead + memory" and "memory bound" scenarios are computed at the end.

Inputs : results/csv/union_*.csv (via paths.union_csv), TREE_FILE (via paths.tree_csv),
         newest results/csv/quality_*.csv, results/csv/acceptance_*.csv
Outputs: results/figures/fig05_decision.png, fig05_panel1.png, fig05_panel2.png
Run    : python 08_decision_surface.py   (from the repo root)
"""
import glob, os
import re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import union_csv, tree_csv

# =====================================================================
# SETTINGS
# =====================================================================

MODEL = "Qwen2.5-0.5B"      # set to "Qwen2.5-1.5B" once 1.5B is measured in 01 and 04
TREE_FILE = "tree_GeForce_RTX_3050_6GB_Laptop_GPU_20260919_1941.csv"

# MLP share of the weights read in one round (embedding/output layer included):
#   0.5B: MLP 313.8M / total read 494M  = 0.64
#   1.5B: MLP 1156M  / total read 1543M = 0.75
F_MLP_TABLE = {"Qwen2.5-0.5B": 0.64, "Qwen2.5-1.5B": 0.75}
WEIGHT_GB   = {"Qwen2.5-0.5B": 0.99, "Qwen2.5-1.5B": 3.09}   # fp16
BANDWIDTH_GBS = 192          # RTX 3050 6GB Laptop, theoretical memory bandwidth

DRAFT_RATIO = 0.0             # draft time / M  (0 = old "free draft" assumption)

ACCEPTANCE_TARGET = "Qwen2.5-1.5B"  # acceptance rate was measured for this target model
A_ASSUMED  = 0.70             # used if there is no acceptance CSV
KS = np.arange(1, 129)


# =====================================================================
# READ DATA
# =====================================================================

F_MLP = F_MLP_TABLE[MODEL]

g = pd.read_csv(union_csv())
g = g[g.model == MODEL]
if g.empty:
    raise SystemExit(f"{MODEL} not found in the union CSV. Add it to the MODELS list "
                     f"in 01_union.py and run it again.")
g = g.groupby(["active_frac", "k"]).union.mean().reset_index()

t = pd.read_csv(tree_csv(TREE_FILE))
t = t[t.model == MODEL].sort_values("N")
if t.empty:
    raise SystemExit(f"{MODEL} not found in the tree CSV.")

quality_path = sorted(glob.glob("results/csv/quality_*.csv"), key=lambda y: (re.findall(r"\d{8}_\d{4}", os.path.basename(y)) or [""])[-1])[-1]
qual = pd.read_csv(quality_path)
print(f"quality file     : {quality_path}")
qm = qual[qual.model == MODEL].sort_values("active_frac")
if qm.empty:
    raise SystemExit(f"{MODEL} not found in the quality CSV. Add it to 04_quality.py.")

# ---- round time model ----
M = float(t.ms.min())
_r = np.clip(t.ms.values - M, 0, None)
c2 = float((_r * t.N.values**2).sum() / (t.N.values**4).sum())
max_dev = np.abs(M + c2 * t.N.values**2 - t.ms.values).max()
print(f"time model       : T(N) = {M:.2f} + {c2:.3e}·N² ms  (max deviation {max_dev:.0f} ms)")

# ---- union law ----
PL = np.array(sorted(g.active_frac.unique()))
par = []
for p in PL:
    a = g[(g.active_frac == p) & (g.k > 1)]
    keff = np.log(1 - a.union) / np.log(1 - p)
    al, lc = np.polyfit(np.log(a.k.values), np.log(keff.values), 1)
    par.append((al, np.exp(lc)))
par = np.array(par)
print("union law        : " + ", ".join(
    f"p={p:.1f}: α={al:.3f} c={c:.3f}" for p, (al, c) in zip(PL, par)))


def U(k, p):
    al = np.interp(p, PL, par[:, 0]); c = np.interp(p, PL, par[:, 1])
    return np.where(p >= 1.0, 1.0, 1 - (1 - p) ** (c * k ** al))


# ---- E(k): measured ----
def E_formula(k, a):
    return (1 - a ** (k + 1)) / (1 - a)

acceptance_paths = sorted(glob.glob("results/csv/acceptance_*.csv"), key=lambda y: (re.findall(r"\d{8}_\d{4}", os.path.basename(y)) or [""])[-1])
E_SOURCE, A_MEASURED = "formula (assumed)", A_ASSUMED
E_table = None
# newest acceptance file that contains ACCEPTANCE_TARGET (the newest one may be for another model)
acc, acceptance_path = pd.DataFrame(), None
for _y in reversed(acceptance_paths):
    _d = pd.read_csv(_y)
    if (_d.target == ACCEPTANCE_TARGET).any():
        acc, acceptance_path = _d[_d.target == ACCEPTANCE_TARGET], _y
        break
if True:
    if len(acc):
        clean = acc[acc.distinct_4gram >= 0.5]
        A_MEASURED = float(clean.a_greedy.mean())
        k_values = sorted(int(c[8:]) for c in acc.columns if c.startswith("tokens_k"))
        E_table = (np.array(k_values),
                   np.array([clean[f"tokens_k{k}"].sum() / clean[f"rounds_k{k}"].sum()
                             for k in k_values]))
        E_SOURCE = f"measured ({len(clean)} non-repetitive prompts, a={A_MEASURED:.2f})"
        print(f"acceptance file  : {acceptance_path}")
print(f"E(k) source      : {E_SOURCE}")
if E_table is not None and ACCEPTANCE_TARGET != MODEL:
    print(f"  NOTE: acceptance rate was measured for target {ACCEPTANCE_TARGET}, "
          f"other measurements are for {MODEL}.")


def E(k):
    if E_table is None:
        return E_formula(k, A_ASSUMED)
    kx, ey = E_table
    return np.interp(np.log(k), np.log(kx), ey)


def speed(k, p, M_=None, O=0.0, a=None, ratio=None):
    """tokens/s. O: fixed overhead not affected by sparsity (ms)."""
    M_ = M if M_ is None else M_
    ratio = DRAFT_RATIO if ratio is None else ratio
    N = k + 1                          # k draft tokens + last accepted token = k+1 positions
    T = k * ratio * (O + M_) + O + M_ * (F_MLP * U(N, p) + (1 - F_MLP)) + c2 * N**2
    e = E(k) if a is None else E_formula(k, a)
    return e / (T / 1000)


# ---- quality wall ----
def wall(limit):
    ok = qm[qm.degradation_pct <= limit]
    return float(ok.active_frac.min()) if len(ok) else 1.0
D1, D5 = wall(1.0), wall(5.0)
print(f"quality wall     : 1% budget -> active ≥ {D1*100:.0f}%,  "
      f"5% budget -> active ≥ {D5*100:.0f}%")

# =====================================================================
# BEST POINTS
# =====================================================================

ps = np.round(np.linspace(PL.min(), 1.0, 91), 3)
K, P = np.meshgrid(KS, ps)
H = speed(K.astype(float), P)


def best_point(pmin, pmax=1.0, **kw):
    """On a tie (difference < 0.1%) the point with LESS sparsity is chosen:
    if there is no gain, sparsity is pointless and only costs quality."""
    best = (0, None, None)
    for p in ps[(ps >= pmin - 1e-9) & (ps <= pmax + 1e-9)][::-1]:
        h = speed(KS.astype(float), p, **kw)
        i = int(h.argmax())
        if h[i] > best[0] * 1.001:
            best = (float(h[i]), int(KS[i]), float(p))
    return best


print("\n" + "=" * 66 + f"\nFASTEST POINT — {MODEL}, RTX 3050, draft = M × {DRAFT_RATIO}\n" + "=" * 66)
none = best_point(1.0)
scenarios = [("no sparsity", none),
             ("1% quality budget", best_point(D1)),
             ("5% quality budget", best_point(D5)),
             ("unconstrained (quality ignored)", best_point(ps.min()))]
for name, (h, k, p) in scenarios:
    print(f"  {name:<33} k={k:>3}  active {p*100:>3.0f}%  {h:6.1f} tokens/s"
          f"  ({h / none[0]:.2f}×)")

print(f"\nGain from sparsity (active {D5*100:.0f}%, fixed k):")
print(f"  {'k':>4} {'no sparsity':>14} {'active '+str(int(D5*100))+'%':>10} {'gain':>7}")
for k in (1, 2, 4, 8, 16, 32, none[1]):
    h1, h2 = speed(float(k), 1.0), speed(float(k), D5)
    print(f"  {k:>4} {h1:>14.1f} {h2:>10.1f} {h2/h1:>6.2f}×")

print(f"\nSensitivity — draft cost (draft time / M), 5% budget:")
for o in (0.0, 0.1, 0.2, 0.25, 0.3, 0.5):
    y, s_ = best_point(1.0, ratio=o), best_point(D5, ratio=o)
    print(f"  {o:.2f}  fastest k={y[1]:>3}  {y[0]:6.1f} tokens/s  "
          f"gain from sparsity {s_[0]/y[0]:.3f}×")

print(f"\nSensitivity — acceptance rate (formula, 5% budget):")
for a in (0.5, 0.6, 0.7, 0.78, 0.8, 0.84, 0.9):
    y, s = best_point(1.0, a=a), best_point(D5, a=a)
    print(f"  a={a:.2f}  fastest k={y[1]:>3}  {y[0]:6.1f} tokens/s  "
          f"gain from sparsity {s[0]/y[0]:.3f}×")

M_memory = WEIGHT_GB[MODEL] / BANDWIDTH_GBS * 1000
print(f"\nSensitivity — call overhead (how much of M is really memory reads?):")
print(f"  measured M = {M:.1f} ms, theoretical memory read ≈ {M_memory:.1f} ms")
for name, kw in [("all of M is memory (main model)", {}),
                 ("measured: call overhead + memory", dict(M_=M_memory, O=M - M_memory)),
                 ("memory bound (call overhead 0)", dict(M_=M_memory, O=0.0))]:
    y, s = best_point(1.0, **kw), best_point(D5, **kw)
    k1 = speed(1.0, D5, **kw) / speed(1.0, 1.0, **kw)
    print(f"  {name:<34} fastest k={y[1]:>3}  gain {s[0]/y[0]:.3f}×   "
          f"(sparsity only, k=1: {k1:.2f}×)")

# =====================================================================
# FIGURE
# =====================================================================

fig, (A, B) = plt.subplots(1, 2, figsize=(15, 5.8),
                           gridspec_kw=dict(width_ratios=[1, 1.25]))

# ---- left: quality wall ----
colour = {"Qwen2.5-0.5B": "#1f77b4", "Qwen2.5-1.5B": "#2ca02c",
          "SmolLM2-360M": "#d62728", "GPT-2": "#7f7f7f"}
for m in qual.model.unique():
    a = qual[qual.model == m].sort_values("active_frac")
    A.plot(a.active_frac * 100, a.degradation_pct, "o-", lw=2, ms=5,
           color=colour.get(m), label=m)
A.axvspan(PL.min() * 100 - 5, D5 * 100, color="red", alpha=0.10)
A.text((PL.min() * 100 + D5 * 100) / 2, 0.35, "unusable region\n(exceeds 5% budget)",
       ha="center", color="#b91c1c", fontsize=9, weight="bold")
for s, r in [(1, "#16a34a"), (5, "#ea580c")]:
    A.axhline(s, ls="--", lw=1.1, color=r); A.text(101, s, f"{s}%", color=r, va="center")
A.set_yscale("symlog", linthresh=1); A.set_ylim(-0.6, 6000)
A.set_xlabel("active neuron fraction (%)"); A.set_ylabel("perplexity degradation (%)")
A.set_title("1 — QUALITY WALL (measured)\nWikiText-2, training-free thresholding")
A.grid(alpha=0.3); A.legend(fontsize=8)

# ---- right: constrained optimisation ----
im = B.pcolormesh(K, P * 100, H, cmap="magma", shading="gouraud")
plt.colorbar(im, ax=B, label="tokens generated per second")
B.contour(K, P * 100, H, levels=8, colors="w", linewidths=0.5, alpha=0.5)
B.axhspan(ps.min() * 100, D5 * 100, color="black", alpha=0.55)
B.text(3, (ps.min() * 100 + D5 * 100) / 2, "EXCLUDED: quality loss > 5%",
       color="w", fontsize=10, weight="bold", va="center")
h, k, p = best_point(D5)
B.plot(k, p * 100, "*", ms=20, color="#39FF14", mec="k", zorder=6, clip_on=False)
B.annotate(f"fastest (within quality)\nk={k}, active {p*100:.0f}%\n{h:.1f} tokens/s\ngain from sparsity {h/none[0]:.2f}×",
           xy=(k, p * 100), xytext=(k * 0.1, 68), color="w", fontsize=9.5,
           weight="bold", arrowprops=dict(arrowstyle="->", color="w"))
B.set_xscale("log", base=2)
B.set_xlabel("tokens verified per round (k)")
B.set_ylabel("active neuron fraction (%)")
B.set_title(f"2 — CONSTRAINED OPTIMISATION\n{MODEL} · RTX 3050 · draft = M×{DRAFT_RATIO} · E(k): {E_SOURCE}",
            fontsize=10.5)

plt.tight_layout()
plt.savefig("results/figures/fig05_decision.png", dpi=150, bbox_inches="tight")
# the two panels separately for the report: fig05_panel1.png (quality wall), fig05_panel2.png (surface)
from matplotlib.transforms import Bbox
fig.canvas.draw()
r = fig.canvas.get_renderer()
box1 = A.get_tightbbox(r)
fig.savefig("results/figures/fig05_panel1.png", dpi=150,
            bbox_inches=box1.transformed(fig.dpi_scale_trans.inverted()).expanded(1.03, 1.03))
box = Bbox.union([B.get_tightbbox(r), fig.axes[-1].get_tightbbox(r)])
fig.savefig("results/figures/fig05_panel2.png", dpi=150,
            bbox_inches=box.transformed(fig.dpi_scale_trans.inverted()).expanded(1.02, 1.03))
print("\nFIGURE : results/figures/fig05_decision.png  (+ panels: fig05_panel1.png, fig05_panel2.png)")
