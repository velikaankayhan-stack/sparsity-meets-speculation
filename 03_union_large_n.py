"""
03_union_large_n.py  -  Sparsity or a large tree?
================================================================
micro-scale project

Direct continuation of 01_union.py. There we measured k up to 8.
Here we go up to 256, because SpecExec uses 128-2048.

QUESTION
--------
When verifying N positions in one round, how many neurons must be fetched?

  small N  -> few neurons, sparsity survives
  large N  -> the union grows, sparsity dies

Where does it die? And which strategy wins after that point?

THE FORK
--------
  Path A : large tree, no sparsity
           many tokens per round, but the whole model is read
  Path B : small N, sparsity on
           few tokens per round, but only part of the model is read

This script compares both on the same axis.

If a CSV from 02_tree_size.py exists, real timings are used.
Otherwise the comparison is based only on the amount of data read.

THIS MEASUREMENT IS MODEL-SPECIFIC - it also runs on Colab.

INPUTS
------
WikiText-2 test split via data.py (or a local .txt file, see TEXT_FILE).
Models in MODELS (Hugging Face hub). Optional: results/csv/tree_<device>_<ts>.csv
from 02_tree_size.py, selected by paths.tree_csv().

OUTPUTS
-------
results/csv/union_large_n_<ts>.csv
results/csv/union_large_n_summary_<ts>.csv
results/figures/union_large_n_<ts>.png

RUN
---
python 03_union_large_n.py   (from the repo root)
"""

import os, gc, glob, time, datetime
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# SETTINGS
# =====================================================================

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
]

SPARSITY  = [0.10, 0.20, 0.30]
N_LIST    = [1, 2, 4, 8, 16, 32, 64, 128, 256]

SEQ_LEN    = 1024          # must be at least 2x the largest N (256)
N_WINDOWS  = 6
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_FILE = None           # give a .txt path for Turkish text

# Model shares - typical values at the 7-8 billion scale
SHARE_MLP, SHARE_ATTN, SHARE_EMBED = 0.70, 0.17, 0.13

# Acceptance rate of the draft model. An assumption until measured on your own setup.
ACCEPTANCE = 0.70

CSV_DIR    = "results/csv"
FIGURE_DIR = "results/figures"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =====================================================================
# TEXT
# =====================================================================

def load_text():
    """There is NO silent fallback. If no data arrives, the script stops."""
    from data import wikitext
    return wikitext(TEXT_FILE)


# =====================================================================
# MLP OUTPUT LAYERS
# =====================================================================

def mlp_layers(model):
    found = []
    for name, mod in model.named_modules():
        if name.endswith("down_proj") or name.endswith("mlp.c_proj"):
            found.append((name, mod))
    if not found:
        raise RuntimeError("No MLP output layer found")
    return found


class Catcher:
    def __init__(self, max_frac):
        self.max_frac = max_frac
        self.idx, self.n_neurons = {}, {}

    def hook(self, name):
        def _f(mod, inputs):
            act = inputs[0]
            if act.dim() == 3:
                act = act[0]
            n = act.shape[-1]
            keep = max(1, int(n * self.max_frac))
            _, i = torch.topk(act.abs().float(), k=keep, dim=-1)
            self.idx[name] = i.to(torch.int32)    # keep it on the GPU
            self.n_neurons[name] = n
        return _f


# =====================================================================
# UNION - vectorised, fast even at N=256
# =====================================================================

def union_curve(idx, n_neurons, keep, n_list):
    """
    Computes the sliding-window union with a cumulative sum.
    No loop over windows, instant even for N=256.
    """
    seq = idx.shape[0]
    top = idx[:, :keep].long()

    mask = torch.zeros(seq, n_neurons, dtype=torch.int16, device=idx.device)
    mask.scatter_(1, top, torch.ones_like(top, dtype=torch.int16))

    # cumulative sum: cs[i] = sum of the first i rows
    cs = torch.zeros(seq + 1, n_neurons, dtype=torch.int32, device=idx.device)
    cs[1:] = mask.cumsum(dim=0)

    result = {}
    for N in n_list:
        if seq < N:
            result[N] = float("nan")
            continue
        # per-window activation count for every neuron
        counts = cs[N:] - cs[:-N]                 # (seq-N+1, n_neurons)
        union = (counts > 0).sum(dim=1).float() / n_neurons
        result[N] = union.mean().item()
        del counts
    del mask, cs
    return result


# =====================================================================
# MAIN MEASUREMENT
# =====================================================================

def process_model(model_id, model_name, text, rows):
    print(f"\n{'='*62}\n{model_name}\n{'='*62}")

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()

    layers = mlp_layers(model)
    print(f"  {len(layers)} MLP layers")

    catcher = Catcher(max(SPARSITY))
    handles = [m.register_forward_pre_hook(catcher.hook(a)) for a, m in layers]

    tokens = tok(text, return_tensors="pt").input_ids[0]

    for window in range(N_WINDOWS):
        start, end = window * SEQ_LEN, (window + 1) * SEQ_LEN
        if end > len(tokens):
            break
        chunk = tokens[start:end].unsqueeze(0).to(DEVICE)

        catcher.idx.clear()
        with torch.no_grad():
            model(chunk)

        for li, (lname, _) in enumerate(layers):
            idx, n = catcher.idx[lname], catcher.n_neurons[lname]
            for p in SPARSITY:
                keep = max(1, int(n * p))
                curve = union_curve(idx, n, keep, N_LIST)
                for N, u in curve.items():
                    rows.append(dict(
                        model=model_name, layer=li, n_neurons=n, window=window,
                        active_frac=p, N=N, union=u,
                        random=1 - (1 - p) ** N,
                    ))
        print(f"  window {window+1}/{N_WINDOWS}")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# =====================================================================
# FORK COMPUTATION
# =====================================================================

def accepted_tokens(a, N):
    """
    Mean number of tokens accepted from N proposals with chain drafting.
    A tree structure does better; this is a LOWER BOUND.
    """
    if N <= 0:
        return 1.0
    return 1 + a * (1 - a ** N) / (1 - a)


def load_timings():
    """Read the output of 02_tree_size.py if present (paths.py does the selection:
    it drops CPU runs and picks the least noisy GPU measurement)."""
    from paths import tree_csv
    try:
        return pd.read_csv(tree_csv())
    except (FileNotFoundError, ValueError):
        return None


def fork_table(df, timings):
    """
    Compare the two paths for every N:
      A: no sparsity   -> 100% of the model is read per round
      B: sparsity on   -> only the union is read per round
    """
    records = []
    for m in df.model.unique():
        for p in SPARSITY:
            for N in N_LIST:
                sub = df[(df.model == m) & (df.active_frac == p) & (df.N == N)]
                if sub.empty:
                    continue
                u = sub.union.mean()
                tokens = accepted_tokens(ACCEPTANCE, N)

                read_A = 1.0
                read_B = u * SHARE_MLP + SHARE_ATTN + SHARE_EMBED

                # time factor: real if 02 output exists, otherwise 1 (data volume only)
                factor = 1.0
                if timings is not None:
                    s = timings[timings.model == m]
                    if not s.empty and N in set(s.N):
                        t1 = s[s.N == 1].ms.iloc[0]
                        tN = s[s.N == N].ms.iloc[0]
                        factor = tN / t1

                records.append(dict(
                    model=m, active_frac=p, N=N, union=u,
                    random=1 - (1 - p) ** N,
                    tokens_per_round=tokens,
                    A_read_per_token=read_A * factor / tokens,
                    B_read_per_token=read_B * factor / tokens,
                    time_factor=factor,
                ))
    return pd.DataFrame(records)


# =====================================================================
# PLOT
# =====================================================================

def plot(df, ft, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    model_name = df.model.unique()[0]

    # --- 1) Union: where sparsity dies
    for p in SPARSITY:
        a = df[(df.model == model_name) & (df.active_frac == p)].groupby("N").union.mean()
        ax[0].plot(a.index, a.values * 100, "o-", lw=2.2, ms=5, label=f"measured p={p}")
        ax[0].plot(a.index, [(1 - (1 - p) ** n) * 100 for n in a.index],
                   "--", lw=1, alpha=.45, color="grey")
    ax[0].axhline(100, color="red", ls=":", lw=1.5)
    ax[0].text(1.3, 92, "sparsity is gone", color="red", fontsize=9)
    ax[0].axvspan(128, 260, color="red", alpha=.06)
    ax[0].text(132, 20, "SpecExec\noperates\nhere", fontsize=8, color="darkred")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("positions verified per round  (N)")
    ax[0].set_ylabel("neurons that must be fetched  (%)")
    ax[0].set_title(f"1 - Where sparsity dies\n{model_name}")
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 108)

    # --- 2) Fork: data read per token
    c = ft[(ft.model == model_name) & (ft.active_frac == 0.20)].sort_values("N")
    ax[1].plot(c.N, c.A_read_per_token, "s-", lw=2.4, ms=6,
               color="#c44", label="A: large tree, NO sparsity")
    ax[1].plot(c.N, c.B_read_per_token, "o-", lw=2.4, ms=6,
               color="#2a7", label="B: sparsity ON (p=0.20)")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("positions verified per round  (N)")
    ax[1].set_ylabel("fraction of model read per token\n(lower = better)")
    ax[1].set_title("2 - The fork\nwhich path reads less")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8.5)

    # --- 3) Advantage of B over A
    for p in SPARSITY:
        c = ft[(ft.model == model_name) & (ft.active_frac == p)].sort_values("N")
        ax[2].plot(c.N, c.A_read_per_token / c.B_read_per_token, "o-",
                   lw=2.2, ms=5, label=f"p={p}")
    ax[2].axhline(1, color="k", ls="--", lw=1.4)
    ax[2].text(1.3, 1.03, "above: sparsity wins", fontsize=8.5)
    ax[2].text(1.3, 0.90, "below: sparsity hurts", fontsize=8.5, color="#c44")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("N"); ax[2].set_ylabel("gain from sparsity (x)")
    ax[2].set_title("3 - When sparsity pays off")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8.5)

    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# =====================================================================

def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print(f"device: {DEVICE}")

    text, rows = load_text(), []
    for mid, mname in MODELS:
        try:
            process_model(mid, mname, text, rows)
        except Exception as e:
            print(f"  ! {mname} skipped: {type(e).__name__}: {e}")
        if rows:
            pd.DataFrame(rows).to_csv(
                f"{CSV_DIR}/union_large_n_{TIMESTAMP}.csv", index=False)

    if not rows:
        print("No measurement could be made."); return

    df = pd.DataFrame(rows)
    ft = fork_table(df, load_timings())
    ft.to_csv(f"{CSV_DIR}/union_large_n_summary_{TIMESTAMP}.csv", index=False)
    plot(df, ft, f"{FIGURE_DIR}/union_large_n_{TIMESTAMP}.png")

    # --- Summary ---
    m0 = df.model.unique()[0]
    print("\n" + "="*72)
    print(f"WHERE SPARSITY DIES   ({m0})")
    print("="*72)
    print(f"{'N':>5}" + "".join(f"{f'p={p}':>10}" for p in SPARSITY))
    print("-"*72)
    for N in N_LIST:
        s = f"{N:>5}"
        for p in SPARSITY:
            a = df[(df.model == m0) & (df.active_frac == p) & (df.N == N)]
            s += f"{a.union.mean()*100:>9.1f}%" if not a.empty else f"{'-':>10}"
        print(s)

    print("\n" + "="*72)
    print(f"FORK   (p=0.20, acceptance rate {ACCEPTANCE})")
    print("="*72)
    print(f"{'N':>5}{'tokens/round':>12}{'path A':>10}{'path B':>10}"
          f"{'A/B advantage':>14}")
    print("-"*72)
    c = ft[(ft.model == m0) & (ft.active_frac == 0.20)].sort_values("N")
    for _, r in c.iterrows():
        print(f"{int(r.N):>5}{r.tokens_per_round:>12.2f}{r.A_read_per_token:>10.3f}"
              f"{r.B_read_per_token:>10.3f}{r.A_read_per_token/r.B_read_per_token:>13.2f}x")

    best_A = c.loc[c.A_read_per_token.idxmin()]
    best_B = c.loc[c.B_read_per_token.idxmin()]
    print(f"\nbest of path A : N={int(best_A.N)}  "
          f"({best_A.A_read_per_token:.3f} read/token)")
    print(f"best of path B : N={int(best_B.N)}  "
          f"({best_B.B_read_per_token:.3f} read/token)")
    winner = "B (sparsity)" if best_B.B_read_per_token < best_A.A_read_per_token else "A (large tree)"
    print(f"WINNER         : {winner}")

    if load_timings() is None:
        print("\nNOTE: no 02_tree_size.py output found, computed from DATA VOLUME "
              "only.\n      For real timings, run 02 on your own GPU first.")

    print(f"\nCSV    : {CSV_DIR}/union_large_n_{TIMESTAMP}.csv")
    print(f"FIGURE : {FIGURE_DIR}/union_large_n_{TIMESTAMP}.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal time: {time.time()-t0:.0f} s")
