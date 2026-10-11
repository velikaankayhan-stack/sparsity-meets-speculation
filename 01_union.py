"""
01_union.py  -  Union dilution experiment
================================================================
micro-scale project

QUESTION
--------
Activation sparsity works PER TOKEN: for each token only a small fraction
of neurons counts as "active".

Speculative decoding, however, verifies SEVERAL TOKENS AT ONCE. So the
UNION of the neurons needed by those k tokens has to be fetched.

If consecutive tokens keep activating the same neurons, the union stays
small and the two methods combine nicely. If each token activates different
neurons, the union grows and the sparsity melts away.

This script measures that curve.

REFERENCE POINT (analytic)
--------------------------
If neurons were picked at random for every token, the union of k tokens at
sparsity level p would be exactly:

    expected_union = 1 - (1 - p)^k

p=0.20, k=4  ->  0.590

If the measured value is close to this number the sets are independent (bad).
If it is close to p the sets are almost identical (ideal).

INPUTS
------
WikiText-2 test split via data.py (or a local .txt file, see TEXT_FILE).
Models in MODELS are downloaded from the Hugging Face hub. A GPU is
recommended; CPU works but is very slow.

OUTPUTS
-------
results/csv/union_<ts>.csv
results/figures/union_<model>_<ts>.png

Result files are never overwritten; the file names contain a timestamp.

RUN
---
python 01_union.py   (from the repo root)
"""

import os
import gc
import json
import time
import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# SETTINGS
# =====================================================================

MODELS = [
    ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
    ("HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
    ("gpt2", "GPT-2"),
]

# Sparsity levels: the fraction of neurons we count as "active"
SPARSITY = [0.10, 0.20, 0.30, 0.50]

# Number of consecutive tokens whose union is examined
K_LIST = [1, 2, 3, 4, 6, 8]

SEQ_LEN     = 512      # tokens processed in one forward pass
N_WINDOWS   = 4        # number of text chunks to process
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# For a Turkish measurement give a .txt path here. If None, WikiText-2 is used.
TEXT_FILE = None
# TEXT_FILE = "data/turkish.txt"

CSV_DIR     = "results/csv"
FIGURE_DIR  = "results/figures"

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =====================================================================
# TEXT PREPARATION
# =====================================================================



def load_text():
    """Return the evaluation text. There is NO silent fallback."""
    from data import wikitext
    return wikitext(TEXT_FILE)


# =====================================================================
# FINDING THE MLP LAYERS
# =====================================================================

def mlp_output_layers(model):
    """
    Return the final MLP projection (down_proj / mlp.c_proj).
    The INPUT of this layer is exactly the neuron activations.

    Modern models (Qwen, Llama, SmolLM): ...mlp.down_proj
    GPT-2                              : ...mlp.c_proj

    CAUTION: GPT-2 also has attn.c_proj, which must not be picked up.
    """
    found = []
    for name, module in model.named_modules():
        if ".mlp." not in name and not name.endswith("mlp"):
            continue
        if name.endswith("down_proj") or name.endswith("mlp.c_proj"):
            found.append((name, module))

    if not found:
        raise RuntimeError("No MLP output layer found - architecture not recognised")
    return found


# =====================================================================
# ACTIVATION CAPTURE
# =====================================================================

class Catcher:
    """
    Captures the input of down_proj and, for every token position, stores
    the indices of the neurons with the largest |value|.

    We do not store the activations themselves - only the ranking.
    That keeps memory usage low.
    """

    def __init__(self, max_frac):
        self.max_frac = max_frac
        self.indices = {}     # layer name -> (seq, keep) int16 tensor
        self.n_neurons = {}

    def hook(self, name):
        def _f(module, inputs):
            act = inputs[0]                     # (batch, seq, n_neurons)
            if act.dim() == 3:
                act = act[0]                    # assumes batch=1
            n_neurons = act.shape[-1]
            keep = max(1, int(n_neurons * self.max_frac))

            # Indices of the neurons with the largest absolute value, descending
            _, idx = torch.topk(act.abs().float(), k=keep, dim=-1)

            self.indices[name]   = idx.to(torch.int16).cpu()
            self.n_neurons[name] = n_neurons
        return _f


# =====================================================================
# UNION COMPUTATION
# =====================================================================

def union_fraction(idx, n_neurons, keep, k):
    """
    idx     : (seq, max_keep) int16, ranked neuron indices for each position
    keep    : how many neurons count as active at this sparsity level
    k       : number of consecutive tokens whose union is taken

    Returns: mean union fraction over all sliding windows
    """
    seq = idx.shape[0]
    if seq < k:
        return float("nan")

    top = idx[:, :keep].long()                        # (seq, keep)

    # Boolean mask for each position
    mask = torch.zeros(seq, n_neurons, dtype=torch.bool)
    mask.scatter_(1, top, True)

    # Union over sliding windows
    fractions = []
    for t in range(seq - k + 1):
        union = mask[t:t + k].any(dim=0).sum().item()
        fractions.append(union / n_neurons)

    return float(np.mean(fractions))


def adjacent_overlap(idx, keep):
    """
    Jaccard similarity between the sets of two consecutive tokens.
    1.0 = identical, 0.0 = no neuron in common.
    """
    seq = idx.shape[0]
    top = idx[:, :keep].long()
    scores = []
    for t in range(seq - 1):
        a = set(top[t].tolist())
        b = set(top[t + 1].tolist())
        scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores))


# =====================================================================
# MAIN LOOP
# =====================================================================

def process_model(model_id, model_name, text, rows):
    print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")

    tok   = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()

    layers = mlp_output_layers(model)
    print(f"  found {len(layers)} MLP layers")

    catcher = Catcher(max_frac=max(SPARSITY))
    handles = [m.register_forward_pre_hook(catcher.hook(a))
               for a, m in layers]

    tokens = tok(text, return_tensors="pt").input_ids[0]
    print(f"  {len(tokens)} tokens")

    for window in range(N_WINDOWS):
        start = window * SEQ_LEN
        end = start + SEQ_LEN
        if end > len(tokens):
            break

        chunk = tokens[start:end].unsqueeze(0).to(DEVICE)

        catcher.indices.clear()
        with torch.no_grad():
            model(chunk)

        for layer_i, (layer_name, _) in enumerate(layers):
            idx       = catcher.indices[layer_name]
            n_neurons = catcher.n_neurons[layer_name]

            for p in SPARSITY:
                keep = max(1, int(n_neurons * p))

                overlap = adjacent_overlap(idx, keep)

                for k in K_LIST:
                    measured = union_fraction(idx, n_neurons, keep, k)
                    random   = 1.0 - (1.0 - p) ** k

                    rows.append(dict(
                        model         = model_name,
                        layer         = layer_i,
                        layer_name    = layer_name,
                        n_neurons     = n_neurons,
                        window        = window,
                        active_frac   = p,
                        k             = k,
                        union         = measured,
                        random        = random,
                        # 0 = identical sets, 1 = independent sets
                        independence  = (measured - p) / max(random - p, 1e-9),
                        adjacent_jaccard = overlap,
                    ))

        print(f"  window {window + 1}/{N_WINDOWS} done")

    for h in handles:
        h.remove()
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# =====================================================================
# PLOT
# =====================================================================

def plot(df, model_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: union curve ---
    ax = axes[0]
    for p in SPARSITY:
        sub = df[(df.model == model_name) & (df.active_frac == p)]
        if sub.empty:
            continue
        mean = sub.groupby("k")["union"].mean()
        ax.plot(mean.index, mean.values, "o-", lw=2, label=f"measured p={p}")

        rand = [1 - (1 - p) ** k for k in mean.index]
        ax.plot(mean.index, rand, "--", lw=1, alpha=0.5, color="gray")

    ax.set_xlabel("tokens verified at once (k)")
    ax.set_ylabel("fraction of neurons in the union")
    ax.set_title(f"{model_name} - union dilution\n"
                 "dashed grey = if selection were random")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --- Right: independence per layer ---
    ax = axes[1]
    for p in SPARSITY:
        sub = df[(df.model == model_name) & (df.active_frac == p) & (df.k == 4)]
        if sub.empty:
            continue
        mean = sub.groupby("layer")["independence"].mean()
        ax.plot(mean.index, mean.values, "o-", lw=1.5, ms=3, label=f"p={p}")

    ax.axhline(0, color="green", ls=":", lw=1)
    ax.axhline(1, color="red", ls=":", lw=1)
    ax.set_xlabel("layer")
    ax.set_ylabel("independence  (0 = same neurons, 1 = random)")
    ax.set_title(f"{model_name} - per layer (k=4)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# =====================================================================
# RUN
# =====================================================================

def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    print(f"device: {DEVICE}")
    if DEVICE == "cpu":
        print("  ! WARNING: no CUDA, running on CPU. That is fine for this experiment")
        print("    but very slow. Consider installing the CUDA build of torch.")
    if DEVICE == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    text = load_text()
    rows = []

    for model_id, model_name in MODELS:
        try:
            process_model(model_id, model_name, text, rows)
        except Exception as e:
            print(f"  ! {model_name} skipped: {type(e).__name__}: {e}")
            continue

        # Save after every model so nothing is lost if a later one crashes
        df = pd.DataFrame(rows)
        csv_path = f"{CSV_DIR}/union_{TIMESTAMP}.csv"
        df.to_csv(csv_path, index=False)

        try:
            plot(df, model_name,
                 f"{FIGURE_DIR}/union_{model_name}_{TIMESTAMP}.png")
        except Exception as e:
            print(f"  ! could not draw plot: {e}")

    if not rows:
        print("\nNo model could be processed.")
        return

    df = pd.DataFrame(rows)
    print(f"\nCSV: {CSV_DIR}/union_{TIMESTAMP}.csv")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY  -  p = 0.20, k = 4")
    print("=" * 70)
    print(f"{'model':<16} {'measured':>9} {'random':>9} "
          f"{'independence':>12} {'jaccard':>9}")
    print("-" * 70)

    for model_name in df.model.unique():
        sub = df[(df.model == model_name) & (df.active_frac == 0.20) & (df.k == 4)]
        if sub.empty:
            continue
        print(f"{model_name:<16} "
              f"{sub.union.mean():>9.3f} "
              f"{sub.random.mean():>9.3f} "
              f"{sub.independence.mean():>12.3f} "
              f"{sub.adjacent_jaccard.mean():>9.3f}")

    print("\nINTERPRETATION:")
    print("  independence close to 0 -> consecutive tokens use the same")
    print("     neurons. Sparsity and speculative decoding COMBINE.")
    print("  independence close to 1 -> every token needs different neurons.")
    print("     Sparsity MELTS AWAY under speculative decoding.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal time: {time.time() - t0:.0f} s")
