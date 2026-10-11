# -*- coding: utf-8 -*-
"""
04_quality.py - Quality cost of activation sparsity (perplexity wall)

Activation sparsity speeds up decoding, but how much quality does it cost?
This script measures how much perplexity degrades when only the top p
fraction of MLP neurons is kept active. The speed formula alone would push
sparsity as far as possible because it has no quality term; this script
provides that term.

Method: training-free, threshold-based sparsity (CATS / TEAL family). In
every layer's MLP, just before down_proj, the top p fraction of intermediate
activations by absolute value is kept and the rest is zeroed. In terms of
quality this is equivalent to never loading those neurons' weights.
Note: this script does NOT measure speed. Masking is done in software, so it
actually runs slower. It only measures quality.

Self-checks:
  1. Stops if the data cannot be loaded (no silent fallback).
  2. Stops if the text is too repetitive (handled in data.py).
  3. The p=1.00 result is compared with the unhooked model. If they differ by
     more than 0.5%, the masking code is broken and the script stops.

Inputs : WikiText-2 test text via data.wikitext()
Outputs: results/csv/quality_<ts>.csv
         results/figures/quality_<ts>.png
Run    : python 04_quality.py   (GPU recommended; CPU works but is slow)
"""

import os, gc, math, datetime
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
    ("Qwen/Qwen2.5-0.5B",          "Qwen2.5-0.5B"),
    ("HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M"),
    ("gpt2",                       "GPT-2"),
]

P_LIST = [1.00, 0.70, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05]

WINDOW       = 1024    # tokens per forward pass
NUM_WINDOWS  = 60      # number of windows (60 x 1024 ~ 61k tokens, enough for PPL)
DTYPE        = torch.float16
TEXT_FILE    = None    # for a Turkish measurement: "data/turkish.txt"

CSV_DIR    = "results/csv"
FIGURE_DIR = "results/figures"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =====================================================================
# DATA - no silent fallback
# =====================================================================

def load_text():
    """There is NO silent fallback. The script stops if data cannot be loaded."""
    from data import wikitext
    return wikitext(TEXT_FILE)


# =====================================================================
# SPARSITY HOOK
# =====================================================================

class Sparsifier:
    """Keeps the top p fraction of the intermediate activations entering down_proj."""

    def __init__(self, ratio):
        self.ratio = ratio
        self.hooks = []

    def _hook(self, module, inputs):
        x = inputs[0]
        if self.ratio >= 1.0:
            return None
        d = x.shape[-1]
        keep = max(1, int(round(d * self.ratio)))
        if keep >= d:
            return None
        # topk is used; kthvalue does not work with CUDA+float16 in some versions
        idx = x.abs().topk(keep, dim=-1, sorted=False).indices
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return (x * mask,) + inputs[1:]

    def attach(self, model):
        targets = []
        for name, m in model.named_modules():
            # Llama/Qwen/SmolLM family: mlp.down_proj
            # GPT-2: mlp.c_proj  (NOT attn.c_proj - the 'mlp.' requirement excludes it)
            if name.endswith("mlp.down_proj") or name.endswith("mlp.c_proj"):
                targets.append((name, m))
        if not targets:
            raise RuntimeError(
                "MLP output layer not found. Model architecture differs from what was expected.")
        for _, m in targets:
            self.hooks.append(m.register_forward_pre_hook(self._hook))
        return len(targets)

    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# =====================================================================
# PERPLEXITY
# =====================================================================

@torch.no_grad()
def perplexity(model, ids, device):
    total_nll, total_tokens = 0.0, 0
    for i in range(NUM_WINDOWS):
        start = i * WINDOW
        end = start + WINDOW
        if end > ids.shape[1]:
            break
        chunk = ids[:, start:end].to(device)
        out = model(chunk, labels=chunk)
        n = chunk.shape[1] - 1
        total_nll += float(out.loss) * n
        total_tokens += n
    if total_tokens == 0:
        raise RuntimeError("No window could be processed - text is too short.")
    return math.exp(total_nll / total_tokens), total_tokens


# =====================================================================

def measure_model(model_id, model_name, text, rows):
    print(f"\n{'='*62}\n{model_name}\n{'='*62}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE if device == "cuda" else torch.float32
    ).to(device).eval()

    ids = tok(text, return_tensors="pt").input_ids
    print(f"  {ids.shape[1]:,} tokens ready, "
          f"{min(NUM_WINDOWS, ids.shape[1]//WINDOW)} windows will be used")

    # --- CHECK: baseline without hooks ---
    clean, used = perplexity(model, ids, device)
    print(f"  baseline perplexity (no hooks) : {clean:.4f}  ({used:,} tokens)")

    base = None
    for p in P_LIST:
        s = Sparsifier(p)
        n_layers = s.attach(model)
        try:
            ppl, _ = perplexity(model, ids, device)
        finally:
            s.detach()

        if p >= 1.0:
            diff = abs(ppl - clean) / clean
            if diff > 0.005:
                raise RuntimeError(
                    f"CHECK FAILED: perplexity at p=1.00 is {ppl:.4f}, "
                    f"without hooks {clean:.4f} ({diff*100:.2f}% difference). "
                    "The masking code is corrupting the model; results are invalid.")
            print(f"  check passed (hooks attached to {n_layers} MLP layers)")
            base = ppl

        degradation = (ppl / base - 1) * 100 if base else 0.0
        rows.append(dict(model=model_name, active_frac=p, ppl=ppl,
                         base_ppl=base, degradation_pct=degradation,
                         layer=n_layers, tokens_evaluated=used))
        print(f"  active {p*100:5.1f}%   PPL {ppl:9.3f}   degradation {degradation:+7.2f}%")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def plot(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    color = dict(zip(df.model.unique(),
                     ["#1f77b4", "#2ca02c", "#d62728", "#7f7f7f"]))
    for m in df.model.unique():
        a = df[df.model == m].sort_values("active_frac")
        ax[0].plot(a.active_frac*100, a.ppl, "o-", lw=2.2, ms=6,
                   color=color[m], label=m)
        ax[1].plot(a.active_frac*100, a.degradation_pct, "o-", lw=2.2, ms=6,
                   color=color[m], label=m)
    for A, (yl, title) in zip(ax, [("perplexity (lower = better)", "1 - Raw perplexity"),
                                   ("degradation vs. baseline (%)", "2 - Quality cost")]):
        A.set_xlabel("active neuron fraction (%)"); A.set_ylabel(yl)
        A.set_title(title); A.grid(alpha=.3); A.legend(fontsize=9)
    ax[0].set_yscale("log")
    for s, r in [(1, "#16a34a"), (5, "#ea580c"), (10, "#b91c1c")]:
        ax[1].axhline(s, ls="--", lw=1.1, color=r)
        ax[1].text(2, s*1.05, f"{s}% degradation", color=r, fontsize=8.5)
    ax[1].set_yscale("symlog", linthresh=1)
    fig.suptitle("Quality cost of sparsity - where the wall is",
                 fontsize=13, y=1.0)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_properties(0).name}")
    else:
        print("device: NONE - running on CPU. Fine for quality measurement, "
              "just slow.")
    print(f"torch : {torch.__version__}  CUDA {torch.version.cuda}\n")

    text = load_text()

    rows = []
    for mid, mname in MODELS:
        try:
            measure_model(mid, mname, text, rows)
        except RuntimeError as e:
            print(f"\n!! {mname} STOPPED: {e}\n")
            raise
        except Exception as e:
            print(f"  ! {mname} skipped: {type(e).__name__}: {e}")
        if rows:
            pd.DataFrame(rows).to_csv(
                f"{CSV_DIR}/quality_{TIMESTAMP}.csv", index=False)

    if not rows:
        print("\nNo measurements could be made.")
        return

    df = pd.DataFrame(rows)
    plot(df, f"{FIGURE_DIR}/quality_{TIMESTAMP}.png")

    print("\n" + "="*62 + "\nSUMMARY - quality wall\n" + "="*62)
    for limit in (1.0, 5.0):
        print(f"\nLowest active fraction if we allow {limit:.0f}% degradation:")
        for m in df.model.unique():
            a = df[(df.model == m) & (df.degradation_pct <= limit)]
            if len(a):
                print(f"  {m:<16} active {a.active_frac.min()*100:.0f}%")
            else:
                print(f"  {m:<16} no level stayed within the limit")

    print(f"\nCSV    : {CSV_DIR}/quality_{TIMESTAMP}.csv")
    print(f"FIGURE : {FIGURE_DIR}/quality_{TIMESTAMP}.png")


if __name__ == "__main__":
    main()
