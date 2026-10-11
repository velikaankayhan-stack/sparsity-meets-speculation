# -*- coding: utf-8 -*-
"""
16_hot_neuron_control.py - Where does the "deviation from independence" in the union come from?

Also runs on CPU (Qwen2.5-0.5B, 32-bit, a few minutes).

Question: in 01_union.py the union over k adjacent tokens came out smaller
than the independence formula 1 - (1 - p)^k predicts. There are two possible
causes:
  (a) HOT NEURONS: some neurons are active for almost every token. This
      shrinks the union even if tokens are independent of each other.
  (b) ADJACENCY: neighbouring tokens use similar neuron sets.
To separate their contributions, four values are compared on the same window:
  adjacent    : the measurement from 01 (k neighbouring tokens)
  random      : k tokens drawn at random from the same window (no adjacency,
                hot neurons present)
  frequency   : computed from each neuron's activation frequency f in the
                window: mean[1 - (1 - f)^k]  (independent tokens, hot neurons
                present)
  independent : 1 - (1 - p)^k  (every neuron equally likely to be active)
Share of the deviation due to hot neurons
  = (independent - frequency) / (independent - adjacent).

Method: same settings as 01: Qwen2.5-0.5B, WikiText-2 test, 4 windows x 512
tokens, in every MLP the top p fraction of the down_proj input by absolute
value counts as "active". Random selection uses 400 samples per
(layer, window, p, k) with a fixed seed.

Inputs : WikiText-2 test text via data.wikitext()
Outputs: results/csv/hot_neuron_control_<ts>.csv; prints the mean over all
         layers and windows, including hot_share.
Run    : python 16_hot_neuron_control.py
"""
import datetime
import gc
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID, MODEL_NAME = "Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"
SPARSITY = [0.10, 0.20, 0.30, 0.50]
K_LIST = [1, 2, 3, 4, 6, 8]
SEQ_LEN, N_WINDOWS = 512, 4
SAMPLES = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")


def make_mask(idx, n_neurons, keep):
    """idx: (seq, max_keep) sorted indices -> (seq, n_neurons) bool mask."""
    m = torch.zeros(idx.shape[0], n_neurons, dtype=torch.bool)
    m.scatter_(1, idx[:, :keep].long(), True)
    return m


def adjacent(m, k):
    seq, n = m.shape
    return float(np.mean([m[t:t + k].any(0).sum().item() / n for t in range(seq - k + 1)]))


def random_union(m, k, rng):
    seq, n = m.shape
    return float(np.mean([m[rng.choice(seq, k, replace=False)].any(0).sum().item() / n
                          for _ in range(SAMPLES)]))


def frequency(m, k):
    f = m.float().mean(0).numpy()           # activation frequency of each neuron
    return float(np.mean(1 - (1 - f) ** k))


def measure(model, tokens, rng):
    layers = [(a, mod) for a, mod in model.named_modules() if a.endswith("mlp.down_proj")]
    captured = {}

    def hook(name):
        def _f(_, inputs):
            act = inputs[0][0]
            keep = max(1, int(act.shape[-1] * max(SPARSITY)))
            captured[name] = torch.topk(act.abs().float(), k=keep, dim=-1).indices.to(torch.int16).cpu()
        return _f

    handles = [mod.register_forward_pre_hook(hook(a)) for a, mod in layers]
    rows = []
    for w in range(N_WINDOWS):
        chunk = tokens[w * SEQ_LEN:(w + 1) * SEQ_LEN]
        if len(chunk) < SEQ_LEN:
            break
        captured.clear()
        with torch.no_grad():
            model(chunk.unsqueeze(0).to(DEVICE))
        for li, (name, mod) in enumerate(layers):
            n = mod.in_features
            for p in SPARSITY:
                m = make_mask(captured[name], n, max(1, int(n * p)))
                for k in K_LIST:
                    rows.append(dict(model=MODEL_NAME, layer=li, window=w, active_frac=p, k=k,
                                     adjacent=adjacent(m, k), random=random_union(m, k, rng),
                                     frequency=frequency(m, k), independent=1 - (1 - p) ** k))
        print(f"  window {w + 1}/{N_WINDOWS} done")
    for h in handles:
        h.remove()
    return pd.DataFrame(rows)


def main():
    from data import wikitext
    os.makedirs("results/csv", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
    tokens = tok(wikitext(None), return_tensors="pt").input_ids[0]
    print(f"device: {DEVICE}   model: {MODEL_NAME}   {N_WINDOWS} × {SEQ_LEN} tokens")
    df = measure(model, tokens, np.random.default_rng(0))
    del model; gc.collect()

    path = f"results/csv/hot_neuron_control_{TIMESTAMP}.csv"
    df.to_csv(path, index=False)
    g = df.groupby(["active_frac", "k"])[["adjacent", "random", "frequency", "independent"]].mean().reset_index()
    g["hot_share"] = (g.independent - g.frequency) / (g.independent - g.adjacent)
    print("\nMean (over all layers and windows):")
    print(g[g.k > 1].round(3).to_string(index=False))
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
