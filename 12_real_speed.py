# -*- coding: utf-8 -*-
"""
12_real_speed.py - end-to-end speed measurement for four configurations.

Earlier speed numbers were computed from separately measured parts. This
script measures directly whatever can be measured.

Part A - end-to-end generation (every model, real measurement)
  1. Baseline generation : target model alone, greedy.
  2. Speculative only    : target model + Qwen2.5-0.5B draft (Hugging Face
                           "assisted generation"), k = 1, 2, 3, 4, 6, 8.
  Also measured: the draft's per-token time (for c) and a lossless check
  (does the speculative output match the baseline output?).

Part B - real round time with a sparse MLP (fp16 model only)
  In every MLP layer only a fraction f of the neurons' weights is read
  (gate/up rows + down columns). The time of an N-position round is measured
  for f = 30% ... 100% and N = 1 ... 9.
  The active neurons are given by a perfect predictor (zero predictor cost),
  so this is the best time a real sparse kernel could reach. Unlike the
  earlier estimate there is no ideal-read assumption: fixed software
  overhead, attention layers etc. are included in the measurement.
  Two variants are measured:
    "ideal"  : selected rows are contiguous in memory (a good kernel)
    "gather" : selected rows are gathered from random positions (index_select)

  In 4-bit models (1.5B 4-bit, 3B, 7B, 14B) weights are packed, so rows cannot
  be selected; Part B is skipped and the sparse configurations keep the
  estimate from 10_speed_model.py.

Result - four configurations relative to baseline generation:
  baseline / speculative only / sparsity only / both combined
  (for the fp16 model the sparse times are real, from Part B; E(k) and the
  union come from the measured CSVs of 09_multi_model.py.)

Run (from repo root)
  python 12_real_speed.py --model Qwen/Qwen2.5-1.5B --quant fp16     (~15 min)
  python 12_real_speed.py --model Qwen/Qwen2.5-1.5B --quant 4bit
  python 12_real_speed.py --model Qwen/Qwen2.5-3B   --quant 4bit
  python 12_real_speed.py --model Qwen/Qwen2.5-7B   --quant 4bit     (~20 min)
  python 12_real_speed.py --model Qwen/Qwen2.5-14B  --quant 4bit     (Colab, Tesla T4)

Inputs
  results/csv/mm_<model>_union_*.csv, mm_<model>_quality_*.csv,
  mm_<model>_acceptance_*.csv (from 09_multi_model.py; fp16 only)

Outputs
  results/csv/speed_<model>_<quant>_<ts>.csv         (Part A, per prompt)
  results/csv/speed_sparse_<model>_<quant>_<ts>.csv  (Part B, T(N, f))
  results/csv/speed_summary.csv                      (one row appended per run)
  results/figures/fig08_real_speed_<model>_<quant>.png  (diagnostic plot, not in the report)
"""
import argparse
import datetime
import gc
import glob
import re
import math
import os
import time
import types

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
ap.add_argument("--quant", default="fp16", choices=["fp16", "4bit"])
ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--prompts", type=int, default=8, help="number of prompts")
ap.add_argument("--gen", type=int, default=128, help="tokens generated per prompt")
ap.add_argument("--quick", action="store_true", help="smoke test: 2 prompts, 32 tokens")
ap.add_argument("--cpu", action="store_true", help="for code testing only")
ARGS = ap.parse_args()
if ARGS.quick:
    ARGS.prompts, ARGS.gen = 2, 32

DEVICE = "cpu" if (ARGS.cpu or not torch.cuda.is_available()) else "cuda"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SHORT = ARGS.model.split("/")[-1]
MODEL_NAME = SHORT + (" (4-bit)" if ARGS.quant == "4bit" else "")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
CSV_DIR, FIG_DIR = "results/csv", "results/figures"
K_LIST = [1, 2, 3, 4, 6, 8]
F_LIST = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_LIST = [1, 2, 3, 4, 5, 7, 9]
WARMUP, REPEAT = (1, 3) if ARGS.quick else (3, 10)
SPARSITY = [0.10, 0.20, 0.30, 0.50]
DRAFT_QUANT = ["fp16"]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def free_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def get_text():
    from data import wikitext
    return wikitext()


def load(model_id, quant):
    if quant == "fp16" or DEVICE == "cpu":
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPE).to(DEVICE).eval()
    from transformers import AutoConfig, BitsAndBytesConfig
    tied = getattr(AutoConfig.from_pretrained(model_id), "tie_word_embeddings", False)
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             llm_int8_skip_modules=["lm_head"] if tied else [])
    return AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=cfg, device_map={"": 0}).eval()


# =====================================================================
# PART A - END TO END
# =====================================================================

@torch.no_grad()
def generate(model, x, eos, draft=None, k=None):
    kw = dict(max_new_tokens=ARGS.gen, min_new_tokens=ARGS.gen, do_sample=False,
              pad_token_id=eos, temperature=None, top_p=None, top_k=None)
    if draft is not None:
        gc_ = draft.generation_config
        gc_.num_assistant_tokens = k
        gc_.num_assistant_tokens_schedule = "constant"
        if hasattr(gc_, "assistant_confidence_threshold"):
            gc_.assistant_confidence_threshold = 0.0     # keep k fixed
        kw["assistant_model"] = draft
    sync(); t0 = time.perf_counter()
    out = model.generate(x, **kw)
    sync()
    return out[0, x.shape[1]:].cpu(), time.perf_counter() - t0


@torch.no_grad()
def single_token_time(model):
    x = torch.randint(0, min(1000, model.config.vocab_size), (1, 1), device=model.device)
    for _ in range(WARMUP + 2):
        model(x)
    s = []
    for _ in range(REPEAT * 3):
        sync(); t0 = time.perf_counter(); model(x); sync()
        s.append((time.perf_counter() - t0) * 1000)
    return float(np.median(s))


def part_a(model, draft, prompts, eos):
    print("\n[A] END-TO-END GENERATION")
    rows = []
    x0 = prompts[0].unsqueeze(0).to(model.device)
    generate(model, x0, eos)                                   # warm-up
    generate(model, x0, eos, draft, 2)
    for i, p in enumerate(prompts):
        x = p.unsqueeze(0).to(model.device)
        ref, t_ar = generate(model, x, eos)
        rows.append(dict(model=MODEL_NAME, prompt=i, config="baseline", k=0,
                         time_s=t_ar, tokens=len(ref), same_token=1.0))
        for k in K_LIST:
            o, t = generate(model, x, eos, draft, k)
            n = min(len(o), len(ref))
            same = float((o[:n] == ref[:n]).float().mean()) if n else 0.0
            rows.append(dict(model=MODEL_NAME, prompt=i, config="speculative", k=k,
                             time_s=t, tokens=len(o), same_token=same))
        print(f"\r  prompt {i + 1}/{len(prompts)}", end="", flush=True)
    print()
    df = pd.DataFrame(rows)
    df["tok_per_s"] = df.tokens / df.time_s
    return df


# =====================================================================
# PART B - REAL ROUND TIME WITH A SPARSE MLP (fp16 only)
# =====================================================================

class SparseConfig:
    f = 1.0          # active neuron fraction
    mode = "ideal"   # "ideal" (contiguous rows) or "gather" (index_select)


def prepare_mlp(model):
    """Adds a transposed copy of down_proj (row = neuron) to every MLP and
    replaces forward with the sparse version. Extra memory: one down_proj."""
    act = None
    for layer in model.model.layers:
        m = layer.mlp
        m._WdT = m.down_proj.weight.t().contiguous()          # [intermediate, hidden]
        m._d = m.gate_proj.weight.shape[0]
        g = torch.Generator().manual_seed(0)
        m._perm = torch.randperm(m._d, generator=g).to(m._WdT.device)
        act = m.act_fn

        def forward(self, x):
            f = SparseConfig.f
            if f >= 1.0:
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            n = max(1, int(round(self._d * f)))
            if SparseConfig.mode == "ideal":
                Wg, Wu, Wd = (self.gate_proj.weight[:n], self.up_proj.weight[:n],
                              self._WdT[:n])
            else:
                idx = self._perm[:n]
                Wg = self.gate_proj.weight.index_select(0, idx)
                Wu = self.up_proj.weight.index_select(0, idx)
                Wd = self._WdT.index_select(0, idx)
            h = self.act_fn(torch.nn.functional.linear(x, Wg)) * \
                torch.nn.functional.linear(x, Wu)
            return h @ Wd

        m.forward = types.MethodType(forward, m)
    return act


@torch.no_grad()
def round_time(model, N, vocab):
    x = torch.randint(0, vocab, (1, N), device=model.device)
    for _ in range(WARMUP):
        model(x)
    s = []
    for _ in range(REPEAT):
        sync(); t0 = time.perf_counter(); model(x); sync()
        s.append((time.perf_counter() - t0) * 1000)
    return float(np.median(s))


def part_b(model, vocab):
    print("\n[B] SPARSE MLP - REAL ROUND TIME")
    prepare_mlp(model)
    rows = []
    for mode in ("ideal", "gather"):
        SparseConfig.mode = mode
        for N in N_LIST:
            for f in F_LIST:
                SparseConfig.f = f
                rows.append(dict(model=MODEL_NAME, mode=mode, N=N, f=f,
                                 ms=round_time(model, N, vocab)))
            r = [d for d in rows if d["mode"] == mode and d["N"] == N]
            print(f"  {mode:<8} N={N}:  " + "  ".join(
                f"{d['f'] * 100:.0f}% {d['ms']:.1f}" for d in r))
    SparseConfig.f = 1.0
    return pd.DataFrame(rows)


# =====================================================================
# RESULT - four configurations
# =====================================================================

def find_csv(pattern):
    d = sorted(glob.glob(os.path.join(CSV_DIR, pattern)), key=lambda y: (re.findall(r"\d{8}_\d{4}", os.path.basename(y)) or [""])[-1])
    return d[-1] if d else None


def measurement_files():
    """Union / quality / acceptance CSVs written by 09 for this model."""
    prefix = f"mm_{SHORT}_"
    candidates = sorted(glob.glob(os.path.join(CSV_DIR, prefix + "quality_*.csv")),
                        key=lambda y: (re.findall(r"\d{8}_\d{4}", os.path.basename(y)) or [""])[-1])
    for q in reversed(candidates):
        ts = q.split("_quality_")[1][:-4]
        qual = pd.read_csv(q)
        if qual.model.iloc[0] != MODEL_NAME:
            continue
        u = os.path.join(CSV_DIR, f"{prefix}union_{ts}.csv")
        a = os.path.join(CSV_DIR, f"{prefix}acceptance_{ts}.csv")
        if os.path.exists(u) and os.path.exists(a):
            return pd.read_csv(u), qual, pd.read_csv(a), ts
    return None


def union_function(uni):
    PL = np.array(SPARSITY); al, c = [], []
    for p in PL:
        a = uni[(uni.active_frac == p) & (uni.k > 1) & (uni.k <= 8)].groupby("k").union.mean()
        keff = np.log(1 - a.values) / np.log(1 - p)
        s, lc = np.polyfit(np.log(a.index.values), np.log(keff), 1)
        al.append(s); c.append(math.exp(lc))
    al, c = np.array(al), np.array(c)
    return lambda n, p: 1.0 if p >= 1 else \
        1 - (1 - p) ** (np.interp(p, PL, c) * n ** np.interp(p, PL, al))


def result(dfA, dfB, td, meas):
    ar = dfA[dfA.config == "baseline"].tok_per_s.mean()
    spec = dfA[dfA.config == "speculative"].groupby("k").tok_per_s.mean()
    k_best = int(spec.idxmax())
    summary = dict(timestamp=TIMESTAMP, model=MODEL_NAME, draft_quant=DRAFT_QUANT[0], device=(torch.cuda.get_device_name(0)
                   if DEVICE == "cuda" else "CPU"), draft_ms=td,
                   baseline_tok_per_s=ar, spec_real=spec.max() / ar, spec_k=k_best,
                   lossless=float(dfA[dfA.config == "speculative"].same_token.mean()))
    print("\n" + "=" * 64 + f"\nREAL MEASUREMENT - {MODEL_NAME}\n" + "=" * 64)
    print(f"  baseline generation     : {ar:6.1f} tok/s")
    for k, v in spec.items():
        print(f"  speculative k={k:<2}        : {v:6.1f} tok/s  ({v / ar:.2f}x)")
    print(f"  lossless (same-token fraction): {summary['lossless']:.3f}")
    print(f"  draft time              : {td:.1f} ms/token")

    if dfB is None or meas is None:
        print("\n  Sparse configurations not measured"
              + (" (4-bit model)" if dfB is None else " (09 CSVs not found)")
              + " -> the estimate in 10_speed_model.py applies.")
        return summary, None

    uni, qual, acc, ts = meas
    U = union_function(uni)
    u = qual[qual.degradation_pct <= 5.0]
    D5 = float(u.active_frac.min()) if len(u) else 1.0
    d = acc[acc.distinct_4gram >= 0.5]
    KS = [1, 2, 3, 4, 6, 8]
    E = {k: d[f"tokens_k{k}"].sum() / d[f"rounds_k{k}"].sum() for k in KS}
    print(f"\n  (09 measurements: {ts}; 5% quality wall at active {D5 * 100:.0f}%; "
          f"non-repetitive a={d.a_greedy.mean():.3f})")

    rows = []
    for mode in ("ideal", "gather"):
        b = dfB[dfB["mode"] == mode]

        def T(N, f):
            Ns = sorted(b.N.unique())
            ys = [np.interp(f, b[b.N == n].f, b[b.N == n].ms) for n in Ns]
            return float(np.interp(N, Ns, ys))

        T1 = T(1, 1.0)
        s_ar = 1000 / T1
        s_sparse = 1000 / T(1, D5)
        s_spec = max(E[k] / (k * td + T(k + 1, 1.0)) * 1000 for k in KS)
        s_comb, k_comb = max((E[k] / (k * td + T(k + 1, U(k + 1, D5))) * 1000, k) for k in KS)
        rows.append(dict(mode=mode, baseline=1.0, spec_only=s_spec / s_ar,
                         sparse_only=s_sparse / s_ar, combined=s_comb / s_ar,
                         combined_k=k_comb))
        summary.update({f"{mode}_spec": s_spec / s_ar, f"{mode}_sparse": s_sparse / s_ar,
                        f"{mode}_combined": s_comb / s_ar, f"{mode}_combined_k": k_comb,
                        f"{mode}_saving_f{int(D5 * 100)}": 1 - T(1, D5) / T1})
    print(f"\n  FOUR CONFIGURATIONS (relative to baseline; sparse times are REAL, "
          f"perfect-predictor assumption)")
    print(f"  {'mode':<9} {'speculative':>11} {'sparsity':>10} {'combined':>9}")
    for r in rows:
        print(f"  {r['mode']:<9} {r['spec_only']:>10.2f}x {r['sparse_only']:>9.2f}x "
              f"{r['combined']:>8.2f}x  (k={r['combined_k']})")
    b = dfB[dfB["mode"] == "ideal"]
    t1 = b[(b.N == 1) & (b.f == 1.0)].ms.iloc[0]
    tf = float(np.interp(D5, b[b.N == 1].f, b[b.N == 1].ms))
    print(f"\n  REAL saving from sparsity (N=1, active {D5 * 100:.0f}%): "
          f"round {t1:.1f} -> {tf:.1f} ms ({(1 - tf / t1) * 100:.0f}% shorter)")
    return summary, pd.DataFrame(rows)


def plot(dfA, dfB, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = 2 if dfB is not None else 1
    fig, ax = plt.subplots(1, n, figsize=(6.5 * n, 4.6))
    ax = np.atleast_1d(ax)
    ar = dfA[dfA.config == "baseline"].tok_per_s.mean()
    s = dfA[dfA.config == "speculative"].groupby("k").tok_per_s.mean()
    ax[0].plot(s.index, s.values, "o-", color="#2a78d6", lw=2, label="speculative (measured)")
    ax[0].axhline(ar, color="#555", ls="--", label="baseline generation")
    ax[0].set(xlabel="draft length k", ylabel="tokens / second",
              title=f"A - Real generation speed, {MODEL_NAME}")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    if dfB is not None:
        colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7f7f7f"]
        b = dfB[dfB["mode"] == "ideal"]
        for N, c in zip(sorted(b.N.unique()), colors):
            r = b[b.N == N].sort_values("f")
            ax[1].plot(r.f * 100, r.ms, "o-", ms=4, lw=1.8, color=c, label=f"N={N}")
        ax[1].set(xlabel="active neuron fraction (%)", ylabel="round time (ms)",
                  title="B - Real round time with sparse MLP")
        ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150)


# =====================================================================

def main():
    os.makedirs(CSV_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    if ARGS.quant == "4bit" and DEVICE == "cuda":
        for lib in ("accelerate", "bitsandbytes"):
            if __import__("importlib.util").util.find_spec(lib) is None:
                raise SystemExit(f"Missing library: pip install {lib}")
    print(f"device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    print(f"model : {ARGS.model} ({ARGS.quant})   draft: {ARGS.draft}"
          f"   {ARGS.prompts} prompts x {ARGS.gen} tokens")

    tok = AutoTokenizer.from_pretrained(ARGS.model)
    if AutoTokenizer.from_pretrained(ARGS.draft).get_vocab() != tok.get_vocab():
        raise SystemExit("Draft and target vocabularies differ.")
    ids = tok(get_text(), return_tensors="pt").input_ids[0]
    step = len(ids) // (ARGS.prompts + 1)
    prompts = [ids[(i + 1) * step:(i + 1) * step + 128] for i in range(ARGS.prompts)]

    model = load(ARGS.model, ARGS.quant)
    draft_quant = "fp16"
    DRAFT_QUANT[0] = "fp16"
    try:
        draft = load(ARGS.draft, "fp16")
        if DEVICE == "cuda":                      # leave ~0.4 GB free for generation
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free < 0.4:
                raise torch.cuda.OutOfMemoryError(f"only {free:.2f} GB free")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ! fp16 draft did not fit ({e}); loading 4-bit draft")
        draft = None; free_memory()
        draft = load(ARGS.draft, "4bit"); draft_quant = DRAFT_QUANT[0] = "4bit"
    td = single_token_time(draft)
    # The 7B/14B embedding table has 152064 rows, the smaller models 151936. The
    # vocabulary is the SAME (checked above), only the table size differs; Hugging
    # Face treats this as a "different vocabulary" and stops, so the draft's
    # declared size is matched to the target. In greedy generation the draft only
    # proposes token ids; its probabilities are not used.
    vh = model.config.get_text_config().vocab_size
    if draft.config.get_text_config().vocab_size != vh:
        print(f"  note: embedding table size matched "
              f"({draft.config.get_text_config().vocab_size} -> {vh}); vocabulary identical")
        draft.config.vocab_size = vh
        draft.config.get_text_config().vocab_size = vh
    if DEVICE == "cuda":
        print(f"  memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB   "
              f"draft {draft_quant}, {td:.1f} ms/token")

    dfA = part_a(model, draft, prompts, tok.eos_token_id)
    dfA.to_csv(f"{CSV_DIR}/speed_{SHORT}_{ARGS.quant}_{TIMESTAMP}.csv", index=False)

    dfB = None
    if ARGS.quant == "fp16":
        del draft; free_memory()
        dfB = part_b(model, len(tok))
        dfB.to_csv(f"{CSV_DIR}/speed_sparse_{SHORT}_{ARGS.quant}_{TIMESTAMP}.csv", index=False)
    else:
        print("\n[B] skipped: neuron rows cannot be selected in 4-bit weights.")

    summary, table = result(dfA, dfB, td, measurement_files() if dfB is not None else None)
    path = f"{CSV_DIR}/speed_summary.csv"
    pd.DataFrame([summary]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    g = f"{FIG_DIR}/fig08_real_speed_{SHORT}_{ARGS.quant}.png"
    plot(dfA, dfB, g)
    print(f"\nCSV    : {CSV_DIR}/speed_*{SHORT}_{ARGS.quant}_{TIMESTAMP}.csv  (+ row: {path})")
    print(f"FIGURE : {g}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
