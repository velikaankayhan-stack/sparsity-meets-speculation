# -*- coding: utf-8 -*-
"""
09_multi_model.py — All measurements for a single model (model size comparison)

Repeats the same measurements on models of different sizes to answer
"does the result depend on model size?". Each run is for ONE large (verifier)
model; the small (draft) model is always Qwen2.5-0.5B.

FIVE MEASUREMENTS (same model, same GPU)
  1. Union      : fraction of neurons that must be fetched when N tokens are processed
                  in the same round, α
  2. Quality    : perplexity degradation as the active neuron fraction drops (quality wall)
  3. Timing     : how round time grows with N, free budget
  4. Acceptance : how often this model accepts the 0.5B draft model's predictions
                  (same method as 07_acceptance.py)
  5. Decision   : the 08_decision_surface.py calculation, using this model's own measurements

Union, quality and acceptance belong to the MODEL (same on any GPU).
Timing and decision belong to the GPU; the GPU name is written to the output files.

HOW TO RUN
  python 09_multi_model.py --model Qwen/Qwen2.5-1.5B --quant fp16     # local GPU
  python 09_multi_model.py --model Qwen/Qwen2.5-3B                    # local GPU (4-bit)
  python 09_multi_model.py --model Qwen/Qwen2.5-7B                    # local GPU (4-bit)
  python 09_multi_model.py --model Qwen/Qwen2.5-14B                   # Colab T4
  python 09_multi_model.py --model Qwen/Qwen2.5-32B                   # Colab Pro (A100/L4)

  Extra options: --no-acceptance (skip acceptance), --no-timing (skip timing),
                 --quick (very short run for testing the code)

SELF-CHECKS
1. Stops if the data cannot be loaded (data.py, no silent fallback).
2. Union: at N=1 the union must equal p exactly.
3. Quality: at 100% active, perplexity must match the hook-free run (within 0.5%).
4. Acceptance: the draft model against itself must give ~1 (self-test).

OUTPUTS (names deliberately differ so they do not affect other scripts' CSV selection)
results/csv/mm_<model>_union_<ts>.csv   (+ _union_all_N_)
results/csv/mm_<model>_quality_<ts>.csv
results/csv/mm_<model>_timing_<ts>.csv
results/csv/mm_<model>_acceptance_<ts>.csv
results/csv/mm_summary.csv            <- each run APPENDS one row (for the size figure)
results/figures/mm_<model>_<ts>.png

REQUIRES: pip install accelerate bitsandbytes datasets
"""

import os, gc, sys, glob, math, time, argparse, datetime
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# SETTINGS
# =====================================================================

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
ap.add_argument("--quant", default="4bit", choices=["4bit", "fp16"])
ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--no-acceptance", action="store_true")
ap.add_argument("--no-timing", action="store_true")
ap.add_argument("--quick", action="store_true")
ARG = ap.parse_args() if __name__ == "__main__" else ap.parse_args([])

MODEL_ID = ARG.model
SHORT = MODEL_ID.split("/")[-1]
MODEL_NAME = SHORT + (" (4-bit)" if ARG.quant == "4bit" else "")
Q = ARG.quick

ACTIVE_FRACS = [0.10, 0.20, 0.30, 0.50]
N_LIST = [1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256]
UNION_SEQ, UNION_WINDOWS = 512, (1 if Q else 4)
P_LIST = [1.00, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
QUAL_SEQ, QUAL_WINDOWS = 512, (2 if Q else 40)
CHUNK = 128
TIMING_N = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
WARMUP, REPEATS = (1, 2) if Q else (3, 7)
N_PROMPTS, PROMPT_LEN, GEN_LEN = (3, 128, 32) if Q else (40, 128, 128)
TOPK = 64
K_SIM = [1, 2, 3, 4, 6, 8, 12, 16, 21, 32, 48, 64, 128]
FREE_THRESHOLD = 1.25
KS = np.arange(1, 129)

CSV_DIR, FIG_DIR = "results/csv", "results/figures"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX = f"{CSV_DIR}/mm_{SHORT}"


# =====================================================================
# HELPERS
# =====================================================================

def load_text():
    """There is NO silent fallback. The script stops if the data cannot be loaded."""
    from data import wikitext
    return wikitext()


def mlp_share(cfg, head_factor=1.0):
    """Share of the BYTES read per round that belong to the MLP (lm_head included, embedding excluded).
    head_factor: how many times more bytes/param lm_head holds than the body
    (4-bit body + 16-bit lm_head -> 4)."""
    h, i, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * hd
    mlp = 3 * h * i * L
    attention = L * (2 * h * h + 2 * h * kv)
    head = cfg.vocab_size * h * head_factor
    return mlp / (mlp + attention + head)


def param_count(cfg):
    """Approximate parameter count from the config (numel() is misleading for 4-bit models)."""
    h, i, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * hd
    embed = cfg.vocab_size * h * (1 if getattr(cfg, "tie_word_embeddings", False) else 2)
    return int(L * (3 * h * i + 2 * h * h + 2 * h * kv) + embed)


def free_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def load_model(model_id, quant):
    """Returns: (model, fully_on_gpu)"""
    if quant == "fp16":
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE).eval()
        return m, True
    from transformers import BitsAndBytesConfig
    # In models whose embedding is tied to lm_head (0.5B, 1.5B, 3B) lm_head cannot be 4-bit:
    # it shares its weight with the embedding table. lm_head then stays 16-bit.
    tied = getattr(AutoConfig.from_pretrained(model_id), "tie_word_embeddings", False)
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                llm_int8_skip_modules=["lm_head"] if tied else [])
    try:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=config, device_map={"": 0})
        full = True
    except (torch.cuda.OutOfMemoryError, ValueError, RuntimeError) as e:
        print(f"  did not fit fully on the GPU ({type(e).__name__}); offloading some layers to RAM")
        free_memory()
        avail = torch.cuda.get_device_properties(0).total_memory / 2**30 - 1.2
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=config, device_map="auto",
            max_memory={0: f"{avail:.1f}GiB", "cpu": "48GiB"})
        full = False
    return m.eval(), full


def mlp_layers(model):
    k = [(a, m) for a, m in model.named_modules() if a.endswith("mlp.down_proj")]
    if not k:
        raise RuntimeError("mlp.down_proj not found — architecture not recognised")
    return k


def body(model):
    """The body before lm_head (Qwen/Llama family: model.model)."""
    return model.model


# =====================================================================
# 1) UNION
# =====================================================================

class UnionHook:
    def __init__(self):
        self.result, self.idx = {}, {}

    def hook(self, name):
        def _f(mod, inputs):
            x = inputs[0][0].abs().float()
            L, d = x.shape
            idx = x.topk(int(d * max(ACTIVE_FRACS)), dim=-1, sorted=True).indices
            for p in ACTIVE_FRACS:
                mask = torch.zeros(L, d, dtype=torch.int32, device=x.device)
                mask.scatter_(1, idx[:, :max(1, int(d * p))], 1)
                cs = torch.zeros(L + 1, d, dtype=torch.int32, device=x.device)
                cs[1:] = mask.cumsum(0)
                for N in N_LIST:
                    if N <= L:
                        u = ((cs[N:] - cs[:-N]) > 0).sum(1).float().mean().item() / d
                        self.result.setdefault((self.idx[name], p, N), []).append(u)
                del mask, cs
        return _f


@torch.no_grad()
def measure_union(model, ids):
    print("\n[1/5] UNION")
    uh = UnionHook()
    layers = mlp_layers(model)
    for i, (a, _) in enumerate(layers):
        uh.idx[a] = i
    handles = [m.register_forward_pre_hook(uh.hook(a)) for a, m in layers]
    try:
        for w in range(UNION_WINDOWS):
            chunk = ids[w * UNION_SEQ:(w + 1) * UNION_SEQ].unsqueeze(0).to(model.device)
            body(model)(chunk)
            print(f"\r  window {w + 1}/{UNION_WINDOWS}", end="", flush=True)
    finally:
        for t in handles:
            t.remove()
    print()
    df = pd.DataFrame([dict(model=MODEL_NAME, layer=k, active_frac=p, k=N,
                            union=float(np.mean(v)), random=1 - (1 - p) ** N)
                       for (k, p, N), v in uh.result.items()])
    for p in ACTIVE_FRACS:
        u1 = df[(df.active_frac == p) & (df.k == 1)].union.mean()
        if abs(u1 - p) > 0.01:
            raise RuntimeError(f"Union check failed: p={p}, {u1:.3f} at N=1")
    print("  check passed (union = p at N=1)")
    return df


def fit_alpha(df):
    par = {}
    for p in ACTIVE_FRACS:
        a = df[(df.active_frac == p) & (df.k > 1) & (df.k <= 8)].groupby("k").union.mean()
        keff = np.log(1 - a.values) / np.log(1 - p)
        al, lc = np.polyfit(np.log(a.index.values), np.log(keff), 1)
        par[p] = (float(al), float(math.exp(lc)))
    return par


# =====================================================================
# 2) QUALITY
# =====================================================================

class Sparsifier:
    def __init__(self, frac):
        self.frac, self.hooks = frac, []

    def _hook(self, mod, inputs):
        x = inputs[0]
        d = x.shape[-1]
        keep = max(1, int(round(d * self.frac)))
        if self.frac >= 1.0 or keep >= d:
            return None
        idx = x.abs().topk(keep, dim=-1, sorted=False).indices
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        return (x * mask,) + inputs[1:]

    def attach(self, model):
        for _, m in mlp_layers(model):
            self.hooks.append(m.register_forward_pre_hook(self._hook))

    def detach(self):
        for k in self.hooks:
            k.remove()
        self.hooks = []


@torch.no_grad()
def perplexity(model, ids):
    nll, n = 0.0, 0
    for w in range(QUAL_WINDOWS):
        chunk = ids[w * QUAL_SEQ:(w + 1) * QUAL_SEQ]
        if len(chunk) < QUAL_SEQ:
            break
        chunk = chunk.unsqueeze(0).to(model.device)
        h = body(model)(chunk).last_hidden_state[0]
        target = chunk[0, 1:]
        for s in range(0, QUAL_SEQ - 1, CHUNK):
            e = min(s + CHUNK, QUAL_SEQ - 1)
            logit = model.lm_head(h[s:e]).float()
            nll += torch.nn.functional.cross_entropy(
                logit, target[s:e].to(logit.device), reduction="sum").item()
            n += e - s
    return math.exp(nll / n), n


def measure_quality(model, ids):
    print("\n[2/5] QUALITY")
    clean, n = perplexity(model, ids)
    print(f"  hook-free perplexity: {clean:.3f}  ({n:,} tokens)")
    rows, base = [], None
    for p in P_LIST:
        s = Sparsifier(p); s.attach(model)
        try:
            ppl, _ = perplexity(model, ids)
        finally:
            s.detach()
        if p >= 1.0:
            if abs(ppl - clean) / clean > 0.005:
                raise RuntimeError(f"Quality check failed: {ppl:.3f} vs {clean:.3f}")
            base = ppl
            print("  check passed")
        deg = (ppl / base - 1) * 100
        rows.append(dict(model=MODEL_NAME, active_frac=p, ppl=ppl,
                         base_ppl=base, degradation_pct=deg, tokens_evaluated=n))
        print(f"  active {p * 100:3.0f}%   PPL {ppl:8.3f}   degradation {deg:+7.2f}%")
    return pd.DataFrame(rows)


def wall(qual, limit):
    u = qual[qual.degradation_pct <= limit]
    return float(u.active_frac.min()) if len(u) else 1.0


# =====================================================================
# 3) TIMING
# =====================================================================

@torch.no_grad()
def measure_timing(model, vocab):
    print("\n[3/5] TIMING")
    rows = []
    for N in TIMING_N:
        inp = torch.randint(0, vocab, (1, N), device=model.device)
        try:
            for _ in range(WARMUP):
                model(inp)
            torch.cuda.synchronize()
            s = []
            for _ in range(REPEATS):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model(inp)
                torch.cuda.synchronize(); s.append((time.perf_counter() - t0) * 1000)
        except torch.cuda.OutOfMemoryError:
            print(f"  N={N:<5} out of memory, stopped")
            free_memory()
            break
        ms, sp = float(np.median(s)), float(np.std(s))
        rows.append(dict(model=MODEL_NAME, N=N, ms=ms, std_ms=sp,
                         device=torch.cuda.get_device_properties(0).name))
        print(f"  N={N:<5} {ms:8.1f} ms  (±{sp:.1f})")
    return pd.DataFrame(rows)


# =====================================================================
# 4) ACCEPTANCE — same two-stage method as 07_acceptance.py
# =====================================================================

def prepare_prompts(ids):
    step = len(ids) // (N_PROMPTS + 1)
    return [ids[(i + 1) * step:(i + 1) * step + PROMPT_LEN] for i in range(N_PROMPTS)]


@torch.no_grad()
def target_generate(model, prompts, eos):
    recs = []
    for i, pr in enumerate(prompts):
        x = pr.unsqueeze(0).to(model.device)
        c = model.generate(x, max_new_tokens=GEN_LEN, min_new_tokens=GEN_LEN,
                           do_sample=False, pad_token_id=eos,
                           temperature=None, top_p=None, top_k=None)
        cont = c[0, len(pr):]
        P = len(pr)
        h = body(model)(c).last_hidden_state[0, P - 1:P - 1 + len(cont)]
        top_p, top_i, arg = [], [], []
        for s in range(0, len(cont), CHUNK):
            lg = model.lm_head(h[s:s + CHUNK]).float()
            o = torch.softmax(lg, -1).topk(TOPK, -1)
            top_p.append(o.values.half().cpu()); top_i.append(o.indices.cpu())
            arg.append(lg.argmax(-1).cpu())
        recs.append(dict(prompt=pr.cpu(), cont=cont.cpu(), top_p=torch.cat(top_p),
                         top_i=torch.cat(top_i),
                         consistency=(torch.cat(arg) == cont.cpu()).float().mean().item()))
        print(f"\r  generation {i + 1}/{len(prompts)}", end="", flush=True)
    print()
    return recs


@torch.no_grad()
def draft_compare(model, recs):
    for r in recs:
        seq = torch.cat([r["prompt"], r["cont"]]).unsqueeze(0).to(model.device)
        P, L = len(r["prompt"]), len(r["cont"])
        q = torch.softmax(model(seq).logits[0, P - 1:P - 1 + L].float(), -1)
        r["match"] = (q.argmax(-1).cpu() == r["cont"]).numpy()
        up, ui = r["top_p"].to(q.device).float(), r["top_i"].to(q.device)
        r["a_samp"] = float(torch.minimum(up, torch.gather(q, 1, ui)).sum(-1).mean())
        r["tail"] = float((1 - up.sum(-1)).mean())
    return recs


def distinct_4gram(ids):
    ids = ids.tolist()
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / max(1, len(g))


def chain(matches, k):
    j = tokens = rounds = 0
    L = len(matches)
    while j < L:
        s = 0
        while s < k and j + s < L and matches[j + s]:
            s += 1
        a = min(s + 1, L - j); tokens += a; j += a; rounds += 1
    return tokens, rounds


def acceptance_table(recs):
    rows = []
    for n, r in enumerate(recs):
        s = dict(target=MODEL_NAME, prompt_id=n, a_greedy=float(r["match"].mean()),
                 a_sampling=r["a_samp"], tail=r["tail"], consistency=r["consistency"],
                 distinct_4gram=distinct_4gram(r["cont"]))
        for k in K_SIM:
            s[f"tokens_k{k}"], s[f"rounds_k{k}"] = chain(r["match"], k)
        rows.append(s)
    return pd.DataFrame(rows)


def draft_self_test(tok, prompts):
    print("  self-test (draft against itself)")
    t, _ = load_model(ARG.draft, "fp16")
    recs = draft_compare(t, target_generate(t, prompts[:3], tok.eos_token_id))
    del t; free_memory()
    a = np.mean([r["match"].mean() for r in recs])
    if a < 0.98:
        raise RuntimeError(f"Self-test failed (a={a:.3f}); alignment code is broken.")
    print(f"  a = {a:.3f}, passed")


# =====================================================================
# 5) DECISION
# =====================================================================

def timing_envelope(timing):
    """Lower envelope of the measured times: for each N, the smallest time at N or above
    (removes warm-up spikes); N values in between are linearly interpolated."""
    s = timing.sort_values("N")
    N, ms = s.N.values.astype(float), s.ms.values.astype(float)
    env = np.minimum.accumulate(ms[::-1])[::-1]
    return lambda n: float(np.interp(n, N, env))


@torch.no_grad()
def draft_time(t):
    """Time for the draft model to produce one token (ms), on the same GPU."""
    x = torch.randint(0, 1000, (1, 1), device=t.device)
    for _ in range(WARMUP + 2):
        t(x)
    s = []
    for _ in range(REPEATS * 3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        t(x)
        torch.cuda.synchronize(); s.append((time.perf_counter() - t0) * 1000)
    return float(np.median(s))


def decision(uni, qual, timing, acc, F, td):
    """SPEED MODEL (old summary without draft cost; the report's speed calculation is in 10_speed_model.py)
      one round = k·td  +  T(k+1)  −  T(1)·F·(1 − U(k+1, p))
                  draft    verify      reads saved by sparsity
      speed     = E(k) / round
    T: lower envelope of the measured times (no formula). td: draft time per token.
    Baseline: plain generation, 1 token / T(1)."""
    par = fit_alpha(uni)
    PL = np.array(ACTIVE_FRACS)
    al_ = np.array([par[p][0] for p in PL]); c_ = np.array([par[p][1] for p in PL])

    def U(n, p):
        return 1.0 if p >= 1.0 else 1 - (1 - p) ** (np.interp(p, PL, c_) * n ** np.interp(p, PL, al_))

    T = timing_envelope(timing); T1 = T(1)
    if acc is not None and len(acc):
        d = acc[acc.distinct_4gram >= 0.5]
        ey = [d[f"tokens_k{k}"].sum() / d[f"rounds_k{k}"].sum() for k in K_SIM]
        E = lambda k: float(np.interp(np.log(k), np.log(K_SIM), ey))
        src = f"measured (a={d.a_greedy.mean():.2f})"
    else:
        E = lambda k: (1 - 0.75 ** (k + 1)) / 0.25
        src = "formula, a=0.75 (acceptance not measured)"

    def speed(k, p, td_):
        return E(k) / ((k * td_ + T(k + 1) - T1 * F * (1 - U(k + 1, p))) / 1000)

    ps = np.round(np.linspace(0.10, 1.0, 91), 3)

    def best_point(pmin, td_):
        best = (0, None, None)
        for p in ps[ps >= pmin - 1e-9][::-1]:          # on a tie, less sparsity wins
            h = np.array([speed(int(k), p, td_) for k in KS]); i = int(h.argmax())
            if h[i] > best[0] * 1.001:
                best = (float(h[i]), int(KS[i]), float(p))
        return best

    D1, D5 = wall(qual, 1.0), wall(qual, 5.0)
    none, b5, b1, free = (best_point(x, td) for x in (1.0, D5, D1, 0.10))
    base = 1000 / T1
    sparse = 1000 / (T1 * (1 - F * (1 - D5)))
    print(f"\n[5/5] DECISION   (E: {src}, F_MLP={F:.2f}, T(1)={T1:.1f} ms, "
          f"draft {td:.1f} ms/token = {td / T1:.2f}× T(1))")
    for name, (h, k, p) in [("no sparsity", none), ("1% quality budget", b1),
                            ("5% quality budget", b5), ("unconstrained", free)]:
        print(f"  {name:<20} k={k:>3}  active {p * 100:3.0f}%  {h:6.1f} tokens/s  "
              f"({h / none[0]:.3f}×)")
    print(f"  vs plain generation: speculative {none[0] / base:.2f}×, sparsity {sparse / base:.2f}×, "
          f"both combined {b5[0] / base:.2f}×")
    print("  sensitivity — draft cost (multiple of T(1)) -> gain from sparsity (5% budget):")
    for r in (0.0, 0.1, 0.2, 0.3):
        y, b = best_point(1.0, r * T1), best_point(D5, r * T1)
        print(f"    {r:.1f}: k={y[1]:>3}  gain {b[0] / y[0]:.3f}×")
    return dict(draft_ms=td, draft_ratio=td / T1, gain_5=b5[0] / none[0],
                gain_1=b1[0] / none[0], unconstrained=free[0] / none[0],
                spec_only=none[0] / base, sparse_only=sparse / base, combined=b5[0] / base,
                best_k=none[1], speed=none[0], M_ms=T1)


# =====================================================================
# FIGURE
# =====================================================================

def plot(uni, qual, timing, acc, path):
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    for p in ACTIVE_FRACS:
        a = uni[uni.active_frac == p].groupby("k").union.mean()
        ax[0].plot(a.index, a.values * 100, "o-", lw=2, label=f"active {p * 100:.0f}%")
        ax[0].plot(a.index, [(1 - (1 - p) ** n) * 100 for n in a.index], "--", c="grey", lw=1, alpha=.5)
    ax[0].set_xscale("log", base=2); ax[0].set_ylim(0, 105)
    ax[0].set_xlabel("tokens processed in the same round (N)"); ax[0].set_ylabel("neurons fetched (%)")
    ax[0].set_title(f"1 — Union · {MODEL_NAME}"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

    k = qual.sort_values("active_frac")
    ax[1].plot(k.active_frac * 100, k.degradation_pct, "o-", lw=2, c="#2ca02c")
    for s, r in [(1, "#16a34a"), (5, "#ea580c")]:
        ax[1].axhline(s, ls="--", lw=1.1, c=r)
    ax[1].set_yscale("symlog", linthresh=1); ax[1].set_xlabel("active neuron fraction (%)")
    ax[1].set_ylabel("perplexity degradation (%)"); ax[1].set_title("2 — Quality wall"); ax[1].grid(alpha=.3)

    if timing is not None and len(timing):
        ax[2].plot(timing.N, timing.ms, "o-", lw=2, c="#8c564b")
        ax[2].axhline(timing.ms.iloc[0] * FREE_THRESHOLD, ls=":", c="k")
        ax[2].set_xscale("log", base=2); ax[2].set_yscale("log")
        ax[2].set_xlabel("positions verified in one round (N)"); ax[2].set_ylabel("round time (ms)")
        ax[2].set_title(f"3 — Round time · {timing.device.iloc[0]}", fontsize=9); ax[2].grid(alpha=.3, which="both")
    else:
        ax[2].text(.5, .5, "timing not measured", ha="center", va="center"); ax[2].axis("off")

    if acc is not None and len(acc):
        d = acc.sort_values("a_greedy").reset_index(drop=True)
        ax[3].bar(range(len(d)), d.a_greedy, color=np.where(d.distinct_4gram >= 0.5, "#1f77b4", "#c7c7c7"))
        a = acc[acc.distinct_4gram >= 0.5].a_greedy.mean()
        ax[3].axhline(a, c="#1f77b4"); ax[3].axhline(0.6, ls="--", c="#d62728")
        ax[3].set_ylim(0.4, 1); ax[3].set_xlabel("text (sorted)"); ax[3].set_ylabel("acceptance rate")
        ax[3].set_title(f"4 — Acceptance rate (non-repetitive mean {a:.2f})"); ax[3].grid(alpha=.3, axis="y")
    else:
        ax[3].text(.5, .5, "acceptance not measured", ha="center", va="center"); ax[3].axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=140, bbox_inches="tight"); plt.close()


# =====================================================================
# MAIN
# =====================================================================

def main():
    os.makedirs(CSV_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    if DEVICE != "cuda":
        raise SystemExit("This script requires a GPU.")
    if ARG.quant == "4bit":
        for lib in ("accelerate", "bitsandbytes"):
            if __import__("importlib.util").util.find_spec(lib) is None:
                raise SystemExit(f"Missing library: pip install {lib}")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU   : {gpu.name}  ({gpu.total_memory / 2**30:.1f} GB)")
    print(f"model : {MODEL_ID}  ({ARG.quant})   draft: {ARG.draft}"
          f"{'   [QUICK MODE]' if Q else ''}\n")

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    lm16 = ARG.quant == "4bit" and getattr(cfg, "tie_word_embeddings", False)
    F = mlp_share(cfg, 4.0 if lm16 else 1.0)
    text = load_text()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(text, return_tensors="pt").input_ids[0]

    do_acceptance = not ARG.no_acceptance
    if do_acceptance:
        tt = AutoTokenizer.from_pretrained(ARG.draft)
        if tt.get_vocab() != tok.get_vocab():
            print("! draft has a different vocabulary: acceptance will not be measured")
            do_acceptance = False
    prompts = prepare_prompts(ids)

    model, full = load_model(MODEL_ID, ARG.quant)
    print(f"  memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB   "
          f"{'fully on GPU' if full else 'PARTLY IN RAM -> timing will not be measured'}   F_MLP={F:.2f}"
          f"{'  (lm_head 16-bit)' if lm16 else ''}")

    uni = measure_union(model, ids)
    uni[uni.k <= 8].to_csv(f"{PREFIX}_union_{TIMESTAMP}.csv", index=False)
    uni.to_csv(f"{PREFIX}_union_all_N_{TIMESTAMP}.csv", index=False)

    qual = measure_quality(model, ids)
    qual.to_csv(f"{PREFIX}_quality_{TIMESTAMP}.csv", index=False)

    timing = None
    if full and not ARG.no_timing:
        timing = measure_timing(model, cfg.vocab_size)
        timing.to_csv(f"{PREFIX}_timing_{TIMESTAMP}.csv", index=False)
    else:
        print("\n[3/5] TIMING skipped" + ("" if full else " (model partly in RAM)"))

    acc, recs = None, None
    DRAFT_MS = [35.0]          # assumed if acceptance is not measured (measured on RTX 3050)
    if do_acceptance:
        print("\n[4/5] ACCEPTANCE — stage 1: this model generates")
        recs = target_generate(model, prompts, tok.eos_token_id)
    del model; free_memory()
    if do_acceptance:
        draft_self_test(tok, prompts)
        print("  stage 2: draft compares")
        t, _ = load_model(ARG.draft, "fp16")
        acc = acceptance_table(draft_compare(t, recs))
        DRAFT_MS[0] = draft_time(t)
        print(f"  draft time: {DRAFT_MS[0]:.1f} ms/token")
        del t; free_memory()
        acc.to_csv(f"{PREFIX}_acceptance_{TIMESTAMP}.csv", index=False)
        d = acc[acc.distinct_4gram >= 0.5]
        print(f"  acceptance rate: {acc.a_greedy.mean():.3f} (all), "
              f"{d.a_greedy.mean():.3f} (non-repetitive {len(d)}/{len(acc)})")

    plot(uni, qual, timing, acc, f"{FIG_DIR}/mm_{SHORT}_{TIMESTAMP}.png")

    par = fit_alpha(uni)
    summary = dict(timestamp=TIMESTAMP, model=MODEL_NAME, model_id=MODEL_ID, quant=ARG.quant,
                   n_param=param_count(cfg), F_MLP=round(F, 3),
                   alpha_p20=par[0.20][0], U_p20_N4=float(uni[(uni.active_frac == .2) & (uni.k == 4)].union.mean()),
                   U_p20_N32=float(uni[(uni.active_frac == .2) & (uni.k == 32)].union.mean()),
                   wall_1=wall(qual, 1.0), wall_5=wall(qual, 5.0),
                   ppl_base=float(qual.base_ppl.iloc[0]), device=gpu.name, quick=Q)
    if acc is not None:
        summary.update(a_all=float(acc.a_greedy.mean()),
                       a_nonrep=float(acc[acc.distinct_4gram >= 0.5].a_greedy.mean()))
    if timing is not None and len(timing) >= 4:
        Tz = timing_envelope(timing)
        summary["free_N"] = int(max(n for n in timing.N if Tz(n) <= Tz(1) * FREE_THRESHOLD))
        summary.update(decision(uni, qual, timing, acc, F, DRAFT_MS[0]))
    path = f"{CSV_DIR}/mm_summary.csv"
    pd.DataFrame([summary]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    print("\n" + "=" * 60 + f"\nSUMMARY — {MODEL_NAME}\n" + "=" * 60)
    for k, v in summary.items():
        if k not in ("timestamp", "model_id", "n_param", "quick"):
            print(f"  {k:<14}: {v:.3f}" if isinstance(v, float) else f"  {k:<14}: {v}")
    print(f"\nCSV    : {PREFIX}_*_{TIMESTAMP}.csv  (+ row appended: {path})")
    print(f"FIGURE : {FIG_DIR}/mm_{SHORT}_{TIMESTAMP}.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
