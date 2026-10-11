# -*- coding: utf-8 -*-
"""
07_acceptance.py - Measuring the acceptance rate (speculative decoding)

Until now the rate at which the target model accepts the draft model's
guesses was ASSUMED to be a = 0.70. This script actually measures a.

It also computes E(k) (tokens gained per round) in two ways:
  1. Formula: E(k) = (1 - a^(k+1)) / (1 - a)   (Leviathan et al., 2023)
     This assumes every guess is accepted INDEPENDENTLY of the others.
  2. Simulation: chain speculative decoding is replayed round by round on the
     real accept/reject sequence. No independence assumption.
If the two differ, acceptances are correlated (similar to the union law).

Why two stages? With greedy speculative decoding the output text is IDENTICAL
to what the target model would generate on its own. The draft's i-th guess is
accepted only if it equals the target's i-th token. Therefore:
  STAGE 1 - The target model is loaded alone and generates 128 tokens per
            prompt. The generated tokens and the top 64 values of the
            probability distribution at each position are stored. The model
            is then freed.
  STAGE 2 - The draft model is loaded, reads the same text in one pass and
            compares its own guess at each position with the target's token.
The two models are never in memory at the same time, so a 4-bit 7B target can
be measured on a 6 GB card.

Two acceptance rates are reported:
  a_greedy   : is the draft's most likely token = the target's token?
               (temperature 0, the case our experiments assume)
  a_sampling : sum_x min(p(x), q(x))  (expected acceptance rate at
               temperature 1, Leviathan et al., 2023). Only the target's top 64
               tokens are stored, so this can be underestimated by at most the
               "tail"; the tail is reported too.

Self-checks:
  1. Stops if data cannot be loaded (no silent fallback, data.py).
  2. SELF-TEST: the draft model is measured against itself; acceptance should
     be ~1. Below 0.98 the alignment (indexing) code is broken and the script
     stops.
  3. Consistency: the token the target picked while generating is compared
     with the token it picks when reading the same text in one pass (should be
     identical apart from fp16 differences). Warns below 97%.
  4. Repetition check: base models may repeat the same sentence under greedy
     generation, which inflates the acceptance rate artificially. Prompts with
     a distinct-4-gram ratio below 50% are flagged and the summary is also
     given without them.

NOTE: for a 4-bit target, a is measured against the 4-bit model. This is
correct, since the target will also run in 4-bit in real use.

Inputs : WikiText-2 test text via data.wikitext()
Outputs: results/csv/acceptance_<ts>.csv
         results/figures/acceptance_<ts>.png
Run    :
  python 07_acceptance.py          # full measurement (~10-15 min, RTX 3050)
  python 07_acceptance.py --quick   # 3 prompts, 32 tokens: just to check the code runs
  python 07_acceptance.py 3B       # only measure targets whose name contains "3B"
  python 07_acceptance.py 7B       # only 7B (~25 min)
Requires (for 4-bit targets): pip install accelerate bitsandbytes
"""

import os, gc, sys, math, datetime
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# SETTINGS
# =====================================================================

DRAFT = "Qwen/Qwen2.5-0.5B"

# (model id, display name, quantization: None = fp16, "4bit" = bitsandbytes NF4)
TARGETS = [
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B", None),
    ("Qwen/Qwen2.5-3B",   "Qwen2.5-3B (4-bit)", "4bit"),
    # 7B: the output layer (lm_head) is 4-bit too, otherwise it does not fit in 6 GB (~5.1 GB)
    ("Qwen/Qwen2.5-7B",   "Qwen2.5-7B (4-bit)", "4bit_full"),
]

FAST = "--quick" in sys.argv
SELECTION = [a for a in sys.argv[1:] if not a.startswith("--")]   # e.g. 3B
NUM_PROMPTS = 3  if FAST else 40       # number of different text passages
PROMPT_LEN = 128                       # tokens per prompt
GEN_LEN = 32 if FAST else 128          # tokens generated per prompt
TOPK = 64                              # number of stored probabilities
K_LIST = [1, 2, 3, 4, 6, 8, 12, 16, 21, 32, 48, 64, 128]

# Same timing model as 08_decision_surface.py (Qwen2.5-0.5B, RTX 3050): T(k) = M + c2*k^2
T_M, T_C2 = 64.13, 1.796e-4

DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_DIR, FIGURE_DIR = "results/csv", "results/figures"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =====================================================================
# DATA
# =====================================================================

def load_text():
    """There is NO silent fallback. The script stops if data cannot be loaded."""
    from data import wikitext
    return wikitext()


def prepare_prompts(text, tok):
    """Cuts evenly spaced prompts from different parts of the text."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    step = len(ids) // (NUM_PROMPTS + 1)
    prompts = [ids[(i + 1) * step:(i + 1) * step + PROMPT_LEN]
               for i in range(NUM_PROMPTS)]
    print(f"prompts: {len(prompts)} x {PROMPT_LEN} tokens, "
          f"{GEN_LEN} tokens will be generated for each")
    return prompts


# =====================================================================
# MODEL LOADING / FREEING
# =====================================================================

def load(model_id, quant=None):
    if quant in ("4bit", "4bit_full"):
        from transformers import BitsAndBytesConfig
        extra = dict(llm_int8_skip_modules=[]) if quant == "4bit_full" else {}
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, **extra)
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map={"": 0})
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DTYPE if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)
    m.eval()
    if DEVICE == "cuda":
        print(f"  memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return m


def free_memory():
    """Must be called AFTER the names referring to the model have been deleted."""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# =====================================================================
# STAGE 1 - target model generates
# =====================================================================

@torch.no_grad()
def target_generate(model, prompts, eos_id):
    records = []
    for i, prompt in enumerate(prompts):
        x = prompt.unsqueeze(0).to(model.device)
        out = model.generate(
            x, max_new_tokens=GEN_LEN, min_new_tokens=GEN_LEN,
            do_sample=False, pad_token_id=eos_id,
            temperature=None, top_p=None, top_k=None)
        cont = out[0, len(prompt):]
        P = len(prompt)
        # read the same text in one pass: probability distribution at each position
        logit = model(out).logits[0, P - 1:P - 1 + len(cont)].float()
        probs = torch.softmax(logit, dim=-1)
        top = probs.topk(TOPK, dim=-1)
        consistency = (logit.argmax(-1) == cont).float().mean().item()
        records.append(dict(
            prompt=prompt.cpu(), cont=cont.cpu(),
            top_p=top.values.half().cpu(), top_i=top.indices.cpu(),
            consistency=consistency))
        print(f"\r  generating {i + 1}/{len(prompts)}", end="", flush=True)
    print()
    return records


# =====================================================================
# STAGE 2 - draft model compares
# =====================================================================

@torch.no_grad()
def draft_compare(model, records):
    for r in records:
        seq = torch.cat([r["prompt"], r["cont"]]).unsqueeze(0).to(model.device)
        P, L = len(r["prompt"]), len(r["cont"])
        q = torch.softmax(model(seq).logits[0, P - 1:P - 1 + L].float(), dim=-1)
        cont = r["cont"].to(q.device)
        r["match"] = (q.argmax(-1) == cont).cpu().numpy()
        top_p = r["top_p"].to(q.device).float()
        top_i = r["top_i"].to(q.device)
        q_top = torch.gather(q, 1, top_i)
        r["a_samp_pos"] = torch.minimum(top_p, q_top).sum(-1).cpu().numpy()
        r["tail"] = (1 - top_p.sum(-1)).mean().item()
    return records


# =====================================================================
# METRICS
# =====================================================================

def distinct_4gram(ids):
    """Fraction of distinct 4-token groups. 1 = no repetition at all."""
    ids = ids.tolist()
    groups = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(groups)) / max(1, len(groups))


def chain_simulation(match, k):
    """Replays chain speculative decoding round by round on the real accept/reject sequence.
    Returns: (tokens produced, number of rounds)."""
    j, tokens, rounds = 0, 0, 0
    L = len(match)
    while j < L:
        run = 0
        while run < k and j + run < L and match[j + run]:
            run += 1
        step = min(run + 1, L - j)   # accepted tokens + the target's own token
        tokens += step
        j += step
        rounds += 1
    return tokens, rounds


def E_formula(a, k):
    return k + 1 if a >= 1 else (1 - a ** (k + 1)) / (1 - a)


def best_k(E_func):
    ks = np.arange(1, 129)
    speed = np.array([E_func(k) / ((T_M + T_C2 * k * k) / 1000) for k in ks])
    return int(ks[speed.argmax()]), float(speed.max())


# =====================================================================
# FULL MEASUREMENT FOR ONE TARGET
# =====================================================================

def measure(target_id, name, quant, prompts, eos_id):
    print(f"\n--- {name} ---")
    print("STAGE 1: target model generating")
    target = load(target_id, quant)
    records = target_generate(target, prompts, eos_id)
    del target; free_memory()

    print("STAGE 2: draft model comparing")
    draft = load(DRAFT)
    records = draft_compare(draft, records)
    del draft; free_memory()

    rows = []
    for n, r in enumerate(records):
        s = dict(target=name, prompt_id=n,
                 a_greedy=float(r["match"].mean()),
                 a_sampling=float(r["a_samp_pos"].mean()),
                 tail=r["tail"], consistency=r["consistency"],
                 distinct_4gram=distinct_4gram(r["cont"]))
        for k in K_LIST:
            tk, rd = chain_simulation(r["match"], k)
            s[f"tokens_k{k}"], s[f"rounds_k{k}"] = tk, rd
        rows.append(s)
    return rows


def summary(df, name):
    d = df[df.target == name]
    clean = d[d.distinct_4gram >= 0.5]
    a = d.a_greedy.mean()
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"a (greedy)         : {a:.3f}   "
          f"[across prompts {d.a_greedy.min():.2f}–{d.a_greedy.max():.2f}]")
    if len(clean) < len(d):
        print(f"a (non-rep. {len(clean)}/{len(d)}) : {clean.a_greedy.mean():.3f}"
              f"   <- {len(d) - len(clean)} prompts contain repetition")
    print(f"a (sampling, T=1)  : {d.a_sampling.mean():.3f}   "
          f"(underestimated by at most {d.tail.mean():.3f})")
    if d.consistency.mean() < 0.97:
        print(f"  ! low consistency: {d.consistency.mean():.3f}")

    print(f"\n{'k':>5} {'E formula':>9} {'E simulation':>13} {'E sim. non-rep.':>17}")
    E_sim = {}
    for k in K_LIST:
        E_sim[k] = d[f"tokens_k{k}"].sum() / d[f"rounds_k{k}"].sum()
        e_c = clean[f"tokens_k{k}"].sum() / max(1, clean[f"rounds_k{k}"].sum())
        print(f"{k:>5} {E_formula(a, k):>9.2f} {E_sim[k]:>13.2f} {e_c:>17.2f}")

    k_m, s_m = best_k(lambda k: E_formula(a, k))
    k_a, s_a = best_k(lambda k: E_formula(0.70, k))
    print(f"\nfastest k (no sparsity, 0.5B timing model):")
    print(f"  assumed a=0.70  : k = {k_a:>3}  ({s_a:.1f} tokens/s)")
    print(f"  measured a={a:.2f} : k = {k_m:>3}  ({s_m:.1f} tokens/s)")
    return a, E_sim


def plot(df, results, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, (a, E_sim)) in enumerate(results.items()):
        ks = np.array(K_LIST)
        color = f"C{i}"
        ax.plot(ks, [E_sim[k] for k in ks], "o-", color=color,
                label=f"{name}: simulation")
        kk = np.arange(1, 129)
        ax.plot(kk, [E_formula(a, k) for k in kk], "--", color=color,
                label=f"{name}: formula (a={a:.2f})")
    kk = np.arange(1, 129)
    ax.plot(kk, [E_formula(0.70, k) for k in kk], ":", color="gray",
            label="previous assumption (a=0.70)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("chain length k")
    ax.set_ylabel("tokens gained per round E(k)")
    ax.set_title(f"Acceptance rate - draft: {DRAFT.split('/')[-1]}")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# =====================================================================
# MAIN
# =====================================================================

def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print(f"device: {torch.cuda.get_device_properties(0).name}"
          if DEVICE == "cuda" else "device: NONE (CPU) - will be very slow")
    print(f"draft : {DRAFT}{'   [FAST MODE]' if FAST else ''}\n")

    tok = AutoTokenizer.from_pretrained(DRAFT)
    prompts = prepare_prompts(load_text(), tok)

    # SELF-TEST: draft model against itself -> a should be ~1
    print("\nSELF-TEST (draft against itself)")
    st = measure(DRAFT, "self-test", None, prompts[:3], tok.eos_token_id)
    a_self = np.mean([s["a_greedy"] for s in st])
    print(f"  a = {a_self:.3f}")
    if a_self < 0.98:
        raise RuntimeError(
            f"Self-test failed (a={a_self:.3f}, expected ~1). The alignment code "
            "is broken; results are unreliable. Script stopped.")
    print("  passed.")

    targets = [t for t in TARGETS if not SELECTION or any(s in t[1] for s in SELECTION)]
    if not targets:
        raise SystemExit(f"No target named '{' '.join(SELECTION)}'. Options: "
                         + ", ".join(t[1] for t in TARGETS))
    if any((t[2] or "").startswith("4bit") for t in targets):
        missing = [m for m in ("accelerate", "bitsandbytes")
                   if __import__("importlib.util").util.find_spec(m) is None]
        if missing:
            print(f"\n! Missing libraries for 4-bit: {', '.join(missing)}\n"
                  f"  Install them with:  pip install {' '.join(missing)}\n"
                  "  4-bit targets will be skipped.")
            targets = [t for t in targets if not (t[2] or "").startswith("4bit")]

    rows, results = [], {}
    for tid, name, quant in targets:
        # does it use the same vocabulary?
        tt = AutoTokenizer.from_pretrained(tid)
        if tt.get_vocab() != tok.get_vocab():
            print(f"\n! {name} skipped: vocabulary differs from the draft.")
            continue
        try:
            rows += measure(tid, name, quant, prompts, tok.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            print(f"\n! {name} skipped: out of memory.")
            free_memory()
            continue
        except (ImportError, ValueError) as e:
            print(f"\n! {name} skipped: {str(e)[:200]}")
            free_memory()
            continue
        df = pd.DataFrame(rows)
        df.to_csv(f"{CSV_DIR}/acceptance_{TIMESTAMP}.csv", index=False)
        results[name] = summary(df, name)

    if not results:
        print("\nNo measurements could be made.")
        return
    plot(pd.DataFrame(rows), results, f"{FIGURE_DIR}/acceptance_{TIMESTAMP}.png")
    print(f"\nCSV    : {CSV_DIR}/acceptance_{TIMESTAMP}.csv")
    print(f"FIGURE : {FIGURE_DIR}/acceptance_{TIMESTAMP}.png")


if __name__ == "__main__":
    main()
