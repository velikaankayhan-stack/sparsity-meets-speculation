# -*- coding: utf-8 -*-
"""
14_draft_sparsity.py - How does applying sparsity to the DRAFT model change c and a?

Additional experiment (10 October 2026).

Idea: in the target model, sparsity melts away during the verification round
because of the union effect. The draft, however, always generates ONE token at
a time (N = 1), so there is no union problem. A draft error also does not
corrupt the output (the target verifies); its only cost is a lower acceptance
rate a. Question: as the active neuron fraction p decreases,
   how much does a(p) drop?   (hardware independent)
   how much does c(p) drop?   (draft time / target round time, on this card)
and how does the rough k = 1 speed (1 + a) / (c + T(2)/T(1)) change?

Method:
  a(p): The target (Qwen2.5-1.5B, 16-bit) generates 128 tokens (greedy) as the
        continuation of 40 WikiText-2 passages. The draft (Qwen2.5-0.5B,
        16-bit) reads the same text; in every MLP, before down_proj, the top p
        fraction of intermediate activations by absolute value is kept (same
        sparsity as 04/09). A token is accepted if the draft's most likely
        token equals the target's. Repetitive texts (distinct-4-gram < 0.5)
        are separated.
  c(p): The draft MLP is modified to read only the weights of the first
        n = p·d neurons ("ideal" mode from 12: contiguous rows, perfect
        predictor), and the single-token time is measured with a 128-token
        cache. The target's T(1) and T(2) are measured in the same session.

Inputs : WikiText-2 test text via data.wikitext()
Outputs: results/csv/draft_sparsity_<device>_<ts>.csv
Run    : python 14_draft_sparsity.py   (GPU required; ~10 min on a T4)
"""
import datetime, gc, os, time, types
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET, DRAFT = "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B"
P_LIST = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2]
NUM_PROMPTS, PROMPT_LEN, GEN_LEN, CONTEXT = 40, 128, 128, 128
WARMUP, REPEATS = 5, 40
DEVICE = "cuda"
GPU_NAME = torch.cuda.get_device_name(0)
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")


def sync(): torch.cuda.synchronize()


def load(m):
    return AutoModelForCausalLM.from_pretrained(m, dtype=torch.float16).to(DEVICE).eval()


def text():
    from data import wikitext
    return wikitext()


def distinct_4gram(ids):
    ids = ids.tolist(); g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / max(1, len(g))


class Mask:                       # for a(p): activation masking (same as 04/09)
    def __init__(self, model):
        self.p, self.k = 1.0, []
        for layer in model.model.layers:
            self.k.append(layer.mlp.down_proj.register_forward_pre_hook(self._f))

    def _f(self, mod, inputs):
        x = inputs[0]; d = x.shape[-1]; keep = max(1, int(round(d * self.p)))
        if keep >= d:
            return None
        idx = x.abs().topk(keep, dim=-1, sorted=False).indices
        m = torch.zeros_like(x, dtype=torch.bool); m.scatter_(-1, idx, True)
        return (x * m,)

    def detach(self):
        for h in self.k: h.remove()


class Sparse:                     # for c(p): an MLP that really reads fewer weights
    p = 1.0


def sparse_mlp(model):
    for layer in model.model.layers:
        m = layer.mlp
        m._WdT = m.down_proj.weight.t().contiguous(); m._d = m.gate_proj.weight.shape[0]

        def forward(self, x):
            if Sparse.p >= 1.0:
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            n = max(1, int(round(self._d * Sparse.p)))
            h = self.act_fn(torch.nn.functional.linear(x, self.gate_proj.weight[:n])) * \
                torch.nn.functional.linear(x, self.up_proj.weight[:n])
            return h @ self._WdT[:n]
        m.forward = types.MethodType(forward, m)


@torch.no_grad()
def round_time(model, N):
    g = torch.Generator().manual_seed(0)
    ctx = torch.randint(10, 1000, (1, CONTEXT), generator=g).to(DEVICE)
    kv = model(ctx, use_cache=True).past_key_values
    x = torch.randint(10, 1000, (1, N), device=DEVICE)
    s = []
    for i in range(WARMUP + REPEATS):
        sync(); t0 = time.perf_counter(); model(x, past_key_values=kv, use_cache=True); sync()
        if i >= WARMUP: s.append((time.perf_counter() - t0) * 1000)
        kv.crop(CONTEXT)
    return float(np.median(s))


def main():
    os.makedirs("results/csv", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(TARGET)
    ids = tok(text(), return_tensors="pt").input_ids[0]
    step = len(ids) // (NUM_PROMPTS + 1)
    prompts = [ids[(i + 1) * step:(i + 1) * step + PROMPT_LEN] for i in range(NUM_PROMPTS)]

    print(f"device: {GPU_NAME}\n[1] target model generating ({NUM_PROMPTS} × {GEN_LEN})")
    h = load(TARGET)
    T1, T2 = round_time(h, 1), round_time(h, 2)
    continuations = []
    with torch.no_grad():
        for i, pr in enumerate(prompts):
            c = h.generate(pr.unsqueeze(0).to(DEVICE), max_new_tokens=GEN_LEN, min_new_tokens=GEN_LEN,
                           do_sample=False, pad_token_id=tok.eos_token_id,
                           temperature=None, top_p=None, top_k=None)
            continuations.append(c[0, len(pr):].cpu())
            print(f"\r  {i + 1}/{NUM_PROMPTS}", end="", flush=True)
    print(f"\n  target model T(1) = {T1:.1f} ms, T(2) = {T2:.1f} ms")
    del h; gc.collect(); torch.cuda.empty_cache()

    t = load(DRAFT)
    nonrep = [distinct_4gram(cv) >= 0.5 for cv in continuations]
    print(f"[2] draft: acceptance rate a(p)  (non-repetitive texts: {sum(nonrep)}/{NUM_PROMPTS})")
    mk = Mask(t); result = {}
    with torch.no_grad():
        for p in P_LIST:
            mk.p = p; matches_all, matches_nr = [], []
            for pr, cv, ok in zip(prompts, continuations, nonrep):
                seq = torch.cat([pr, cv]).unsqueeze(0).to(DEVICE); P = len(pr)
                guess = t(seq).logits[0, P - 1:P - 1 + len(cv)].argmax(-1).cpu()
                e = (guess == cv).float().mean().item(); matches_all.append(e)
                if ok: matches_nr.append(e)
            result[p] = dict(a_all=float(np.mean(matches_all)), a=float(np.mean(matches_nr)))
            print(f"  active {p * 100:3.0f}%:  a = {result[p]['a']:.3f}  (all texts {result[p]['a_all']:.3f})")
    mk.detach()

    print("[3] draft: single-token time with a real sparse MLP")
    sparse_mlp(t); rows = []
    for p in P_LIST:
        Sparse.p = p; td = round_time(t, 1)
        c = td / T1; a = result[p]["a"]
        speed = (1 + a) / (c + T2 / T1)
        rows.append(dict(device=GPU_NAME, target=TARGET, draft=DRAFT, active_frac=p, a=a, a_all=result[p]["a_all"],
                         draft_ms=td, T1_ms=T1, T2_ms=T2, c=c, c_lt_a=c < a, speed_k1=speed))
        print(f"  active {p * 100:3.0f}%:  draft {td:5.1f} ms   c = {c:.3f}   a = {a:.3f}   "
              f"{'c<a' if c < a else 'c>a'}   k=1 rough speed {speed:.3f}×")
    df = pd.DataFrame(rows)
    path = f"results/csv/draft_sparsity_{GPU_NAME.replace(' ', '_')}_{TIMESTAMP}.csv"
    df.to_csv(path, index=False); print("CSV:", path)


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"total {(time.time() - t0) / 60:.1f} min")
