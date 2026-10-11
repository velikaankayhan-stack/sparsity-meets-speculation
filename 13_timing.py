# -*- coding: utf-8 -*-
"""
13_timing.py - re-measure verification round times in a single session.

Why
  The speed model in 10_speed_model.py relies on two times:
    T(N)     : one verification round of the target model over N positions
    t_draft  : time for the draft (0.5B) to produce one token
  Previously these came from DIFFERENT sessions (draft on 4 Oct, targets on
  5 Oct). The 5 Oct times were ~2x slower (power mode), which made the draft
  cost c = t_draft / T(1) look smaller than it is.
  This script measures T(N) for each target and the time of the draft that
  fits in memory ALONGSIDE THAT TARGET in the SAME session, with the SAME method.

Method
  * KV-cached measurement as in real generation: first a CONTEXT=128 token
    context is processed, then a round of N new positions is timed and the
    cache is cropped back to 128.
  * For each N: warm-up + repeats, median time.
  * The target is measured alone, the draft right after it (same session).
    With both in memory, the 7B on a 6 GB card spilled into system RAM and
    times multiplied (10 Oct 01:35 run). Whether the draft is fp16 or 4-bit
    is chosen by whether it would fit next to the target in real use.
  * Memory spill check: if card memory fills up, that N and larger are skipped.
  * Stability check: T(1) and the draft time are measured twice; a warning is
    printed if they differ by more than 10% (re-measure that model).

Run (from repo root; laptop on charger, "Best performance", other apps closed)
  RTX 3050 :  python 13_timing.py                          (~15-20 min, 4 models)
  Colab T4 :  python 13_timing.py --models Qwen/Qwen2.5-14B:4bit
  Smoke    :  python 13_timing.py --quick

Outputs
  results/csv/timing_<model>_<quant>_<ts>.csv   (N, ms, std_ms, draft_ms, ...)
  results/csv/timing_summary.csv                (one row appended per model)
"""
import argparse
import datetime
import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

DEFAULT_MODELS = ["Qwen/Qwen2.5-1.5B:fp16", "Qwen/Qwen2.5-1.5B:4bit",
                  "Qwen/Qwen2.5-3B:4bit", "Qwen/Qwen2.5-7B:4bit"]

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="model:quant ...")
ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--quick", action="store_true", help="code test: fewer N, fewer repeats")
ap.add_argument("--cpu", action="store_true", help="for code testing only")
ARGS = ap.parse_args()

DEVICE = "cpu" if (ARGS.cpu or not torch.cuda.is_available()) else "cuda"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
CSV_DIR = "results/csv"
CONTEXT = 128
N_LIST = ([1, 2, 3, 5, 9] if ARGS.quick else
          [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17, 22, 25, 33, 49, 65, 97, 129])
WARMUP, REPEAT = (1, 3) if ARGS.quick else (3, 15)
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def free_memory():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


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


class Timer:
    """Measures a model's KV-cached round time."""

    def __init__(self, model):
        self.m = model
        self.vocab = min(1000, model.config.get_text_config().vocab_size)
        g = torch.Generator().manual_seed(0)
        ctx = torch.randint(10, self.vocab, (1, CONTEXT), generator=g).to(model.device)
        with torch.no_grad():
            self.kv = model(ctx, use_cache=True).past_key_values
        self.croppable = hasattr(self.kv, "crop")
        self.ctx = ctx

    @torch.no_grad()
    def _round(self, x):
        if self.croppable:
            sync(); t0 = time.perf_counter()
            self.m(x, past_key_values=self.kv, use_cache=True)
            sync(); t = time.perf_counter() - t0
            self.kv.crop(CONTEXT)
        else:                                   # older version: rebuild the context
            kv = self.m(self.ctx, use_cache=True).past_key_values
            sync(); t0 = time.perf_counter()
            self.m(x, past_key_values=kv, use_cache=True)
            sync(); t = time.perf_counter() - t0
        return t * 1000

    def measure(self, N, repeat=REPEAT):
        x = torch.randint(10, self.vocab, (1, N), device=self.m.device)
        for _ in range(WARMUP):
            self._round(x)
        s = [self._round(x) for _ in range(repeat)]
        return float(np.median(s)), float(np.std(s))


def draft_quant_for(target):
    """In real use the draft must fit NEXT TO the target. An fp16 0.5B is ~1 GB;
    ~0.6 GB is kept free for generation. If it does not fit, use a 4-bit draft
    (same rule as in 12)."""
    if DEVICE != "cuda":
        return "fp16"
    total = torch.cuda.mem_get_info()[1] / 2**30
    used = torch.cuda.memory_allocated() / 2**30
    return "fp16" if used + 1.0 + 0.6 < total else "4bit"


def memory_spill():
    """When GPU memory fills up, the Windows driver spills into system RAM and
    times multiply. This is detected via the fraction of reserved memory."""
    if DEVICE != "cuda":
        return False
    return torch.cuda.memory_reserved() / torch.cuda.mem_get_info()[1] > 0.93


def one_model(model_id, quant):
    short = model_id.split("/")[-1]
    print(f"\n=== {short} ({quant}) ===")
    # 1) Target measured ALONE (with the draft in memory the 7B spills on a 6 GB card)
    target = load(model_id, quant)
    dq = draft_quant_for(target)
    tt = Timer(target)
    if DEVICE == "cuda":
        print(f"  memory {torch.cuda.memory_allocated() / 1e9:.2f} GB "
              f"(in real use the draft would be {dq})")
    rows = []
    for N in N_LIST:
        try:
            ms, sd = tt.measure(N, REPEAT if N <= 33 else max(3, REPEAT // 2))
        except torch.cuda.OutOfMemoryError:
            print(f"  N={N}: out of memory, stopping"); free_memory(); break
        if memory_spill():
            print(f"  N={N}: memory spill (using system RAM) - "
                  f"this and larger N skipped"); break
        rows.append(dict(model=short, quant=quant, device=DEVICE_NAME, N=N, ms=ms, std_ms=sd))
        print(f"  N={N:<4} {ms:7.1f} ms  (+/-{sd:.1f})")
    T1_last = tt.measure(1)[0]
    del tt, target
    free_memory()

    # 2) Draft measured right after, in the same session
    draft = load(ARGS.draft, dq)
    td_timer = Timer(draft)
    td_first = td_timer.measure(1, REPEAT * 2)[0]
    td_last = td_timer.measure(1, REPEAT * 2)[0]
    del td_timer, draft
    free_memory()

    T1 = rows[0]["ms"]
    td = (td_first + td_last) / 2
    for r in rows:
        r.update(draft_quant=dq, draft_ms=td, context=CONTEXT, timestamp=TIMESTAMP)
    pd.DataFrame(rows).to_csv(f"{CSV_DIR}/timing_{short}_{quant}_{TIMESTAMP}.csv", index=False)

    drift = max(abs(T1_last / T1 - 1), abs(td_last / td_first - 1))
    print(f"  draft {dq}: {td_first:.1f} / {td_last:.1f} ms   T(1): {T1:.1f} -> {T1_last:.1f} ms")
    print(f"  c = draft / T(1) = {td / T1:.2f}")
    if drift > 0.10:
        print(f"  !!! WARNING: times drifted {drift * 100:.0f}% during measurement - "
              f"re-measure this model: --models {model_id}:{quant}")
    summary = dict(timestamp=TIMESTAMP, model=short, quant=quant, device=DEVICE_NAME, T1_ms=T1,
                   T1_last_ms=T1_last, draft_quant=dq, draft_ms=td, c=td / T1,
                   drift=drift)
    path = f"{CSV_DIR}/timing_summary.csv"
    pd.DataFrame([summary]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return summary


def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    print(f"device: {DEVICE_NAME}   context {CONTEXT}   N: {N_LIST}")
    summaries = []
    for m in ARGS.models:
        model_id, quant = m.split(":")
        summaries.append(one_model(model_id, quant))
    print("\n" + "=" * 60 + "\nSUMMARY (same session)\n" + "=" * 60)
    for o in summaries:
        print(f"  {o['model']:<14} {o['quant']:<5} T(1) {o['T1_ms']:6.1f} ms   "
              f"draft({o['draft_quant']}) {o['draft_ms']:5.1f} ms   c = {o['c']:.2f}"
              + ("   !! unstable" if o["drift"] > 0.10 else ""))
    print(f"\nCSV: {CSV_DIR}/timing_*_{TIMESTAMP}.csv  (+ {CSV_DIR}/timing_summary.csv)")
    print("Next:  python 10_speed_model.py")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
