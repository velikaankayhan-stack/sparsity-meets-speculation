"""
02_tree_size.py  -  Free verification budget
================================================================
micro-scale project

QUESTION
--------
In speculative decoding the large model checks the N positions proposed by
the draft in a single pass. SpecExec assumes this is FREE: the weights are
read once no matter how many positions you check.

That assumption holds up to a point. Beyond it, compute starts to dominate
and the time grows with N.

Where is that point? We call it the "free budget".

A weak GPU has a small budget, a strong one a large budget. SpecExec uses
2048 - but they measured on a 4090 and an A100. Does 2048 make sense on a
6 GB card?

THIS MEASUREMENT IS HARDWARE-SPECIFIC - run it on your own GPU, not on Colab.

INPUTS
------
Models in MODELS (downloaded from the Hugging Face hub). Inputs are random
token ids, so no dataset is needed. Needs a CUDA GPU; a CPU run is possible
but its results are ignored by paths.tree_csv().

OUTPUTS
-------
results/csv/tree_<device>_<ts>.csv
results/figures/tree_<device>_<ts>.png

RUN
---
python 02_tree_size.py   (from the repo root)
"""

import os, gc, time, datetime
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
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    # If it does not fit on the GPU, the script skips it on its own.
    # ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
]

N_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

WARMUP   = 3       # untimed warm-up runs
REPEATS  = 7       # timed repeats; the median is reported
DTYPE    = torch.float16

CSV_DIR    = "results/csv"
FIGURE_DIR = "results/figures"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")


# =====================================================================
# DEVICE INFO
# =====================================================================

def device_info():
    if not torch.cuda.is_available():
        return dict(name="CPU", vram_gb=0.0, bandwidth=0.0, pcie_gen=0, pcie_x=0)

    props = torch.cuda.get_device_properties(0)
    name = props.name
    vram = props.total_memory / 1e9

    try:
        gen = torch.cuda.get_device_properties(0).__dict__.get("pci_bus_id", None)
    except Exception:
        gen = None

    print(f"GPU       : {name}")
    print(f"VRAM      : {vram:.1f} GB")
    print(f"SM count  : {props.multi_processor_count}")
    print(f"torch     : {torch.__version__}  CUDA {torch.version.cuda}")
    print()
    print("For PCIe info, run this in a terminal and paste the output into log.txt:")
    print("  nvidia-smi --query-gpu=name,memory.total,"
          "pcie.link.width.current,pcie.link.gen.current --format=csv")
    print()

    return dict(name=name, vram_gb=vram, sm=props.multi_processor_count)


# =====================================================================
# MEASUREMENT
# =====================================================================

def time_forward(model, n, device, vocab_size):
    """Median time (ms) of a single forward pass over N positions."""
    inputs = torch.randint(0, vocab_size, (1, n), device=device)

    with torch.no_grad():
        for _ in range(WARMUP):
            model(inputs)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(REPEATS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times)), float(np.std(times))


def measure_model(model_id, model_name, gpu, rows):
    print(f"\n{'='*62}\n{model_name}\n{'='*62}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok   = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE if device == "cuda" else torch.float32
    ).to(device).eval()

    n_param = sum(p.numel() for p in model.parameters())
    vocab   = model.config.vocab_size
    print(f"  {n_param/1e9:.2f} billion parameters, ~{n_param*2/1e9:.2f} GB (fp16)")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for n in N_LIST:
        try:
            ms, std = time_forward(model, n, device, vocab)
        except torch.cuda.OutOfMemoryError:
            print(f"  N={n:<5} out of memory, stopping")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  N={n:<5} error: {type(e).__name__}: {e}")
            break

        peak = torch.cuda.max_memory_allocated()/1e9 if device == "cuda" else 0.0

        rows.append(dict(
            device     = gpu["name"],
            model      = model_name,
            n_param    = n_param,
            N          = n,
            ms         = ms,
            std_ms     = std,
            ms_per_position = ms / n,
            peak_gb    = peak,
        ))
        print(f"  N={n:<5} {ms:8.2f} ms   (±{std:.2f})   "
              f"{ms/n:7.3f} ms per position")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# =====================================================================
# FREE BUDGET
# =====================================================================

def free_budget(sub, threshold=1.25):
    """
    Largest N whose time is <= threshold times the N=1 time.
    In other words, the verification width that is still "almost free".
    """
    base = sub[sub.N == 1].ms.iloc[0]
    ok = sub[sub.ms <= base * threshold]
    return int(ok.N.max()) if len(ok) else 1


# =====================================================================
# PLOT
# =====================================================================

def plot(df, gpu_name, path):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
    colors = plt.cm.tab10(np.linspace(0, .5, df.model.nunique()))

    for color, m in zip(colors, df.model.unique()):
        sub = df[df.model == m].sort_values("N")
        base = sub[sub.N == 1].ms.iloc[0]
        b = free_budget(sub)

        # 1) Total time
        ax[0].plot(sub.N, sub.ms, "o-", lw=2, ms=5, color=color, label=m)
        ax[0].axhline(base, color=color, ls=":", lw=1, alpha=.6)

        # 2) Time per position - the plot that tells the story
        ax[1].plot(sub.N, sub.ms_per_position, "o-", lw=2, ms=5,
                   color=color, label=f"{m}  (free budget ≈ {b})")
        ax[1].axvline(b, color=color, ls="--", lw=1.2, alpha=.7)

        # 3) Verification throughput
        ax[2].plot(sub.N, sub.N / (sub.ms / 1000), "o-", lw=2, ms=5,
                   color=color, label=m)

    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("positions verified per round  (N)")
    ax[0].set_ylabel("time per round (ms)")
    ax[0].set_title("1 - Round time\nhorizontal dotted line = N=1 level")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)

    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("positions verified per round  (N)")
    ax[1].set_ylabel("ms per position")
    ax[1].set_title("2 - Cost per position\n"
                    "straight descent = free region, past the minimum = compute-bound")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)

    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("N"); ax[2].set_ylabel("positions verified per second")
    ax[2].set_title("3 - Verification throughput\npeak = most efficient N")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)

    fig.suptitle(f"Free verification budget - {gpu_name}", y=1.02, fontsize=12)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# =====================================================================

def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    gpu = device_info()
    gpu_short = gpu["name"].replace(" ", "_").replace("NVIDIA_", "")

    rows = []
    for mid, mname in MODELS:
        try:
            measure_model(mid, mname, gpu, rows)
        except Exception as e:
            print(f"  ! {mname} skipped: {type(e).__name__}: {e}")

        if rows:
            pd.DataFrame(rows).to_csv(
                f"{CSV_DIR}/tree_{gpu_short}_{TIMESTAMP}.csv", index=False)

    if not rows:
        print("\nNo measurement could be made.")
        return

    df = pd.DataFrame(rows)
    plot(df, gpu["name"], f"{FIGURE_DIR}/tree_{gpu_short}_{TIMESTAMP}.png")

    print("\n" + "="*62)
    print("SUMMARY")
    print("="*62)
    print(f"{'model':<16}{'free budget':>14}{'best N':>14}{'throughput':>12}")
    print("-"*62)
    for m in df.model.unique():
        sub = df[df.model == m].sort_values("N")
        b = free_budget(sub)
        cap = sub.N / (sub.ms/1000)
        best = int(sub.N.iloc[int(np.argmax(cap.values))])
        print(f"{m:<16}{b:>14}{best:>14}{cap.max():>12.0f} pos/s")

    print("\nINTERPRETATION:")
    print("  free budget = number of positions you can verify ~for free in one round.")
    print("  SpecExec uses 128-2048 (RTX 4090 / A100).")
    print("  If your budget is far below that, the large-tree strategy")
    print("  does not work on this GPU - and that is a finding in itself.")
    print(f"\nCSV    : {CSV_DIR}/tree_{gpu_short}_{TIMESTAMP}.csv")
    print(f"FIGURE : {FIGURE_DIR}/tree_{gpu_short}_{TIMESTAMP}.png")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\ntotal time: {time.time()-t0:.0f} s")
