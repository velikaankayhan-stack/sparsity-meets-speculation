# micro-scale-llm

This repo contains the code and measurements for my project on running large language models on a small GPU. The question I wanted to answer: **can activation sparsity and speculative decoding be used together on a consumer GPU, or do they cancel each other out?**

All measurements were done on my laptop (NVIDIA RTX 3050 Laptop GPU, 6 GB). The 14B model did not fit, so it was measured on a Tesla T4 in Google Colab. The models are from the Qwen2.5 family (0.5B to 14B). The draft model for speculative decoding is always Qwen2.5-0.5B.

This is the first part of a series. Later projects will go into separate folders.

## Short summary of the results

- **Union.** When several tokens are verified in the same round, the set of neurons they need grows quickly. I modeled this with an empirical formula, `U = 1 - (1 - p)^(b * k^alpha)`, where alpha is about 0.73-0.82 for all models.
- **Free draft.** If the draft model is assumed to cost nothing, the best draft length is long (10-48 tokens). At that length almost every neuron is needed, so sparsity gives no extra speed (about 1.00x).
- **Real draft cost.** On my hardware the draft costs 32-93% of a target round, because both models are small and the fixed per-layer overhead of the standard Hugging Face code dominates. The best draft length drops to 1-3 tokens. Speculative decoding helps only when c < a (c = draft cost, a = acceptance rate): 1.10-1.26x for 1.5B 4-bit, 3B and 14B, slower for 1.5B fp16 and 7B.
- **Sparsity, computed vs. measured.** The calculation says 1.60x for sparsity alone (1.5B fp16). With a real sparse MLP the round got only about 7% shorter, which is inside the measurement noise. Most of the round is fixed software overhead, not weight reading.

So with standard software, the theoretical gain from sparsity does not show up on this kind of hardware. The full discussion is in the report.

## Folders

```
*.py              scripts (table below)
data.py           loads WikiText-2 (test split); used by every measurement script
paths.py          picks the right file from results/csv
results/csv/      raw measurements
results/figures/  figures used in the report
```

## Scripts

| Script | What it does | GPU | Output |
|---|---|---|---|
| 01_union.py | fraction of neurons needed when N tokens are processed together (N <= 8) | recommended, also runs on CPU | `union_*.csv` |
| 02_tree_size.py | first round-time measurement T(N) for N up to 2048, GPU and CPU (no KV cache) | yes | `tree_*.csv` |
| 03_union_large_n.py | union for large N (up to 256) | yes | `union_large_n_*.csv` |
| 04_quality.py | perplexity loss vs. fraction of active neurons ("quality wall") | yes | `quality_*.csv` |
| 05_plot_union.py | union figures | no | `fig01_*.png` |
| 07_acceptance.py | acceptance rate of the draft model and E(k) simulation | yes | `acceptance_*.csv` |
| 08_decision_surface.py | first speed surface for 0.5B (draft assumed free) | no | `fig05_*.png` |
| 09_multi_model.py | union, quality and acceptance for 1.5B, 3B, 14B | yes | `mm_*.csv` |
| 10_speed_model.py | speed calculation (with and without draft cost), compared with the real measurements | no | `speed_model.csv` |
| 11_model_comparison.py | alpha, quality wall and acceptance rate vs. model size | no | `model_comparison.csv`, `fig07_*.png` |
| 12_real_speed.py | end-to-end speed; sparse MLP round times for the fp16 model | yes | `speed_*.csv`, `speed_summary.csv` |
| 13_timing.py | target round time and draft time, measured in the same session with KV cache | yes | `timing_*.csv` |
| 14_draft_sparsity.py | small test: what happens if sparsity is applied to the draft model instead | yes | `draft_sparsity_*.csv` |
| 15_report_figures.py | the remaining report figures | no | `fig02/03/04/05/06/09*.png` |
| 16_hot_neuron_control.py | checks how much of the union effect comes from "hot" neurons and how much from neighbouring tokens | no (a few minutes on CPU) | `hot_neuron_control_*.csv` |

There is no script 06. It was merged into another script during the project and I kept the numbers so they match the report.

## Reproducing the figures and numbers (no GPU needed)

```
pip install -r requirements.txt
python 05_plot_union.py
python 08_decision_surface.py
python 10_speed_model.py
python 11_model_comparison.py
python 15_report_figures.py
```

Run them from the repo folder, the paths are relative. These five scripts only read `results/csv`, and their output matches the numbers and figures in the report.

The measurement scripts need a GPU and downloaded models, for example:

```
python 13_timing.py --models Qwen/Qwen2.5-1.5B:fp16
python 12_real_speed.py --model Qwen/Qwen2.5-1.5B --quant fp16
python 09_multi_model.py --model Qwen/Qwen2.5-3B --quant 4bit
```

Scripts 07, 09, 12 and 13 have a `--quick` option to check that the code runs before starting a long measurement.

## Which file is behind which figure

| Report section | Figure | Script | Data |
|---|---|---|---|
| Union | fig01_panel1, fig01_panel2 | 05 | union_20260922_2332 |
| Tree size | fig02_verify_cost | 15 | timing_* (TIMING_FILES in 10) |
| Tree size | fig03a_cpu_gpu, fig03b_specexec_trees | 15 | tree_CPU_20260919_0257, tree_..._20260919_1941 |
| Acceptance | fig04_acceptance | 15 | acceptance_20261004_2312 |
| Surface | fig05_sparsity_vs_N | 15 | union_large_n_20260923_0003 |
| Surface | fig05_panel1 (quality wall), fig05_panel2 (surface) | 08 | union, tree_..._1941, quality, acceptance |
| Multi-model check | fig07_panelA-D | 11 | mm_*, b7_*, quality, acceptance |
| Draft cost | fig06_draft_cost | 15 (using 10) | timing_*, mm_*, b7_*, acceptance |
| Real speed | fig09a_speculative, fig09b_four_configs | 15 | speed_model, speed_summary |

## Notes on the data

- **7B data.** `b7_union_*` and `b7_quality_*` were made with an older version of 09 on Colab. The 7B acceptance rate comes from 07 (`acceptance_20261005_0100`).
- **Round times.** 10 reads the `timing_*` files by name (`TIMING_FILES`), so running 13 again does not silently change the results. For 1.5B 4-bit and 7B, 13 gave its own drift warning (first and last N=1 time differ by more than 10%). Most of this comes from warm-up in the first measurement, and the round-time curve is fitted with isotonic regression, but the c values of these two models are less certain.
- **Two kinds of "real" speed (fp16 1.5B).** *End to end*: the whole generation was timed (baseline and speculative only). *From measured round times*: the round times measured with the sparse MLP were combined with the measured acceptance data and draft time (speculative 0.89x, sparsity 1.07x, both 0.88x). The sparse configurations could not be timed end to end because there is no real predictor. Both are in `speed_model.csv` (`r_spec`, and `r_spec_timing`, `r_sparse`, `r_combined`).
- **Gather mode.** If the selected neuron rows are gathered from random places in memory (closer to a real kernel), sparsity made the round slower (0.79x). This is in `speed_summary.csv` (`gather_*`).
- **14B real speed.** Only the summary row in `speed_summary.csv` was saved from the Colab run; the per-prompt file is missing.
- **Oracle sparsity.** Which neurons are active was decided after computing all of them. No real predictor was built, so the sparsity numbers are an upper limit.
- **File choice.** When there are several runs, scripts pick the newest by the timestamp in the file name (`_YYYYMMDD_HHMM`), not by file modification time.

## Environment

Windows 11, Python 3.14, PyTorch 2.11 (CUDA 12.8), Transformers 5.17, bitsandbytes 0.50 (see `requirements.txt`). Dataset: WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`, test split).

## Acknowledgements

I used an AI assistant (Claude) for help with writing code, checking calculations and translating. The research question, the measurements on my hardware and the interpretation of the results are my own work.
