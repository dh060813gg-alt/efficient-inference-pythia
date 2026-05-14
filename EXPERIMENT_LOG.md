# Experiment Log

This file records executed tests, environment information, observed issues, and current smoke-test results.

## Environment

- **OS**: Windows
- **Python environment**: user local Python
- **Torch**: `2.11.0+cu128`
- **CUDA available**: `True`
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Transformers**: `5.3.0`
- **Datasets**: `4.8.5`
- **Accelerate**: `1.13.0`
- **Model**: `EleutherAI/pythia-70m`

## Fixed Issues

- **Transformers 5 DynamicCache compatibility**
  - Initial code assumed legacy tuple-style `past_key_values`.
  - Transformers 5 returns `transformers.cache_utils.DynamicCache`.
  - Fixed cache size/counting and compression code to handle `DynamicCache.layers[*].keys/values`.

- **Hybrid attention fallback**
  - Default SDPA attention does not return attention weights with `output_attentions=True`.
  - Hybrid method now uses key-norm salience by default, avoiding forced eager attention.

## Smoke Test Commands

```powershell
python scripts/run_ppl.py --dataset builtin --max-samples 3 --max-length 128 --stride 64 --device cpu --output results/ppl_builtin.json
python scripts/run_speed.py --method dense --max-new-tokens 8 --device cpu --output results/speed_dense.json
python scripts/run_speed.py --method window --cache-budget 8 --recent 4 --max-new-tokens 8 --device cpu --output results/speed_window.json
python scripts/run_speed.py --method hybrid --cache-budget 8 --recent 4 --max-new-tokens 8 --device cpu --output results/speed_hybrid.json
```

## Smoke Test Results

### Perplexity

| Dataset | Samples | Tokens | PPL | Mean NLL |
| --- | ---: | ---: | ---: | ---: |
| builtin | 3 | 104 | 365.5567 | 5.9014 |

### Speed and Cache

| Method | Generated tokens | TTFT (s) | TPOT (s) | Throughput (tok/s) | Total time (s) | Final cache length | Final cache elements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 8 | 0.0371 | 0.0073 | 89.8464 | 0.0890 | 18 | 110592 |
| window | 8 | 0.0409 | 0.0072 | 84.8002 | 0.0943 | 8 | 49152 |
| hybrid | 8 | 0.0378 | 0.0068 | 85.5623 | 0.0935 | 8 | 49152 |

## Interpretation

- The smoke test confirms that the full pipeline runs on CPU with Pythia-70M.
- Window and hybrid compression reduce final KV cache elements from `110592` to `49152`, a reduction of approximately `55.56%` in this small setting.
- The short CPU smoke test is too small for reliable speed conclusions. It is mainly used to validate correctness and data logging.
- For final reporting, run larger Wikitext and PG-19 experiments, preferably on GPU.

## Added Metrics

The speed script now also records:

- **dense_cache_length_estimate**: estimated final dense cache length for the same prompt and generation length.
- **kv_cache_reduction_ratio**: relative cache-length reduction compared with dense decoding.
- **avg_cache_length**: average retained cache length during generation.
- **attention_kv_accesses_estimate**: cache-length sum multiplied by the number of transformer layers, used as a lightweight proxy for attention-side KV reads.
- **device**: execution device used by the speed benchmark.
- **cuda_peak_memory_mb**: peak allocated GPU memory during generation, recorded when CUDA is used.

## Next Long Experiments

Run these commands when convenient and send back the generated JSON files or the output of `summarize_results.py`:

```powershell
python scripts/run_ppl.py --dataset wikitext --split test --max-samples 16 --max-length 256 --stride 128 --device cpu --output results/ppl_wikitext_16.json
python scripts/run_ppl.py --dataset pg19 --split test --max-samples 1 --max-length 512 --stride 256 --device cpu --output results/ppl_pg19_1.json
python scripts/run_speed.py --method dense --max-new-tokens 64 --device cpu --output results/speed_dense_64.json
python scripts/run_speed.py --method window --cache-budget 32 --recent 16 --max-new-tokens 64 --device cpu --output results/speed_window_b32_64.json
python scripts/run_speed.py --method hybrid --cache-budget 32 --recent 16 --max-new-tokens 64 --device cpu --output results/speed_hybrid_b32_64.json
python scripts/summarize_results.py --results-dir results
```

## Long Experiment Results (2026-05-12)

### Wikitext PPL

| Dataset | Samples | Tokens | PPL | Mean NLL |
| --- | ---: | ---: | ---: | ---: |
| wikitext (test) | 16 | 1376 | 93.3659 | 4.5365 |

### Speed (64 tokens, budget 32, recent 16)

| Method | TTFT | TPOT | Throughput | Cache len | KV reduction | KV accesses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.0303 | 0.0065 | 143.5980 | 74 | 0.00% | 16320 |
| window | 0.0306 | 0.0060 | 149.5136 | 32 | 56.76% | 10902 |
| hybrid | 0.0288 | 0.0048 | 174.3158 | 32 | 56.76% | 10902 |

### Key Findings

- Both compressed methods reduce KV cache length by **56.76%** and KV accesses by **33.2%**.
- Hybrid achieves the best throughput (174.32 tok/s) and lowest TPOT (0.0048 s).
- PG-19 PPL: **43.6556** over 10219 tokens (built-in long-text substitute, 2026-05-14).

## GPU Experiment Results (2026-05-13)

Installed CUDA-enabled PyTorch with the official CUDA 12.8 wheel. Verification:

```text
torch: 2.11.0+cu128
torch.version.cuda: 12.8
torch.cuda.is_available(): True
```

Commands:

```powershell
python scripts/run_speed.py --method dense --max-new-tokens 64 --device cuda --output results/speed_dense_gpu.json
python scripts/run_speed.py --method window --cache-budget 32 --recent 16 --max-new-tokens 64 --device cuda --output results/speed_window_gpu.json
python scripts/run_speed.py --method hybrid --cache-budget 32 --recent 16 --max-new-tokens 64 --device cuda --output results/speed_hybrid_gpu.json
```

| Method | Device | TTFT | TPOT | Throughput | Cache len | KV reduction | KV accesses | CUDA peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | cuda | 0.2481 | 0.0059 | 101.2409 | 74 | 0.00% | 16320 | 146.4995 |
| window | cuda | 0.2283 | 0.0045 | 118.2867 | 32 | 56.76% | 10902 | 146.4995 |
| hybrid | cuda | 0.2196 | 0.0045 | 106.4172 | 32 | 56.76% | 10902 | 146.4995 |
