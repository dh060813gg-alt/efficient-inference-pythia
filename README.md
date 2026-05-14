# Efficient Inference for Pythia-70M

This repository implements a small, reproducible NLP course project for training-free efficient inference of `EleutherAI/pythia-70m`.

## Method

The project compares three decoding cache policies:

- **Dense**: standard Hugging Face generation with all KV cache entries retained.
- **Window**: keeps only the most recent `cache_budget` KV tokens.
- **Hybrid**: keeps a recent window plus older tokens selected by a mixed salience score. The score combines key-vector norm or attention salience with a positional recency prior.

The hybrid policy is a lightweight improvement inspired by KV cache compression methods such as StreamingLLM, H2O, SnapKV, and PyramidKV. It does not train or modify model parameters.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick smoke test

The default dataset is a built-in tiny text set, so the code can be tested without downloading benchmark datasets. If CUDA is available, scripts automatically use GPU; otherwise they fall back to CPU. You can force a device with `--device cuda` or `--device cpu`.

```powershell
python scripts/run_ppl.py --dataset builtin --max-samples 3 --device cpu --output results/ppl_builtin.json
python scripts/run_speed.py --method dense --max-new-tokens 8 --device cpu --output results/speed_dense.json
python scripts/run_speed.py --method window --cache-budget 8 --recent 4 --max-new-tokens 8 --device cpu --output results/speed_window.json
python scripts/run_speed.py --method hybrid --cache-budget 8 --recent 4 --max-new-tokens 8 --device cpu --output results/speed_hybrid.json
```

GPU example:

```powershell
python scripts/run_speed.py --method hybrid --cache-budget 32 --recent 16 --max-new-tokens 64 --device cuda --output results/speed_hybrid_gpu.json
```

## Benchmark commands

For Wikitext perplexity:

```powershell
python scripts/run_ppl.py --dataset wikitext --split test --max-samples 32 --max-length 512 --stride 256 --output results/ppl_wikitext.json
```

For PG-19 long-text perplexity (uses built-in substitute when network unavailable):

```powershell
python scripts/run_ppl.py --dataset pg19 --split test --max-samples 1 --max-length 512 --stride 256 --output results/ppl_pg19.json
```

For speed comparison:

```powershell
python scripts/run_speed.py --method dense --max-new-tokens 64 --output results/speed_dense.json
python scripts/run_speed.py --method window --cache-budget 32 --recent 16 --max-new-tokens 64 --output results/speed_window.json
python scripts/run_speed.py --method hybrid --cache-budget 32 --recent 16 --max-new-tokens 64 --output results/speed_hybrid.json
```

Or run the small bundled experiment script:

```powershell
.\scripts\run_experiments.ps1 -Device cpu
```

## Output metrics

The scripts write JSON files under `results/`:

- **PPL**: perplexity, mean negative log likelihood, evaluated token count.
- **TTFT**: time to first generated token.
- **TPOT**: average time per output token after the first token.
- **Throughput**: generated tokens per second.
- **Cache metrics**: final KV cache length and total stored KV elements.
- **KV cache reduction ratio**: relative cache-length reduction versus dense decoding.
- **Attention KV accesses estimate**: a lightweight proxy for attention-side KV reads, computed from cache length across decoding steps and model layers.
- **CUDA peak memory MB**: peak allocated GPU memory during generation, recorded when running on CUDA.

## Summarize results

After running experiments, generate markdown tables with:

```powershell
python scripts/summarize_results.py --results-dir results
```

## Smoke-test results

The following values are from a CPU smoke test with `max_new_tokens=8`, `cache_budget=8`, and `recent=4`. They validate correctness and logging.

| Method | PPL | TTFT ↓ | TPOT ↓ | Throughput ↑ | Cache len ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 365.5567 | 0.0371 | 0.0073 | 89.8464 | 18 |
| Window | 365.5567 | 0.0409 | 0.0072 | 84.8002 | 8 |
| Hybrid | 365.5567 | 0.0378 | 0.0068 | 85.5623 | 8 |

## Wikitext perplexity

| Dataset | Samples | Tokens | PPL | Mean NLL |
| --- | ---: | ---: | ---: | ---: |
| wikitext (test) | 16 | 1376 | 93.3659 | 4.5365 |

## PG-19 long-text perplexity

PG-19 could not be downloaded due to network and library constraints. A built-in long text (public-domain book excerpt, ~10219 tokens) is used as a substitute.

| Dataset | Samples | Tokens | PPL | Mean NLL |
| --- | ---: | ---: | ---: | ---: |
| pg19 (substitute) | 1 | 10219 | 43.6556 | 3.7763 |

## Speed benchmark (64 tokens, cache budget 32, recent 16)

| Method | TTFT ↓ | TPOT ↓ | Throughput ↑ | Cache len ↓ | KV reduction | KV accesses ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.0303 | 0.0065 | 143.5980 | 74 | 0.00% | 16320 |
| Window | 0.0306 | 0.0060 | 149.5136 | 32 | 56.76% | 10902 |
| Hybrid | 0.0288 | 0.0048 | 174.3158 | 32 | 56.76% | 10902 |

## GPU speed benchmark (RTX 4060 Laptop GPU, 64 tokens, cache budget 32, recent 16)

| Method | TTFT ↓ | TPOT ↓ | Throughput ↑ | Cache len ↓ | KV reduction | KV accesses ↓ | CUDA peak MB ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.2481 | 0.0059 | 101.2409 | 74 | 0.00% | 16320 | 146.4995 |
| Window | 0.2283 | 0.0045 | 118.2867 | 32 | 56.76% | 10902 | 146.4995 |
| Hybrid | 0.2196 | 0.0045 | 106.4172 | 32 | 56.76% | 10902 | 146.4995 |

## Notes

- The first run downloads `EleutherAI/pythia-70m` from Hugging Face.
- GPU is used automatically when available; CPU remains supported for reproducibility.
- The implementation prioritizes reproducibility and clarity over maximum performance.
