from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=4):
    if value is None:
        return "--"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Summarize result JSON files as markdown tables.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    root = Path(args.results_dir)

    ppl_files = sorted(root.glob("ppl*.json"))
    speed_files = sorted(root.glob("speed*.json"))

    if ppl_files:
        print("## Perplexity")
        print()
        print("| File | Dataset | Samples | Tokens | PPL | Mean NLL |")
        print("| --- | --- | ---: | ---: | ---: | ---: |")
        for path in ppl_files:
            data = load_json(path)
            print(
                "| "
                + " | ".join(
                    [
                        path.name,
                        fmt(data.get("dataset")),
                        fmt(data.get("samples")),
                        fmt(data.get("tokens")),
                        fmt(data.get("ppl")),
                        fmt(data.get("mean_nll")),
                    ]
                )
                + " |"
            )
        print()

    if speed_files:
        print("## Speed")
        print()
        print("| File | Method | Device | TTFT | TPOT | Throughput | Cache len | Cache reduction | KV accesses | CUDA peak MB |")
        print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for path in speed_files:
            data = load_json(path)
            print(
                "| "
                + " | ".join(
                    [
                        path.name,
                        fmt(data.get("method")),
                        fmt(data.get("device")),
                        fmt(data.get("ttft_seconds")),
                        fmt(data.get("tpot_seconds")),
                        fmt(data.get("throughput_tokens_per_second")),
                        fmt(data.get("final_cache_length")),
                        fmt(data.get("kv_cache_reduction_ratio")),
                        fmt(data.get("attention_kv_accesses_estimate")),
                        fmt(data.get("cuda_peak_memory_mb")),
                    ]
                )
                + " |"
            )


if __name__ == "__main__":
    main()
