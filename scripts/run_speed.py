from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from efficient_inference.eval import run_speed


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark autoregressive decoding speed with cache policies.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--method", default="dense", choices=["dense", "window", "hybrid"])
    parser.add_argument("--prompt", default="Efficient language model inference reduces latency and memory usage by")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--prompt-chars", type=int, default=1200)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-budget", type=int, default=128)
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/speed.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = run_speed(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
