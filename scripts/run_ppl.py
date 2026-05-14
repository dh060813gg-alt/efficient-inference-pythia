from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from efficient_inference.eval import run_ppl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate causal LM perplexity.")
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", default="builtin", choices=["builtin", "wikitext", "pg19"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/ppl.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = run_ppl(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
