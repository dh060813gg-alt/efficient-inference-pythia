from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data import builtin_texts, iter_token_windows, load_dataset_texts, load_local_texts
from .methods import CacheConfig
from .modeling import generate_with_cache_policy, load_model


@torch.no_grad()
def compute_ppl(model, tokenizer, texts: list[str], device, max_length: int, stride: int) -> dict[str, Any]:
    losses: list[float] = []
    token_count = 0
    for text in tqdm(texts, desc="ppl"):
        encoded = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encoded["input_ids"].to(device)
        for window in iter_token_windows(input_ids, max_length=max_length, stride=stride):
            if window.size(1) < 2:
                continue
            outputs = model(input_ids=window, labels=window)
            valid_tokens = window.size(1) - 1
            losses.append(float(outputs.loss.item()) * valid_tokens)
            token_count += valid_tokens
    mean_nll = sum(losses) / max(token_count, 1)
    return {"ppl": math.exp(mean_nll), "mean_nll": mean_nll, "tokens": token_count, "samples": len(texts)}


def load_texts(args) -> list[str]:
    local = load_local_texts(args.text_file)
    if local is not None:
        return local[: args.max_samples]
    if args.dataset == "builtin":
        return builtin_texts()[: args.max_samples]
    return load_dataset_texts(args.dataset, args.split, args.max_samples)


def save_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_ppl(args) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, args.device)
    texts = load_texts(args)
    result = compute_ppl(model, tokenizer, texts, device, args.max_length, args.stride)
    result.update({"model": args.model, "dataset": args.dataset})
    save_json(args.output, result)
    return result


def run_speed(args) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, args.device)
    prompt = args.prompt
    if args.text_file:
        prompt = load_local_texts(args.text_file)[0][: args.prompt_chars]
    config = CacheConfig(policy=args.method, budget=args.cache_budget, recent=args.recent, alpha=args.alpha)
    text, metrics = generate_with_cache_policy(model, tokenizer, prompt, args.max_new_tokens, config, device)
    result = metrics.to_dict()
    result.update({"model": args.model, "prompt_preview": prompt[:120], "generated_preview": text[:300]})
    save_json(args.output, result)
    return result
