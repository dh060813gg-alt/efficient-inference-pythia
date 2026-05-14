from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

CachePolicy = Literal["dense", "window", "hybrid"]


@dataclass
class CacheConfig:
    policy: CachePolicy = "dense"
    budget: int = 128
    recent: int = 32
    alpha: float = 0.7


def cache_size(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    past_key_values = _legacy_cache(past_key_values)
    first_key = past_key_values[0][0]
    return int(first_key.size(-2))


def cache_tokens(past_key_values) -> int:
    if past_key_values is None:
        return 0
    past_key_values = _legacy_cache(past_key_values)
    total = 0
    for key, value in past_key_values:
        total += key.numel() + value.numel()
    return int(total)


def compress_past_key_values(past_key_values, config: CacheConfig, attentions=None):
    if past_key_values is None or config.policy == "dense":
        return past_key_values
    current_length = cache_size(past_key_values)
    if current_length <= config.budget:
        return past_key_values
    legacy_cache = _legacy_cache(past_key_values)
    if config.policy == "window":
        indices = torch.arange(current_length - config.budget, current_length, device=legacy_cache[0][0].device)
    elif config.policy == "hybrid":
        indices = _hybrid_indices(legacy_cache, config, attentions)
    else:
        raise ValueError(f"Unknown cache policy: {config.policy}")
    compressed = tuple((_select_cache(key, indices), _select_cache(value, indices)) for key, value in legacy_cache)
    return _replace_cache(past_key_values, compressed)


def _select_cache(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(dim=-2, index=indices.to(tensor.device))


def _legacy_cache(past_key_values):
    if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if past_key_values is not None and hasattr(past_key_values, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_key_values.layers)
    return past_key_values


def _replace_cache(original_cache, compressed):
    if hasattr(original_cache, "layers"):
        for layer, (key, value) in zip(original_cache.layers, compressed):
            layer.keys = key
            layer.values = value
        return original_cache
    return compressed


def _hybrid_indices(past_key_values, config: CacheConfig, attentions=None) -> torch.Tensor:
    length = cache_size(past_key_values)
    device = past_key_values[0][0].device
    budget = min(config.budget, length)
    recent = min(config.recent, budget, length)
    old_budget = budget - recent
    recent_indices = torch.arange(length - recent, length, device=device)
    if old_budget <= 0:
        return recent_indices

    old_length = length - recent
    if attentions:
        score = _attention_score(attentions, old_length, device)
    else:
        score = _key_norm_score(past_key_values, old_length)
    position = torch.linspace(0.0, 1.0, steps=old_length, device=device)
    mixed = config.alpha * score + (1.0 - config.alpha) * position
    top_indices = torch.topk(mixed, k=old_budget, largest=True).indices.sort().values
    return torch.cat([top_indices, recent_indices], dim=0)


def _attention_score(attentions, old_length: int, device) -> torch.Tensor:
    last = attentions[-1]
    if isinstance(last, (tuple, list)):
        last = last[-1]
    score = last.detach().mean(dim=(0, 1, 2))[:old_length].to(device)
    denom = score.max().clamp_min(1e-8)
    return score / denom


def _key_norm_score(past_key_values, old_length: int) -> torch.Tensor:
    scores = []
    for key, _ in past_key_values:
        layer_score = key.detach().float().pow(2).mean(dim=(0, 1, 3))[:old_length]
        scores.append(layer_score)
    score = torch.stack(scores).mean(dim=0)
    denom = score.max().clamp_min(1e-8)
    return score / denom
