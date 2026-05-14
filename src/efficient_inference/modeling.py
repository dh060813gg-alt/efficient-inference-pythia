from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .methods import CacheConfig, cache_size, cache_tokens, compress_past_key_values


@dataclass
class GenerationMetrics:
    method: str
    prompt_tokens: int
    generated_tokens: int
    ttft_seconds: float
    tpot_seconds: float
    throughput_tokens_per_second: float
    total_seconds: float
    final_cache_length: int
    final_cache_elements: int
    dense_cache_length_estimate: int
    kv_cache_reduction_ratio: float
    avg_cache_length: float
    attention_kv_accesses_estimate: int
    device: str
    cuda_peak_memory_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model(model_name: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, torch.device(device)


@torch.no_grad()
def generate_with_cache_policy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cache_config: CacheConfig,
    device: torch.device,
) -> tuple[str, GenerationMetrics]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    generated = input_ids.clone()
    past_key_values = None
    first_token_time = 0.0
    decode_times: list[float] = []
    cache_lengths: list[int] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_total = time.perf_counter()
    next_input = input_ids

    for step in range(max_new_tokens):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_step = time.perf_counter()
        outputs = model(
            input_ids=next_input,
            attention_mask=attention_mask if step == 0 else None,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        elapsed = time.perf_counter() - start_step
        if step == 0:
            first_token_time = elapsed
        else:
            decode_times.append(elapsed)
        generated = torch.cat([generated, next_token], dim=1)
        past_key_values = compress_past_key_values(outputs.past_key_values, cache_config, outputs.attentions)
        cache_lengths.append(cache_size(past_key_values))
        next_input = next_token
        attention_mask = None
        if next_token.item() == tokenizer.eos_token_id:
            break

    total_seconds = time.perf_counter() - start_total
    produced = generated.size(1) - input_ids.size(1)
    tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0
    throughput = produced / total_seconds if total_seconds > 0 else 0.0
    dense_cache_length = input_ids.size(1) + max(produced - 1, 0)
    final_cache_length = cache_size(past_key_values)
    reduction = 1.0 - (final_cache_length / dense_cache_length) if dense_cache_length > 0 else 0.0
    avg_cache_length = sum(cache_lengths) / len(cache_lengths) if cache_lengths else 0.0
    num_layers = getattr(model.config, "num_hidden_layers", 0)
    attention_kv_accesses = int(sum(cache_lengths) * max(num_layers, 1))
    cuda_peak_memory_mb = None
    if device.type == "cuda":
        cuda_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    metrics = GenerationMetrics(
        method=cache_config.policy,
        prompt_tokens=input_ids.size(1),
        generated_tokens=produced,
        ttft_seconds=first_token_time,
        tpot_seconds=tpot,
        throughput_tokens_per_second=throughput,
        total_seconds=total_seconds,
        final_cache_length=final_cache_length,
        final_cache_elements=cache_tokens(past_key_values),
        dense_cache_length_estimate=dense_cache_length,
        kv_cache_reduction_ratio=reduction,
        avg_cache_length=avg_cache_length,
        attention_kv_accesses_estimate=attention_kv_accesses,
        device=str(device),
        cuda_peak_memory_mb=cuda_peak_memory_mb,
    )
    return text, metrics
