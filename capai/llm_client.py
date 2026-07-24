"""
capai.llm_client
=================
Unified LLM client with a three-tier provider chain and usage tracking.

Behaviour:
  - Tries Cerebras first if CEREBRAS_API_KEY is set (primary — free tier
    offers 1,000,000 tokens/day, 30 requests/minute, and runs on
    Cerebras' custom Wafer-Scale Engine hardware which is dramatically
    faster than typical GPU-based inference. This replaced NVIDIA NIM
    as primary after NVIDIA's shared free-tier infrastructure was
    observed under heavy congestion — a single trivial request took
    41+ seconds during testing, with no code-side fix possible for
    that kind of upstream congestion.)
  - On Cerebras failure, falls back to NVIDIA NIM if NVIDIA_API_KEY is set.
  - On NVIDIA failure, falls back to Anthropic if ANTHROPIC_API_KEY is set.
  - Groq is not part of the chain at all (see git history — its free
    tier's hard 100k-tokens/day ceiling repeatedly blocked benchmark runs).
  - Every call's token usage (when the provider reports it) is recorded
    in-memory and, if MongoDB is configured, persisted for cost tracking.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from . import config

_usage_lock = threading.Lock()
_usage_log: list = []


def _record_usage(provider: str, model: str, input_tokens: int, output_tokens: int, fallback_used: bool):
    entry = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "fallback_used": fallback_used,
        "timestamp": time.time(),
    }
    with _usage_lock:
        _usage_log.append(entry)
        if len(_usage_log) > 5000:
            del _usage_log[:1000]
    if config.MONGODB_URI:
        try:
            from pymongo import MongoClient
            client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=3000)
            client["capai"]["usage_log"].insert_one(entry)
        except Exception:
            pass  # usage tracking must never break the actual request


def get_usage_summary() -> dict:
    with _usage_lock:
        log = list(_usage_log)
    total_calls = len(log)
    total_input = sum(e["input_tokens"] for e in log)
    total_output = sum(e["output_tokens"] for e in log)
    fallback_count = sum(1 for e in log if e["fallback_used"])
    by_provider = {}
    for e in log:
        by_provider.setdefault(e["provider"], 0)
        by_provider[e["provider"]] += 1
    return {
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "fallback_count": fallback_count,
        "calls_by_provider": by_provider,
    }


def complete(prompt: str, max_tokens: int = 800) -> str:
    """
    Send a single-turn prompt to the configured LLM and return the text.
    Provider chain: NVIDIA NIM (primary) -> Cerebras -> Anthropic.
    NVIDIA is now primary using mistralai/mistral-medium-3.5-128b —
    a less-congested model than the previously-tried Nemotron Ultra
    (52M monthly calls) chosen based on real catalog usage data:
    Mistral Medium 3.5 has 5M monthly calls (10x less contention).
    Cerebras is kept as first fallback given its 1M tokens/day budget.
    Raises RuntimeError if no LLM is configured at all, or if every
    configured tier fails.
    """
    if not (config.NVIDIA_API_KEY or config.CEREBRAS_API_KEY or config.ANTHROPIC_API_KEY):
        raise RuntimeError(
            "complete() called with no LLM configured "
            "(need NVIDIA_API_KEY, CEREBRAS_API_KEY, or ANTHROPIC_API_KEY)"
        )

    last_error = None

    if config.NVIDIA_API_KEY:
        try:
            return _complete_nvidia(prompt, max_tokens)
        except Exception as e:
            last_error = e
            print(f"[llm_client] NVIDIA NIM failed ({e}) — trying next provider.")

    if config.CEREBRAS_API_KEY:
        try:
            return _complete_cerebras(prompt, max_tokens, fallback=last_error is not None)
        except Exception as e:
            last_error = e
            print(f"[llm_client] Cerebras failed ({e}) — trying next provider.")

    if config.ANTHROPIC_API_KEY:
        try:
            return _complete_anthropic(prompt, max_tokens, fallback=last_error is not None)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"All configured LLM providers failed. Last error: {last_error}")


def _complete_cerebras(prompt: str, max_tokens: int, fallback: bool = False) -> str:
    """
    Cerebras Inference API — hosted at api.cerebras.ai, OpenAI-compatible.
    Free tier: 1,000,000 tokens/day, 30 requests/minute, runs on
    Cerebras' Wafer-Scale Engine (WSE) hardware rather than GPUs, which
    is why it is dramatically faster per-token than typical providers.

    Default model (gpt-oss-120b) is a reasoning model that can return
    intermediate "thinking" content on a separate field from the final
    answer. This function defensively checks message.content first and
    only falls back to message.reasoning_content if content comes back
    empty — the same pattern used for the NVIDIA reasoning model in
    _complete_nvidia, so an empty string never silently reaches
    downstream code that expects real content (e.g. _strip_fences()).
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=config.CEREBRAS_API_KEY,
    )
    response = client.chat.completions.create(
        model=config.CEREBRAS_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(response, "usage", None)
    _record_usage(
        "cerebras", config.CEREBRAS_MODEL,
        getattr(usage, "prompt_tokens", 0) if usage else 0,
        getattr(usage, "completion_tokens", 0) if usage else 0,
        fallback,
    )

    message = response.choices[0].message
    content = (message.content or "").strip()
    if not content:
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        content = (reasoning or "").strip()
    return content


def _complete_nvidia(prompt: str, max_tokens: int, fallback: bool = False) -> str:
    """
    NVIDIA NIM — hosted at build.nvidia.com, fully OpenAI-compatible.
    Kept as a fallback tier (not primary — see module docstring for why
    it was demoted after observed free-tier congestion).

    Uses streaming to correctly handle reasoning models that emit a
    separate `reasoning_content` stream before the final `content` —
    accumulated separately, only final content is returned. If a plain
    instruct model is configured (the current default), reasoning_content
    simply never appears and this still works correctly.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=config.NVIDIA_API_KEY,
    )
    stream = client.chat.completions.create(
        model=config.NVIDIA_MODEL,
        max_tokens=max_tokens,
        temperature=1,
        top_p=0.95,
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": config.NVIDIA_ENABLE_THINKING},
            "reasoning_budget": config.NVIDIA_REASONING_BUDGET,
        },
        stream=True,
    )

    content_parts = []
    reasoning_parts = []
    prompt_tokens = completion_tokens = 0

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
        if delta.content is not None:
            content_parts.append(delta.content)
        usage = getattr(chunk, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", prompt_tokens)
            completion_tokens = getattr(usage, "completion_tokens", completion_tokens)

    _record_usage("nvidia", config.NVIDIA_MODEL, prompt_tokens, completion_tokens, fallback)

    final_content = "".join(content_parts).strip()
    if not final_content:
        final_content = "".join(reasoning_parts).strip()
    return final_content


def _complete_anthropic(prompt: str, max_tokens: int, fallback: bool = False) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(response, "usage", None)
    _record_usage(
        "anthropic", config.ANTHROPIC_MODEL,
        getattr(usage, "input_tokens", 0) if usage else 0,
        getattr(usage, "output_tokens", 0) if usage else 0,
        fallback,
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()
