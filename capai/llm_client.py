"""
capai.llm_client
=================
Unified LLM client with a two-tier provider chain and usage tracking.

Behaviour:
  - Tries NVIDIA NIM first if NVIDIA_API_KEY is set (primary — free tier
    has no daily token cap, only a 40 req/min rate limit, which is far
    more forgiving for repeated benchmark/synthesis calls than Groq's
    hard 100k-tokens/day ceiling that this replaced).
  - On NVIDIA failure, falls back to Anthropic if ANTHROPIC_API_KEY is set.
  - Groq has been removed from the provider chain entirely (see the
    `complete()` docstring below for why).
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
    Provider chain: NVIDIA NIM (primary) -> Anthropic (fallback).
    Groq has been removed from the chain entirely — its free tier's
    hard 100k-tokens/day ceiling was blocking benchmark runs and it was
    causing confusion about which provider actually served a given
    request, so it is no longer attempted even if GROQ_API_KEY happens
    to still be set in the environment.
    Raises RuntimeError if no LLM is configured at all, or if every
    configured tier fails.
    """
    if not (config.NVIDIA_API_KEY or config.ANTHROPIC_API_KEY):
        raise RuntimeError("complete() called with no LLM configured (need NVIDIA_API_KEY or ANTHROPIC_API_KEY)")

    last_error: Optional[Exception] = None

    if config.NVIDIA_API_KEY:
        try:
            return _complete_nvidia(prompt, max_tokens)
        except Exception as e:
            last_error = e
            print(f"[llm_client] NVIDIA NIM failed ({e}) — trying next provider.")

    if config.ANTHROPIC_API_KEY:
        try:
            return _complete_anthropic(prompt, max_tokens, fallback=last_error is not None)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"All configured LLM providers failed. Last error: {last_error}")


def _complete_nvidia(prompt: str, max_tokens: int, fallback: bool = False) -> str:
    """
    NVIDIA NIM — hosted at build.nvidia.com, fully OpenAI-compatible.
    Free tier: 40 requests/minute, no daily token cap (unlike Groq's
    hard 100k-tokens/day ceiling, which is what motivated adding this
    as the primary provider).

    Uses streaming because config.NVIDIA_MODEL defaults to a reasoning
    model (nemotron-3-ultra-550b-a55b) which emits a separate
    `reasoning_content` stream before the final `content` — the two are
    accumulated separately and only the final content is returned to
    the caller (CapAI needs the code/answer, not the model's visible
    chain-of-thought). If a non-reasoning model is configured instead,
    reasoning_content simply never appears and this still works
    correctly with no code change needed.
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
        # some reasoning models put everything in reasoning_content on
        # very short/simple prompts — fall back to that rather than
        # returning an empty string, which would otherwise silently
        # break every downstream _strip_fences() call
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
