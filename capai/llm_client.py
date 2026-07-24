"""
capai.llm_client
=================
Unified LLM client: NVIDIA NIM (primary) -> Groq (fallback).
Clean two-provider chain — Cerebras and Anthropic removed for now.
Usage tracking persisted to MongoDB when available.
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
        "provider": provider, "model": model,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "fallback_used": fallback_used, "timestamp": time.time(),
    }
    with _usage_lock:
        _usage_log.append(entry)
        if len(_usage_log) > 5000:
            del _usage_log[:1000]
    if config.MONGODB_URI:
        try:
            from pymongo import MongoClient
            MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=3000)["capai"]["usage_log"].insert_one(entry)
        except Exception:
            pass


def get_usage_summary() -> dict:
    with _usage_lock:
        log = list(_usage_log)
    by_provider = {}
    for e in log:
        by_provider.setdefault(e["provider"], 0)
        by_provider[e["provider"]] += 1
    return {
        "total_calls": len(log),
        "total_input_tokens": sum(e["input_tokens"] for e in log),
        "total_output_tokens": sum(e["output_tokens"] for e in log),
        "fallback_count": sum(1 for e in log if e["fallback_used"]),
        "calls_by_provider": by_provider,
    }


def complete(prompt: str, max_tokens: int = 800) -> str:
    """
    Provider chain: NVIDIA NIM (primary) -> Groq (fallback).
    NVIDIA uses meta/llama-3.1-70b-instruct — chosen for lower
    congestion vs Nemotron Ultra (5M vs 52M monthly calls on the
    shared free infrastructure). Groq is fallback for when NVIDIA
    is slow or unavailable; its 100k-tokens/day limit means it cannot
    sustain a full benchmark run alone, but works fine as a fallback
    for individual calls.
    """
    if not (config.NVIDIA_API_KEY or config.GROQ_API_KEY):
        raise RuntimeError("No LLM configured — set NVIDIA_API_KEY or GROQ_API_KEY")

    last_error: Optional[Exception] = None

    if config.NVIDIA_API_KEY:
        try:
            return _complete_nvidia(prompt, max_tokens)
        except Exception as e:
            last_error = e
            print(f"[llm_client] NVIDIA failed ({e}) — trying Groq.")

    if config.GROQ_API_KEY:
        try:
            return _complete_groq(prompt, max_tokens, fallback=last_error is not None)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"All providers failed. Last error: {last_error}")


def _complete_nvidia(prompt: str, max_tokens: int, fallback: bool = False) -> str:
    from openai import OpenAI
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=config.NVIDIA_API_KEY)
    stream = client.chat.completions.create(
        model=config.NVIDIA_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": config.NVIDIA_ENABLE_THINKING},
            "reasoning_budget": config.NVIDIA_REASONING_BUDGET,
        },
        stream=True,
    )
    content_parts, reasoning_parts = [], []
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
    final = "".join(content_parts).strip()
    return final or "".join(reasoning_parts).strip()


def _complete_groq(prompt: str, max_tokens: int, fallback: bool = False) -> str:
    from groq import Groq
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(response, "usage", None)
    _record_usage(
        "groq", config.GROQ_MODEL,
        getattr(usage, "prompt_tokens", 0) if usage else 0,
        getattr(usage, "completion_tokens", 0) if usage else 0,
        fallback,
    )
    return response.choices[0].message.content.strip()
