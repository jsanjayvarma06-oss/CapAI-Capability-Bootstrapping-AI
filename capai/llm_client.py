"""
capai.llm_client
=================
Unified LLM client wrapping Groq and Anthropic with automatic fallback
and usage tracking.

Behaviour:
  - Tries Groq first if GROQ_API_KEY is set.
  - On Groq failure (over-capacity, timeout, error) automatically falls
    back to Anthropic if ANTHROPIC_API_KEY is also set, rather than
    failing the whole request.
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
    Tries Groq first, falls back to Anthropic on any failure if both
    keys are configured. Raises RuntimeError if no LLM is configured.
    """
    if not config.LLM_ENABLED:
        raise RuntimeError("complete() called with no LLM configured")

    if config.GROQ_API_KEY:
        try:
            return _complete_groq(prompt, max_tokens)
        except Exception as e:
            if config.ANTHROPIC_API_KEY:
                print(f"[llm_client] Groq failed ({e}) — falling back to Anthropic.")
                return _complete_anthropic(prompt, max_tokens, fallback=True)
            raise

    return _complete_anthropic(prompt, max_tokens)


def _complete_groq(prompt: str, max_tokens: int, fallback: bool = False) -> str:
    from groq import Groq

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        max_tokens=max_tokens,
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
