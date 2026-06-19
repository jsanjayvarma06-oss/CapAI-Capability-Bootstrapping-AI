"""
capai.llm_client
=================
Unified LLM client that wraps both Groq and Anthropic transparently.

Priority:
  1. GROQ_API_KEY set  → use Groq (llama-3.3-70b by default)
  2. ANTHROPIC_API_KEY set → use Anthropic (claude-sonnet-4-6 by default)
  3. Neither → offline heuristic mode (no LLM calls at all)

All agents call  llm_client.complete(prompt, max_tokens)  and never talk
to the provider SDKs directly, so swapping providers is a one-env-var change.
"""
from __future__ import annotations

from . import config


def complete(prompt: str, max_tokens: int = 800) -> str:
    """
    Send a single-turn prompt to the configured LLM and return the response text.
    Raises RuntimeError if called when LLM_ENABLED is False.
    """
    if not config.LLM_ENABLED:
        raise RuntimeError("complete() called with no LLM configured")

    if config.GROQ_API_KEY:
        return _complete_groq(prompt, max_tokens)
    else:
        return _complete_anthropic(prompt, max_tokens)


def _complete_groq(prompt: str, max_tokens: int) -> str:
    from groq import Groq  # type: ignore

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def _complete_anthropic(prompt: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()
