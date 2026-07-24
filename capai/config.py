import os

# ---------------------------------------------------------------- MongoDB
MONGODB_URI = os.environ.get("MONGODB_URI")

# ---------------------------------------------------------------- LLM
# Provider chain: NVIDIA NIM (primary) -> Groq (fallback).
# Cerebras and Anthropic removed from the chain for simplicity.
# NVIDIA model is meta/llama-3.1-70b-instruct — lower congestion than
# the previously-tried Nemotron Ultra (5M vs 52M monthly calls).
# Groq is fallback only; its 100k-tokens/day cap means it can't carry
# a full benchmark run alone, but works for individual fallback calls.

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL   = os.environ.get("CAPAI_NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_ENABLE_THINKING  = os.environ.get("CAPAI_NVIDIA_THINKING", "false").lower() == "true"
NVIDIA_REASONING_BUDGET = int(os.environ.get("CAPAI_NVIDIA_REASONING_BUDGET", "4096"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL   = os.environ.get("CAPAI_GROQ_MODEL", "llama-3.3-70b-versatile")

# Kept for backward compat — not in the active chain
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
CEREBRAS_MODEL   = os.environ.get("CAPAI_CEREBRAS_MODEL", "gpt-oss-120b")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = os.environ.get("CAPAI_MODEL", "claude-sonnet-4-6")

LLM_ENABLED = bool(NVIDIA_API_KEY or GROQ_API_KEY)
