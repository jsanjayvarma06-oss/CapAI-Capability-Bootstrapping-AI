"""
capai.config
=============
Centralised, environment-driven configuration. Every other module reads
these as `config.SOMETHING` at call time (not at import time) so tests
can monkeypatch them per-test without needing to reload modules.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- persistence
# MongoDB URI for permanent capability storage (recommended).
# Set MONGODB_URI env var to your Atlas connection string.
# If not set, falls back to local JSON file (lost on Render restart).
MONGODB_URI = os.environ.get("MONGODB_URI")

# ---------------------------------------------------------------- LLM
# Provider priority: Cerebras > NVIDIA NIM > Anthropic > offline heuristic.
# Cerebras is primary: 1,000,000 tokens/day free, 30 req/min, runs on
# custom Wafer-Scale Engine hardware (much faster than typical GPU
# inference). NVIDIA NIM is kept as a fallback after being demoted from
# primary — its shared free-tier infrastructure was observed under
# heavy congestion (a trivial request took 41+ seconds during testing).
# Groq is NOT part of the chain — its free tier's hard 100k-tokens/day
# ceiling repeatedly blocked benchmark runs. GROQ_API_KEY / GROQ_MODEL
# are kept below only so an old .env with that variable still set
# doesn't break anything; they are never read by llm_client.complete().
#
# Set CEREBRAS_API_KEY to use Cerebras (primary).
# Set NVIDIA_API_KEY (starts with "nvapi-") to use NVIDIA NIM (fallback).
# Set ANTHROPIC_API_KEY to use Anthropic Claude (final fallback).
# Set none to run in offline heuristic mode (no LLM, no cost).

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
# gpt-oss-120b is a reasoning model on Cerebras (generates intermediate
# thinking tokens before its final answer, similar to the NVIDIA
# Nemotron model tested earlier) — llm_client.py handles this the same
# defensive way: prefer final content, fall back to reasoning content
# only if content comes back empty. Switched from llama-3.3-70b after
# that model returned a 404 "model not found / no access" error for
# this account despite matching Cerebras' documented model ID.
CEREBRAS_MODEL = os.environ.get("CAPAI_CEREBRAS_MODEL", "gpt-oss-120b")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
# Default is a plain instruct model, NOT a reasoning model — testing
# showed nvidia/nemotron-3-ultra-550b-a55b (a large reasoning model)
# took 8-40 seconds per synthesis call and burned through free-tier
# credits far faster than a plain instruct model for the same task,
# since CapAI's code-synthesis prompts don't need deep chain-of-thought.
NVIDIA_MODEL = os.environ.get("CAPAI_NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_ENABLE_THINKING  = os.environ.get("CAPAI_NVIDIA_THINKING", "false").lower() == "true"
NVIDIA_REASONING_BUDGET = int(os.environ.get("CAPAI_NVIDIA_REASONING_BUDGET", "4096"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")   # unused — kept for backward compatibility only
GROQ_MODEL   = os.environ.get("CAPAI_GROQ_MODEL", "llama-3.3-70b-versatile")  # unused

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = os.environ.get("CAPAI_MODEL", "claude-sonnet-4-6")

# LLM_ENABLED reflects only the providers actually used by llm_client.complete()
LLM_ENABLED = bool(CEREBRAS_API_KEY or NVIDIA_API_KEY or ANTHROPIC_API_KEY)

# ---------------------------------------------------------------- storage
CAPAI_HOME      = Path(os.environ.get("CAPAI_HOME", Path.cwd() / ".capai"))
MCP_SERVERS_DIR = CAPAI_HOME / "mcp_servers"
REGISTRY_PATH   = CAPAI_HOME / "registry.json"

MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loop behaviour
MAX_ACQUISITION_ATTEMPTS = int(os.environ.get("CAPAI_MAX_ATTEMPTS", "3"))
SANDBOX_TIMEOUT_SECONDS  = float(os.environ.get("CAPAI_SANDBOX_TIMEOUT", "5"))
