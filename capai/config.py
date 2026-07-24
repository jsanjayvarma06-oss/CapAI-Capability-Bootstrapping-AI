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
# Provider priority: NVIDIA NIM > Groq > Anthropic > offline heuristic.
# NVIDIA is primary: free tier has no daily token cap (only a 40 req/min
# rate limit), unlike Groq's hard 100k-tokens/day ceiling which was
# hit during benchmarking and motivated this change.
# Set NVIDIA_API_KEY (starts with "nvapi-") to use NVIDIA NIM.
# Set GROQ_API_KEY to use Groq as first fallback.
# Set ANTHROPIC_API_KEY to use Anthropic Claude as second fallback.
# Set none to run in offline heuristic mode (no LLM, no cost).

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL   = os.environ.get("CAPAI_NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
# This is a large reasoning model — noticeably slower per call than a
# plain 70B instruct model, since it "thinks" before answering. If
# synthesis latency matters more than answer quality for your use case,
# set CAPAI_NVIDIA_MODEL to a smaller/faster model instead, e.g.
# "meta/llama-3.1-70b-instruct".
NVIDIA_ENABLE_THINKING  = os.environ.get("CAPAI_NVIDIA_THINKING", "true").lower() == "true"
NVIDIA_REASONING_BUDGET = int(os.environ.get("CAPAI_NVIDIA_REASONING_BUDGET", "16384"))

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")
GROQ_MODEL       = os.environ.get("CAPAI_GROQ_MODEL", "llama-3.3-70b-versatile")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = os.environ.get("CAPAI_MODEL", "claude-sonnet-4-6")

# LLM_ENABLED is True if any provider key is present
LLM_ENABLED = bool(NVIDIA_API_KEY or GROQ_API_KEY or ANTHROPIC_API_KEY)

# ---------------------------------------------------------------- storage
CAPAI_HOME      = Path(os.environ.get("CAPAI_HOME", Path.cwd() / ".capai"))
MCP_SERVERS_DIR = CAPAI_HOME / "mcp_servers"
REGISTRY_PATH   = CAPAI_HOME / "registry.json"

MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loop behaviour
MAX_ACQUISITION_ATTEMPTS = int(os.environ.get("CAPAI_MAX_ATTEMPTS", "3"))
SANDBOX_TIMEOUT_SECONDS  = float(os.environ.get("CAPAI_SANDBOX_TIMEOUT", "5"))
