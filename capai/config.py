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
# Provider priority: NVIDIA NIM > Cerebras > Anthropic > offline heuristic.
# NVIDIA is primary using mistralai/mistral-medium-3.5-128b — chosen
# from the live build.nvidia.com catalog based on monthly call counts:
# Nemotron Ultra had 52M calls/month (caused 41s latency in testing),
# Mistral Medium 3.5 has 5M calls/month (10x less contention on the
# same shared free infrastructure). Cerebras is first fallback given
# its 1M tokens/day budget and fast WSE hardware.
# Groq is NOT part of the chain — hard 100k-tokens/day ceiling hit
# repeatedly during benchmarking. GROQ_API_KEY kept for compatibility only.

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_MODEL   = os.environ.get("CAPAI_NVIDIA_MODEL", "mistralai/mistral-medium-3.5-128b")
NVIDIA_ENABLE_THINKING  = os.environ.get("CAPAI_NVIDIA_THINKING", "false").lower() == "true"
NVIDIA_REASONING_BUDGET = int(os.environ.get("CAPAI_NVIDIA_REASONING_BUDGET", "4096"))

CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
CEREBRAS_MODEL   = os.environ.get("CAPAI_CEREBRAS_MODEL", "gpt-oss-120b")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")   # unused — kept for backward compatibility
GROQ_MODEL   = os.environ.get("CAPAI_GROQ_MODEL", "llama-3.3-70b-versatile")  # unused

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = os.environ.get("CAPAI_MODEL", "claude-sonnet-4-6")

LLM_ENABLED = bool(NVIDIA_API_KEY or CEREBRAS_API_KEY or ANTHROPIC_API_KEY)

# ---------------------------------------------------------------- storage
CAPAI_HOME      = Path(os.environ.get("CAPAI_HOME", Path.cwd() / ".capai"))
MCP_SERVERS_DIR = CAPAI_HOME / "mcp_servers"
REGISTRY_PATH   = CAPAI_HOME / "registry.json"

MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loop behaviour
MAX_ACQUISITION_ATTEMPTS = int(os.environ.get("CAPAI_MAX_ATTEMPTS", "3"))
SANDBOX_TIMEOUT_SECONDS  = float(os.environ.get("CAPAI_SANDBOX_TIMEOUT", "5"))
