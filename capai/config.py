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

# ---------------------------------------------------------------- LLM
# Provider priority: Groq > Anthropic > offline heuristic.
# Set GROQ_API_KEY to use Groq (faster, free tier available).
# Set ANTHROPIC_API_KEY to use Anthropic Claude.
# Set neither to run in offline heuristic mode (no LLM, no cost).

GROQ_API_KEY     = os.environ.get("GROQ_API_KEY")
GROQ_MODEL       = os.environ.get("CAPAI_GROQ_MODEL", "llama-3.3-70b-versatile")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = os.environ.get("CAPAI_MODEL", "claude-sonnet-4-6")

# LLM_ENABLED is True if either provider key is present
LLM_ENABLED = bool(GROQ_API_KEY or ANTHROPIC_API_KEY)

# ---------------------------------------------------------------- storage
CAPAI_HOME      = Path(os.environ.get("CAPAI_HOME", Path.cwd() / ".capai"))
MCP_SERVERS_DIR = CAPAI_HOME / "mcp_servers"
REGISTRY_PATH   = CAPAI_HOME / "registry.json"

MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loop behaviour
MAX_ACQUISITION_ATTEMPTS = int(os.environ.get("CAPAI_MAX_ATTEMPTS", "3"))
SANDBOX_TIMEOUT_SECONDS  = float(os.environ.get("CAPAI_SANDBOX_TIMEOUT", "5"))
