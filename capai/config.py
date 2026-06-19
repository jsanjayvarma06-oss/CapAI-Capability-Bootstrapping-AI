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
# If no key is configured, every agent that would normally call an LLM
# falls back to a small built-in heuristic instead, so the full
# acquisition loop is still runnable offline (see capai/demo.py).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("CAPAI_MODEL", "claude-sonnet-4-6")
LLM_ENABLED = bool(ANTHROPIC_API_KEY)

# ---------------------------------------------------------------- storage
# Everything CapAI writes to disk (MCP server workdirs + git repos, and
# the Main Registry's JSON file) lives under this one directory so a
# whole CapAI instance can be reset with `rm -rf .capai`.
CAPAI_HOME = Path(os.environ.get("CAPAI_HOME", Path.cwd() / ".capai"))
MCP_SERVERS_DIR = CAPAI_HOME / "mcp_servers"
REGISTRY_PATH = CAPAI_HOME / "registry.json"

MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- loop behaviour
MAX_ACQUISITION_ATTEMPTS = int(os.environ.get("CAPAI_MAX_ATTEMPTS", "3"))
SANDBOX_TIMEOUT_SECONDS = float(os.environ.get("CAPAI_SANDBOX_TIMEOUT", "5"))
