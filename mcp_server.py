"""
capai.mcp_server
=================
Represents one "isolated MCP server" from the architecture diagram: a
dedicated, versioned workspace created the moment the Orchestrator detects
a missing capability. The Diagnostic Agent, Code Writer, and Testing Agent
all do their work inside one of these rather than touching the host
process or the Main Registry directly.

Two things live on disk per MCP server:
  - a local git repository, so every generated draft of the capability is
    committed and the module's history is genuinely versioned (Section 3.3
    of the report: "committed to its own versioned GitHub repository" —
    this prototype versions locally with git; pushing to an actual GitHub
    remote only requires adding a `git remote add origin ...` once you have
    a repo URL and credentials, which is intentionally left as a deployment
    step rather than hard-coded here).
  - a history.json file recording every previous attempt (spec, code,
    verdict) so the Testing Agent's "replay against prior MCP history"
    layer has something real to replay against.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from . import config
from .models import CapabilitySpec, new_mcp_id


class MCPServer:
    def __init__(self, capability_name: str, mcp_id: Optional[str] = None):
        self.capability_name = capability_name
        self.id = mcp_id or new_mcp_id(capability_name)
        self.workdir = config.MCP_SERVERS_DIR / self.id
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.module_path = self.workdir / "capability.py"
        self.history_path = self.workdir / "history.json"
        if not self.history_path.exists():
            self.history_path.write_text("[]")
        self._ensure_git_repo()

    # ------------------------------------------------------------- git repo
    def _ensure_git_repo(self) -> None:
        if (self.workdir / ".git").exists():
            return
        self._git("init", "-q")
        self._git("config", "user.email", "capai@local")
        self._git("config", "user.name", "CapAI Code Writer")

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.workdir, capture_output=True, text=True)

    def commit_module(self, source_code: str, message: str) -> str:
        """Write the latest draft to disk and commit it. Returns the commit hash."""
        self.module_path.write_text(source_code)
        self._git("add", "capability.py")
        self._git("commit", "-q", "-m", message, "--allow-empty")
        result = self._git("rev-parse", "HEAD")
        return result.stdout.strip()

    # ------------------------------------------------------------- history
    def load_history(self) -> list[dict]:
        return json.loads(self.history_path.read_text())

    def record_attempt(self, spec: CapabilitySpec, source_code: str, passed: bool, notes: str = "") -> None:
        history = self.load_history()
        history.append({
            "spec_name": spec.name,
            "signature": spec.signature,
            "source_code": source_code,
            "passed": passed,
            "notes": notes,
        })
        self.history_path.write_text(json.dumps(history, indent=2))

    def __repr__(self) -> str:
        return f"MCPServer(id={self.id!r}, capability={self.capability_name!r}, workdir={self.workdir})"
