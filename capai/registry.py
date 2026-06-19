"""
capai.registry
===============
The Main Registry from the architecture diagram: the single source of
truth for which capabilities CapAI can currently call without re-running
the acquisition loop. Persisted as JSON under config.REGISTRY_PATH so it
survives between process runs.

By convention (enforced socially, not technically, in this prototype)
`add`, `merge`, and `retire` are only ever called by ManagerAgent — see
manager_agent.py's module docstring.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from . import config
from .models import Capability, CapabilitySpec


class CapabilityRegistry:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or config.REGISTRY_PATH
        self._lock = threading.Lock()
        self._capabilities: dict[str, Capability] = {}
        self._load()

    # ------------------------------------------------------------ lookups
    def has(self, name: str) -> bool:
        cap = self._capabilities.get(name)
        return cap is not None and cap.approved and not cap.retired

    def get(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def list_active(self) -> list[Capability]:
        return [c for c in self._capabilities.values() if c.approved and not c.retired]

    def list_all(self) -> list[Capability]:
        """Including retired / never-promoted entries — mainly useful for debugging."""
        return list(self._capabilities.values())

    # ------------------------------------------------------------ mutations
    def add(self, capability: Capability) -> None:
        with self._lock:
            self._capabilities[capability.name] = capability
            self._save()

    def merge(self, keep_name: str, redundant_name: str) -> None:
        """
        The Manager Agent found that `redundant_name` does the same job as
        the already-active capability `keep_name`. Rather than silently
        deleting the redundant draft (losing the provenance of why this
        gap was actually closed), it's recorded as a retired alias that
        points at the capability callers should use instead.
        """
        with self._lock:
            alias = self._capabilities.get(redundant_name)
            if alias is None:
                alias = Capability(
                    name=redundant_name,
                    description=f"alias of '{keep_name}'",
                    source_code="",
                    spec=CapabilitySpec(name=redundant_name, description="", signature=""),
                    mcp_id="",
                )
            alias.approved = False
            alias.retired = True
            alias.verification = {**(alias.verification or {}), "alias_of": keep_name}
            self._capabilities[redundant_name] = alias
            self._save()

    def retire(self, name: str) -> bool:
        with self._lock:
            cap = self._capabilities.get(name)
            if cap is None:
                return False
            cap.retired = True
            self._save()
            return True

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text())
        for entry in raw.values():
            spec = CapabilitySpec(**entry["spec"])
            cap = Capability(
                name=entry["name"],
                description=entry["description"],
                source_code=entry["source_code"],
                spec=spec,
                mcp_id=entry["mcp_id"],
                version=entry.get("version", "0.0.1"),
                verified=entry.get("verified", False),
                approved=entry.get("approved", False),
                retired=entry.get("retired", False),
                verification=entry.get("verification"),
            )
            self._capabilities[cap.name] = cap

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            name: {
                "name": c.name,
                "description": c.description,
                "source_code": c.source_code,
                "spec": vars(c.spec),
                "mcp_id": c.mcp_id,
                "version": c.version,
                "verified": c.verified,
                "approved": c.approved,
                "retired": c.retired,
                "verification": c.verification,
            }
            for name, c in self._capabilities.items()
        }
        self.path.write_text(json.dumps(serialisable, indent=2))
