"""
capai.registry
===============
The Main Registry — single source of truth for all capabilities.

Persistence priority:
  1. MongoDB (if MONGODB_URI is set) — permanent, survives all restarts
  2. JSON file (CAPAI_HOME/registry.json) — local fallback
  3. In-memory only — if neither is available
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from . import config
from .models import Capability, CapabilitySpec


def _cap_to_dict(c: Capability) -> dict:
    return {
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


def _dict_to_cap(entry: dict) -> Capability:
    spec_data = entry["spec"]
    # handle extra fields CapabilitySpec might not expect
    valid_fields = {"name", "description", "signature", "example_inputs",
                    "expected_behavior", "root_cause"}
    spec = CapabilitySpec(**{k: v for k, v in spec_data.items() if k in valid_fields})
    return Capability(
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


class CapabilityRegistry:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or (config.CAPAI_HOME / "registry.json")
        self._lock = threading.Lock()
        self._capabilities: dict[str, Capability] = {}
        self._mongo = None
        self._collection = None
        self._init_mongo()
        self._load()

    # ── MongoDB setup ─────────────────────────────────────────────────────────

    def _init_mongo(self):
        uri = config.MONGODB_URI
        if not uri:
            return
        try:
            from pymongo import MongoClient
            self._mongo = MongoClient(uri, serverSelectionTimeoutMS=5000)
            db = self._mongo["capai"]
            self._collection = db["capabilities"]
            # test connection
            self._mongo.admin.command("ping")
            print("[registry] MongoDB connected — capabilities will persist permanently.")
        except Exception as e:
            print(f"[registry] MongoDB unavailable ({e}) — falling back to JSON file.")
            self._mongo = None
            self._collection = None

    # ── lookups ───────────────────────────────────────────────────────────────

    def has(self, name: str) -> bool:
        cap = self._capabilities.get(name)
        return cap is not None and cap.approved and not cap.retired

    def get(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def list_active(self) -> list[Capability]:
        return [c for c in self._capabilities.values() if c.approved and not c.retired]

    def list_all(self) -> list[Capability]:
        return list(self._capabilities.values())

    # ── mutations ─────────────────────────────────────────────────────────────

    def add(self, capability: Capability) -> None:
        with self._lock:
            self._capabilities[capability.name] = capability
            self._save_one(capability)

    def merge(self, keep_name: str, redundant_name: str) -> None:
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
            self._save_one(alias)

    def retire(self, name: str) -> bool:
        with self._lock:
            cap = self._capabilities.get(name)
            if cap is None:
                return False
            cap.retired = True
            self._save_one(cap)
            return True

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load from MongoDB first, fall back to JSON file."""
        if self._collection is not None:
            self._load_mongo()
        else:
            self._load_json()

    def _load_mongo(self) -> None:
        try:
            docs = self._collection.find({})
            count = 0
            for doc in docs:
                doc.pop("_id", None)
                cap = _dict_to_cap(doc)
                self._capabilities[cap.name] = cap
                count += 1
            if count:
                print(f"[registry] Loaded {count} capabilities from MongoDB.")
        except Exception as e:
            print(f"[registry] Failed to load from MongoDB: {e}")

    def _load_json(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
            for entry in raw.values():
                cap = _dict_to_cap(entry)
                self._capabilities[cap.name] = cap
        except Exception as e:
            print(f"[registry] Failed to load JSON registry: {e}")

    def _save_one(self, cap: Capability) -> None:
        """Save a single capability — MongoDB upsert or full JSON rewrite."""
        if self._collection is not None:
            self._save_one_mongo(cap)
        else:
            self._save_json()

    def _save_one_mongo(self, cap: Capability) -> None:
        try:
            self._collection.update_one(
                {"name": cap.name},
                {"$set": _cap_to_dict(cap)},
                upsert=True,
            )
        except Exception as e:
            print(f"[registry] MongoDB save failed for '{cap.name}': {e} — falling back to JSON.")
            self._save_json()

    def _save_json(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serialisable = {name: _cap_to_dict(c) for name, c in self._capabilities.items()}
            self.path.write_text(json.dumps(serialisable, indent=2))
        except Exception as e:
            print(f"[registry] JSON save failed: {e}")
