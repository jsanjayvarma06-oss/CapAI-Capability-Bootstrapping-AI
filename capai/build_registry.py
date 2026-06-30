"""
capai.build_registry
======================
Persists advanced /build results (multi-function modules, classes,
pipelines) to MongoDB so a repeated complex request is instant instead
of re-running the full write → test → critique loop every time.

Keyed by a hash of the normalised description, so semantically
identical asks ("build a stack class with push pop peek" vs the exact
same text with different whitespace) hit the same cache entry.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

from . import config


def _hash_description(description: str) -> str:
    normalised = " ".join(description.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:24]


class BuildRegistry:
    def __init__(self):
        self._collection = None
        self._memory: dict = {}
        self._init_mongo()

    def _init_mongo(self):
        if not config.MONGODB_URI:
            return
        try:
            from pymongo import MongoClient
            client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
            db = client["capai"]
            self._collection = db["advanced_builds"]
            client.admin.command("ping")
        except Exception as e:
            print(f"[build_registry] MongoDB unavailable ({e}) — using memory only.")
            self._collection = None

    def get(self, description: str) -> Optional[dict]:
        key = _hash_description(description)
        if self._collection is not None:
            try:
                doc = self._collection.find_one({"_id": key})
                if doc:
                    doc.pop("_id", None)
                    return doc
            except Exception as e:
                print(f"[build_registry] MongoDB read failed: {e}")
        return self._memory.get(key)

    def set(self, description: str, result: dict) -> None:
        key = _hash_description(description)
        result = {**result, "description": description, "cached_at": time.time()}
        self._memory[key] = result
        if self._collection is not None:
            try:
                self._collection.update_one(
                    {"_id": key}, {"$set": result}, upsert=True
                )
            except Exception as e:
                print(f"[build_registry] MongoDB write failed: {e}")

    def list_all(self) -> list:
        if self._collection is not None:
            try:
                docs = list(self._collection.find({}))
                for d in docs:
                    d.pop("_id", None)
                return docs
            except Exception:
                pass
        return list(self._memory.values())
