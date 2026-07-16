"""
capai.capability_sync
========================
Multi-Agent Capability Sync (research extension #4): allows
independent CapAI servers — each with their own MongoDB, each having
built different capabilities — to exchange verified capabilities
without rebuilding them from scratch.

IMPORTANT — scope and honesty about what this is and is not:
This module implements the CLIENT-SIDE sync protocol and conflict-
resolution policy: fetching a peer's capability manifest, detecting
overlaps via capability_matcher (exact name match AND fuzzy description
match), and deciding what to merge. It does NOT implement:
  - network transport security (auth, TLS pinning, rate limiting) —
    assume this sits behind the same infrastructure as the rest of the
    CapAI REST API in production;
  - distributed consensus in the formal sense (no Raft/Paxos, no
    guarantee of global convergence order across N>2 servers syncing
    concurrently) — this is a pairwise, pull-based, eventually-
    consistent merge, appropriate for a small number of trusted peers,
    not a Byzantine-fault-tolerant protocol for adversarial nodes;
  - automatic bidirectional real-time sync — sync is triggered
    explicitly (e.g. by a cron job or manual call), not push-based.

Conflict resolution policy (the actual research question here): when
both servers have a capability under the same name with DIFFERENT
source code, which one wins? This module resolves ties using the
existing confidence score (Section III-B of the CapAI paper) — the
higher-confidence version is kept, and the loser is retained under a
suffixed name (e.g. `is_prime__peer_v2`) rather than silently discarded,
so a human can review and reconcile if the scores were close.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from capability_matcher import similarity_score


@dataclass
class RemoteCapability:
    """What a peer server reports about one of its capabilities."""
    name: str
    description: str
    source_code: str
    confidence: int
    coverage_percent: float
    origin_server: str


@dataclass
class SyncResult:
    added: list = field(default_factory=list)          # genuinely new capabilities pulled in
    skipped_duplicate: list = field(default_factory=list)  # exact/near-identical, nothing to do
    conflicts_resolved: list = field(default_factory=list)  # same name, different code, one won
    fuzzy_matches_flagged: list = field(default_factory=list)  # similar description, different name — flagged not merged


class LocalRegistry:
    """
    Minimal in-memory stand-in for CapAI's real MongoDB-backed registry,
    used here so the sync protocol is independently testable without a
    live database. In production, `capabilities` would be backed by
    the same MongoDB collection api.py already uses.
    """
    def __init__(self):
        self.capabilities: dict = {}  # name -> RemoteCapability-like dict

    def get(self, name: str) -> Optional[dict]:
        return self.capabilities.get(name)

    def set(self, name: str, entry: dict) -> None:
        self.capabilities[name] = entry

    def all(self) -> list:
        return list(self.capabilities.values())


class CapabilitySyncClient:
    """
    Pulls capabilities from a peer's manifest into a local registry,
    applying deduplication and conflict resolution. `fetch_peer_manifest`
    is injected so this is testable without real network calls; in
    production it would be a GET to the peer's /capabilities/export
    endpoint (a natural addition to api.py alongside the existing
    /capabilities endpoint).
    """

    def __init__(self, local_registry: LocalRegistry, fuzzy_threshold: float = 0.6):
        self.local = local_registry
        self.fuzzy_threshold = fuzzy_threshold

    def sync_from(self, peer_capabilities: list) -> SyncResult:
        result = SyncResult()

        for remote in peer_capabilities:
            existing = self.local.get(remote.name)

            if existing is None:
                # check for a fuzzy (different-name, similar-meaning) match first —
                # this is the intersection with Capability Generalization (#1):
                # don't blindly add a near-duplicate under a different name
                fuzzy_hit = self._find_fuzzy_match(remote)
                if fuzzy_hit:
                    result.fuzzy_matches_flagged.append({
                        "remote_name": remote.name,
                        "local_name": fuzzy_hit,
                        "similarity": similarity_score(remote.description, self.local.get(fuzzy_hit)["description"]),
                        "action": "flagged_for_review — not auto-merged",
                    })
                    continue

                # genuinely new — add it
                self.local.set(remote.name, {
                    "name": remote.name, "description": remote.description,
                    "source_code": remote.source_code, "confidence": remote.confidence,
                    "coverage_percent": remote.coverage_percent,
                    "origin_server": remote.origin_server, "synced_at": time.time(),
                })
                result.added.append(remote.name)
                continue

            # same name already exists locally
            if existing["source_code"] == remote.source_code:
                result.skipped_duplicate.append(remote.name)
                continue

            # same name, DIFFERENT code — conflict. Higher confidence wins.
            if remote.confidence > existing["confidence"]:
                loser_name = f"{remote.name}__local_v_conf{existing['confidence']}"
                self.local.set(loser_name, existing)
                self.local.set(remote.name, {
                    "name": remote.name, "description": remote.description,
                    "source_code": remote.source_code, "confidence": remote.confidence,
                    "coverage_percent": remote.coverage_percent,
                    "origin_server": remote.origin_server, "synced_at": time.time(),
                })
                result.conflicts_resolved.append({
                    "name": remote.name, "winner": "remote", "winner_confidence": remote.confidence,
                    "loser_confidence": existing["confidence"], "loser_retained_as": loser_name,
                })
            else:
                loser_name = f"{remote.name}__peer_{remote.origin_server}_conf{remote.confidence}"
                self.local.set(loser_name, {
                    "name": loser_name, "description": remote.description,
                    "source_code": remote.source_code, "confidence": remote.confidence,
                    "coverage_percent": remote.coverage_percent,
                    "origin_server": remote.origin_server, "synced_at": time.time(),
                })
                result.conflicts_resolved.append({
                    "name": remote.name, "winner": "local", "winner_confidence": existing["confidence"],
                    "loser_confidence": remote.confidence, "loser_retained_as": loser_name,
                })

        return result

    def _find_fuzzy_match(self, remote: RemoteCapability) -> Optional[str]:
        for local_cap in self.local.all():
            if local_cap["name"] == remote.name:
                continue
            score = similarity_score(remote.description, local_cap["description"])
            if score >= self.fuzzy_threshold:
                return local_cap["name"]
        return None
