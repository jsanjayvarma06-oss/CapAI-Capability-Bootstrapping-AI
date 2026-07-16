"""
capai.capability_evolution
=============================
Automatic Capability Evolution (research extension #2): versions
capabilities, tracks failure signals against them, and triggers a
verified rewrite-and-promote cycle when evidence accumulates that a
capability is wrong — WITHOUT ever silently replacing a working version
in place.

Design principle driving every choice below: the paper's central
finding (Section V-A of the CapAI paper) is that a single unverified
promotion, once cached, is served with permanent false confidence.
Auto-evolution is exactly the mechanism that could make this WORSE if
built carelessly (a "self-healing" system that silently swaps in a
new, equally-unverified version on every failure report is just moving
the same risk one level up). This module therefore enforces three
hard rules:

  1. A new version is NEVER promoted to "active" without passing the
     same sandboxed test suite CapAI already uses for first-time
     synthesis (Section III-B of the paper) — evolution reuses
     verification, it does not bypass it.
  2. Every prior active version is retained, never deleted, so a bad
     v2 can be rolled back to v1 instantly and losslessly.
  3. Evolution triggers only after a CONFIGURABLE THRESHOLD of
     independent failure reports (default 3), not on a single report —
     a single wrong bug report should not be able to trigger a rewrite
     cycle by itself.

This module implements the version/failure/trigger state machine. It
does NOT itself call an LLM to rewrite code — it is designed to be
wired to CapAI's existing advanced_writer.build() (or code_writer) via
the `rewrite_fn` callback, so the actual code-generation and testing
logic is not duplicated, only the versioning/triggering policy is new.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class VersionStatus(Enum):
    ACTIVE = "active"        # currently served to callers
    SUPERSEDED = "superseded"  # was active, replaced by a later verified version
    REJECTED = "rejected"     # a rewrite attempt that failed verification


@dataclass
class CapabilityVersion:
    version: int
    source_code: str
    status: VersionStatus
    created_at: float
    confidence: int = 0
    coverage_percent: float = 0.0
    promoted_reason: str = ""  # "initial_synthesis" | "evolution:<failure_count> failures"


@dataclass
class FailureReport:
    reported_at: float
    reason: str            # e.g. "wrong output for input X", "type mismatch"
    reporter: str = "unknown"  # e.g. "user", "periodic_reverification", capability name of caller


@dataclass
class EvolvingCapability:
    name: str
    versions: list = field(default_factory=list)     # list[CapabilityVersion], index 0 = v1
    failure_reports: list = field(default_factory=list)  # list[FailureReport] against the CURRENT active version

    @property
    def active_version(self) -> Optional[CapabilityVersion]:
        for v in reversed(self.versions):
            if v.status == VersionStatus.ACTIVE:
                return v
        return None

    def report_failure(self, reason: str, reporter: str = "unknown") -> None:
        self.failure_reports.append(FailureReport(time.time(), reason, reporter))


class EvolutionEngine:
    """
    Orchestrates the observe -> rewrite -> verify -> promote cycle for
    a set of capabilities. `rewrite_fn` and `test_fn` are injected
    dependencies so this class contains zero LLM-specific or sandbox-
    specific logic itself — in production these would be
    advanced_writer.build's internal write/test steps; here they are
    injected so the state machine is independently testable.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        rewrite_fn: Optional[Callable[[str, str, list], str]] = None,
        test_fn: Optional[Callable[[str, str], tuple]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.rewrite_fn = rewrite_fn   # (name, description, failure_reasons) -> new_source_code
        self.test_fn = test_fn         # (name, source_code) -> (passed: bool, confidence: int, coverage: float)
        self.capabilities: dict = {}   # name -> EvolvingCapability

    def register_initial(self, name: str, source_code: str, confidence: int, coverage: float) -> EvolvingCapability:
        cap = EvolvingCapability(name=name)
        cap.versions.append(CapabilityVersion(
            version=1, source_code=source_code, status=VersionStatus.ACTIVE,
            created_at=time.time(), confidence=confidence, coverage_percent=coverage,
            promoted_reason="initial_synthesis",
        ))
        self.capabilities[name] = cap
        return cap

    def report_failure(self, name: str, reason: str, reporter: str = "unknown") -> dict:
        """
        Record a failure against a capability's currently active version.
        Returns a dict describing what happened: whether evolution was
        triggered, and the outcome if so.
        """
        if name not in self.capabilities:
            raise KeyError(f"Unknown capability: {name}")

        cap = self.capabilities[name]
        cap.report_failure(reason, reporter)

        if len(cap.failure_reports) < self.failure_threshold:
            return {
                "triggered": False,
                "failure_count": len(cap.failure_reports),
                "threshold": self.failure_threshold,
                "message": f"{len(cap.failure_reports)}/{self.failure_threshold} failures reported — below threshold, no action taken.",
            }

        return self._evolve(cap)

    def _evolve(self, cap: EvolvingCapability) -> dict:
        if self.rewrite_fn is None or self.test_fn is None:
            return {
                "triggered": True,
                "outcome": "no_rewrite_backend_configured",
                "message": "Failure threshold reached, but no rewrite_fn/test_fn was injected — "
                           "wire this to advanced_writer.build() to actually attempt a fix.",
            }

        current = cap.active_version
        failure_reasons = [f.reason for f in cap.failure_reports]

        new_code = self.rewrite_fn(cap.name, current.source_code, failure_reasons)
        passed, confidence, coverage = self.test_fn(cap.name, new_code)

        new_version_number = len(cap.versions) + 1

        if passed:
            current.status = VersionStatus.SUPERSEDED
            new_version = CapabilityVersion(
                version=new_version_number, source_code=new_code, status=VersionStatus.ACTIVE,
                created_at=time.time(), confidence=confidence, coverage_percent=coverage,
                promoted_reason=f"evolution:{len(cap.failure_reports)} failures",
            )
            cap.versions.append(new_version)
            cap.failure_reports = []  # reset counter against the new active version
            return {
                "triggered": True,
                "outcome": "promoted",
                "new_version": new_version_number,
                "confidence": confidence,
                "coverage_percent": coverage,
                "message": f"v{current.version} superseded by v{new_version_number} after "
                           f"{len(failure_reasons)} failure reports. v{current.version} retained for rollback.",
            }
        else:
            rejected_version = CapabilityVersion(
                version=new_version_number, source_code=new_code, status=VersionStatus.REJECTED,
                created_at=time.time(), promoted_reason=f"evolution_attempt:{len(cap.failure_reports)} failures",
            )
            cap.versions.append(rejected_version)
            # current active version is UNCHANGED — a failed rewrite never displaces a working version
            return {
                "triggered": True,
                "outcome": "rewrite_failed_verification",
                "message": f"Rewrite attempt for v{new_version_number} failed its own test suite — "
                           f"v{current.version} remains active. Manual review recommended.",
            }

    def rollback(self, name: str, to_version: int) -> dict:
        """Explicitly roll back to a specific prior version, e.g. after
        a bad promotion is noticed some other way (not via report_failure)."""
        cap = self.capabilities[name]
        target = next((v for v in cap.versions if v.version == to_version), None)
        if target is None:
            raise ValueError(f"No version {to_version} exists for '{name}'")
        if target.status == VersionStatus.REJECTED:
            raise ValueError(f"Cannot roll back to v{to_version} — it never passed verification")

        current = cap.active_version
        if current:
            current.status = VersionStatus.SUPERSEDED
        target.status = VersionStatus.ACTIVE
        cap.failure_reports = []
        return {"rolled_back_to": to_version, "message": f"'{name}' is now serving v{to_version}"}

    def history(self, name: str) -> list:
        return [
            {"version": v.version, "status": v.status.value, "confidence": v.confidence,
             "coverage_percent": v.coverage_percent, "reason": v.promoted_reason}
            for v in self.capabilities[name].versions
        ]
