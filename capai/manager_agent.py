"""
capai.manager_agent
====================
Section 3.3 / 4.2 of the report: the only component with authority to
promote a capability into the Main Registry, and the one responsible for
keeping the registry from accumulating duplicates or dead weight over
time. Nothing reaches `CapabilityRegistry` except through this class.
"""
from __future__ import annotations

import difflib

from .models import Capability, VerificationResult
from .registry import CapabilityRegistry


class ManagerAgent:
    def __init__(self, registry: CapabilityRegistry, similarity_threshold: float = 0.85):
        self.registry = registry
        self.similarity_threshold = similarity_threshold

    def review_and_promote(self, capability: Capability, verification: VerificationResult) -> bool:
        """
        Approve a capability only if it passed verification AND isn't a
        near-duplicate of something already active. Returns True if the
        capability is now live in the Main Registry.
        """
        if not verification.passed:
            return False

        capability.verified = True
        capability.verification = {
            "passed": verification.passed,
            "layer_results": verification.layer_results,
            "details": verification.details,
        }

        duplicate = self._find_near_duplicate(capability)
        if duplicate is not None:
            # Same job, different name — merge rather than bloat the registry
            # with two capabilities that do the same thing.
            self.registry.merge(keep_name=duplicate.name, redundant_name=capability.name)
            capability.approved = False
            return False

        capability.approved = True
        self.registry.add(capability)
        return True

    def retire(self, name: str) -> bool:
        """Explicit lifecycle action: pull a stale capability out of active service."""
        return self.registry.retire(name)

    def _find_near_duplicate(self, capability: Capability) -> Capability | None:
        for existing in self.registry.list_active():
            if existing.name == capability.name:
                continue  # same-name updates are handled as versions, not duplicates
            ratio = difflib.SequenceMatcher(
                None, existing.description.lower(), capability.description.lower()
            ).ratio()
            if ratio >= self.similarity_threshold:
                return existing
        return None
