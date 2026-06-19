"""
capai.sharing
==============
Section 3.3.7 of the report: lets one CapAI instance reuse a capability
another instance already built and verified, instead of re-running the
whole acquisition loop from scratch.

This module implements the *behaviour* of that layer — capability
discovery plus the tiered trust model — as plain Python classes, so it
can be exercised and tested without standing up a real network service.
`CapabilityExchange` is deliberately the seam where a real transport
would go: swapping this for an actual GraphQL server (e.g. with
`strawberry-graphql` or `ariadne`) means writing resolvers that call
these same methods, not redesigning the trust logic.

Trust tiers, as specified in the report:
  - self-generated  -> full trust, used immediately (handled entirely by
                        Orchestrator/ManagerAgent; this module isn't
                        involved)
  - peer MCP / peer instance -> medium trust: re-verified from scratch
                        before being allowed anywhere near the receiving
                        instance's Main Registry (TrustedImporter, below)
  - external / unverified source -> zero trust: never imported
                        (there is deliberately no code path here that
                        imports a capability without re-verification)
"""
from __future__ import annotations

from typing import Optional

from .manager_agent import ManagerAgent
from .mcp_server import MCPServer
from .models import Capability
from .registry import CapabilityRegistry
from .testing_agent import TestingAgent


class CapabilityExchange:
    """Wraps one instance's registry as something a peer instance can query and import from."""

    def __init__(self, registry: CapabilityRegistry, owner_id: str):
        self.registry = registry
        self.owner_id = owner_id

    def list_offered(self) -> list[dict]:
        """
        The 'capability passport' for everything this instance is willing
        to share: enough metadata for a receiving instance to decide
        whether it's even worth importing, without handing over source
        code until it actually asks.
        """
        return [
            {
                "name": c.name,
                "description": c.description,
                "version": c.version,
                "owner_id": self.owner_id,
                "verification": c.verification,
            }
            for c in self.registry.list_active()
        ]

    def export_capability(self, name: str) -> Optional[Capability]:
        cap = self.registry.get(name)
        if cap is None or not cap.approved or cap.retired:
            return None
        return cap


class TrustedImporter:
    """
    Sits on the *receiving* instance. A capability arriving via
    `import_from` is medium trust by definition — it is independently
    re-verified in a brand-new MCP server on the receiving side before
    the Manager Agent is even asked to promote it. The exporting
    instance's own verdict is never taken on faith.
    """

    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
        self.manager_agent = ManagerAgent(registry)
        self.testing_agent = TestingAgent()

    def import_from(self, exchange: CapabilityExchange, name: str) -> bool:
        candidate = exchange.export_capability(name)
        if candidate is None:
            return False

        mcp = MCPServer(capability_name=candidate.name)
        mcp.commit_module(candidate.source_code, message=f"imported from {exchange.owner_id}")

        verification = self.testing_agent.verify(candidate.spec, candidate.source_code, mcp)
        mcp.record_attempt(candidate.spec, candidate.source_code, passed=verification.passed,
                            notes=f"cross-agent import from {exchange.owner_id}")
        if not verification.passed:
            return False

        candidate.mcp_id = mcp.id
        candidate.verification = {
            "passed": verification.passed,
            "layer_results": verification.layer_results,
            "details": verification.details,
            "imported_from": exchange.owner_id,
        }
        return self.manager_agent.review_and_promote(candidate, verification)
