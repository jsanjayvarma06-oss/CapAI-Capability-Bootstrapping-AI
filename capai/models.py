"""
capai.models
=============
Every dataclass shared across the agents. Nothing in here has behaviour
beyond simple helpers — the agents (Diagnostic, CodeWriter, Testing,
Manager, Orchestrator) own all the logic; this module just owns the
shapes they pass to each other.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Task:
    """A unit of work a host application asked CapAI to perform."""
    name: str
    description: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


@dataclass
class CapabilitySpec:
    """
    The Diagnostic Agent's structured output: exactly what needs to be
    built, never a vague guess. This is the contract between the
    Diagnostic Agent and the Code Writer.
    """
    name: str
    description: str
    signature: str
    example_inputs: list = field(default_factory=list)
    expected_behavior: str = ""
    root_cause: str = ""


@dataclass
class VerificationResult:
    """What the Testing Agent hands back after running all three layers."""
    passed: bool
    layer_results: dict
    details: list = field(default_factory=list)


@dataclass
class Capability:
    """
    A capability as tracked once source code exists for it. `approved`
    and `retired` are only ever set by the Manager Agent — see Section
    3.3 / 4.2 of the report: "Nothing reaches CapabilityRegistry except
    through this class."
    """
    name: str
    description: str
    source_code: str
    spec: CapabilitySpec
    mcp_id: str
    version: str = "0.0.1"
    verified: bool = False
    approved: bool = False
    retired: bool = False
    verification: Optional[dict] = None


def new_mcp_id(capability_name: str) -> str:
    """Stable, readable, collision-resistant id for a new MCP server's workdir."""
    slug = re.sub(r"[^a-z0-9]+", "_", capability_name.lower()).strip("_") or "capability"
    return f"{slug}__{int(time.time())}_{uuid.uuid4().hex[:6]}"
