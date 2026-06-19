"""
capai.orchestrator
====================
Section 3.3 / 4 of the report: the one component every call goes through.
On a registry hit it executes directly. On a miss it runs the full
acquisition loop — spin up an isolated MCP server, diagnose the gap,
write a candidate module, verify it through three layers, get the
Manager Agent's approval, then retry the original task — for up to
config.MAX_ACQUISITION_ATTEMPTS rounds before giving up and raising back
to the caller.
"""
from __future__ import annotations

from typing import Callable, Optional

from . import config
from .code_writer import CodeWriter
from .diagnostic_agent import DiagnosticAgent
from .exceptions import CapabilityAcquisitionError
from .manager_agent import ManagerAgent
from .mcp_server import MCPServer
from .models import Capability, Task
from .registry import CapabilityRegistry
from .sandbox import run_capability
from .testing_agent import TestingAgent


class AcquisitionEvent:
    """
    A small structured log entry emitted at every stage of the loop, so a
    host application (or capai/demo.py) can show a live trace without
    needing to know anything about CapAI's internals.
    """

    def __init__(self, stage: str, message: str, **data):
        self.stage = stage
        self.message = message
        self.data = data

    def __repr__(self) -> str:
        return f"[{self.stage}] {self.message}"


class Orchestrator:
    def __init__(
        self,
        registry: Optional[CapabilityRegistry] = None,
        diagnostic_agent: Optional[DiagnosticAgent] = None,
        code_writer: Optional[CodeWriter] = None,
        testing_agent: Optional[TestingAgent] = None,
        manager_agent: Optional[ManagerAgent] = None,
        on_event: Optional[Callable[[AcquisitionEvent], None]] = None,
    ):
        self.registry = registry or CapabilityRegistry()
        self.diagnostic_agent = diagnostic_agent or DiagnosticAgent()
        self.code_writer = code_writer or CodeWriter()
        self.testing_agent = testing_agent or TestingAgent()
        self.manager_agent = manager_agent or ManagerAgent(self.registry)
        self._on_event = on_event or (lambda event: None)

    def run(self, task: Task):
        """Public entry point: execute `task`, acquiring whatever capability is missing first."""
        self._emit("task_received", f"Task '{task.name}' received.")

        if self.registry.has(task.name):
            self._emit("registry_hit", f"'{task.name}' already in registry — executing directly.")
            return self._execute_registered(task)

        self._emit("registry_miss", f"'{task.name}' not found in registry — starting acquisition loop.")
        return self._acquire_and_run(task)

    # ------------------------------------------------------------ internals
    def _execute_registered(self, task: Task):
        capability = self.registry.get(task.name)
        result = run_capability(capability.source_code, capability.name, task.args)
        if not result.success:
            # A previously-working capability just failed on a new input.
            # Treat this exactly like a fresh gap rather than crashing the host.
            self._emit(
                "regression_detected",
                f"'{task.name}' failed on new input ({result.error_message}); re-running acquisition.",
            )
            return self._acquire_and_run(task, last_error=RuntimeError(result.error_message))
        return result.return_value

    def _acquire_and_run(self, task: Task, last_error: Exception = None):
        mcp = MCPServer(capability_name=task.name)
        self._emit(
            "mcp_created", f"Created isolated MCP server '{mcp.id}' for '{task.name}'.", mcp_id=mcp.id
        )

        error = last_error
        for attempt in range(1, config.MAX_ACQUISITION_ATTEMPTS + 1):
            self._emit("diagnosing", f"Attempt {attempt}: Diagnostic Agent analysing the gap.")
            spec = self.diagnostic_agent.diagnose(task, error)
            self._emit("diagnosed", f"Root cause: {spec.root_cause}", spec=spec)

            self._emit("writing_code", f"Code Writer generating '{spec.name}'.")
            source_code = self.code_writer.write(spec)
            mcp.commit_module(source_code, message=f"attempt {attempt}: draft for {spec.name}")

            self._emit("testing", "Testing Agent running 3-layer verification.")
            verification = self.testing_agent.verify(spec, source_code, mcp)
            mcp.record_attempt(
                spec, source_code, passed=verification.passed, notes="; ".join(verification.details[-3:])
            )
            self._emit(
                "tested",
                f"Verification {'PASSED' if verification.passed else 'FAILED'}.",
                layer_results=verification.layer_results,
            )

            if not verification.passed:
                error = RuntimeError("; ".join(verification.details))
                continue

            capability = Capability(
                name=spec.name,
                description=spec.description,
                source_code=source_code,
                spec=spec,
                mcp_id=mcp.id,
            )
            promoted = self.manager_agent.review_and_promote(capability, verification)
            if not promoted:
                self._emit(
                    "merged_or_rejected",
                    f"Manager Agent did not promote '{spec.name}' as a new capability "
                    "(likely merged into an existing near-duplicate).",
                )
                if self.registry.has(task.name):
                    return self._execute_registered(task)
                error = RuntimeError("Capability rejected by Manager Agent and no equivalent exists.")
                continue

            self._emit("promoted", f"'{spec.name}' is now live in the Main Registry.")
            return self._execute_registered(task)

        self._emit(
            "acquisition_failed",
            f"Could not acquire a working capability for '{task.name}' "
            f"after {config.MAX_ACQUISITION_ATTEMPTS} attempts.",
        )
        raise CapabilityAcquisitionError(
            f"CapAI could not build a working capability for '{task.name}' after "
            f"{config.MAX_ACQUISITION_ATTEMPTS} attempts. Last error: {error}"
        )

    def _emit(self, stage: str, message: str, **data) -> None:
        self._on_event(AcquisitionEvent(stage, message, **data))
