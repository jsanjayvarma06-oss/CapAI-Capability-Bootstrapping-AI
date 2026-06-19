"""
capai
=====
Public entry point.

    from capai import CapAI

    ai = CapAI()                                   # offline mode, no API key needed
    ai = CapAI(base_model="claude-sonnet-4-6")      # uses Anthropic for diagnosis + code generation

    result = ai.run("celsius_to_fahrenheit", "Convert Celsius to Fahrenheit", 100)

`base_model` selects the model the Diagnostic Agent and Code Writer use
internally. CapAI is a capability-acquisition layer that sits in front of
whatever your own application calls — it does not proxy a "host" model's
conversation on your behalf.
"""
from __future__ import annotations

from typing import Callable, Optional

from . import config
from .code_writer import CodeWriter
from .diagnostic_agent import DiagnosticAgent
from .exceptions import CapAIError, CapabilityAcquisitionError
from .manager_agent import ManagerAgent
from .models import Capability, CapabilitySpec, Task, VerificationResult
from .orchestrator import AcquisitionEvent, Orchestrator
from .registry import CapabilityRegistry
from .testing_agent import TestingAgent

__all__ = [
    "CapAI",
    "Task",
    "Capability",
    "CapabilitySpec",
    "VerificationResult",
    "AcquisitionEvent",
    "CapAIError",
    "CapabilityAcquisitionError",
    "CapabilityRegistry",
]
__version__ = "0.1.0"


class CapAI:
    """The installable plugin surface. One instance owns one Main Registry."""

    def __init__(self, base_model: Optional[str] = None,
                 on_event: Optional[Callable[[AcquisitionEvent], None]] = None,
                 registry: Optional[CapabilityRegistry] = None):
        self._registry = registry or CapabilityRegistry()
        self._orchestrator = Orchestrator(
            registry=self._registry,
            diagnostic_agent=DiagnosticAgent(),
            code_writer=CodeWriter(),
            testing_agent=TestingAgent(),
            manager_agent=ManagerAgent(self._registry),
            on_event=on_event,
        )

    def run(self, name: str, description: str, *args, **kwargs):
        """
        Ask CapAI to perform `name`. Runs instantly if a matching
        capability is already registered; otherwise the acquisition loop
        builds, verifies, and registers it first, then runs it.
        """
        task = Task(name=name, description=description, args=args, kwargs=kwargs)
        return self._orchestrator.run(task)

    def capabilities(self) -> list[str]:
        """Names of every capability currently live in the Main Registry."""
        return [c.name for c in self._registry.list_active()]

    def has_capability(self, name: str) -> bool:
        return self._registry.has(name)

    @property
    def registry(self) -> CapabilityRegistry:
        return self._registry
