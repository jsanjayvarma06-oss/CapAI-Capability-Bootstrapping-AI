"""capai.exceptions"""
from __future__ import annotations


class CapAIError(Exception):
    """Base class for every error CapAI raises deliberately."""


class CapabilityAcquisitionError(CapAIError):
    """
    Raised when the Orchestrator exhausts config.MAX_ACQUISITION_ATTEMPTS
    without producing a capability that passes verification. This is a
    real failure mode, not a bug: it means the Diagnostic Agent and Code
    Writer genuinely could not close the gap, most often because the LLM
    is disabled and the offline heuristic Code Writer doesn't recognise
    the requested task (see capai/code_writer.py).
    """
