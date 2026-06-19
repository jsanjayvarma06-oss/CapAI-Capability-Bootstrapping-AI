"""
capai.diagnostic_agent
=======================
Section 3.3 / 4.1 of the report: "rather than guessing at a fix from the
symptom of the failure, it probes the failure directly... to pin down
precisely what capability is missing."

In this prototype, "probing the failure" means looking at three things
together: the task's declared name and natural-language description, the
arguments the host agent actually tried to pass, and (if this is a retry)
the exception that came back the previous time. When an LLM key is
configured, Claude is asked to turn that into a structured CapabilitySpec;
otherwise a deterministic heuristic builds a reasonable spec directly from
the task, which is enough to keep the rest of the loop runnable offline.
"""
from __future__ import annotations

import json
from typing import Optional

from . import config
from .models import CapabilitySpec, Task

_DIAGNOSIS_PROMPT = """You are the Diagnostic Agent inside CapAI, a self-expanding AI capability \
framework. A host agent just failed to complete a task because it has no matching capability. \
Your only job is root-cause analysis: figure out exactly what function needs to be built.

Task name: {name}
Task description: {description}
Arguments the host agent tried to pass: {args}
Keyword arguments: {kwargs}
{error_block}
Respond with ONLY a JSON object (no markdown fences, no commentary) with these keys:
  "name": a valid Python identifier for the function (snake_case)
  "description": one sentence describing what the function does
  "signature": a single-line Python function signature with type hints, e.g.
               "def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:"
  "example_inputs": a list of 2-3 example argument lists that should work
  "expected_behavior": one or two sentences describing correct behaviour, including
                        how it should handle invalid input
  "root_cause": one sentence on why the original task failed
"""


def _heuristic_spec(task: Task, error: Optional[Exception]) -> CapabilitySpec:
    """Deterministic, LLM-free fallback so the loop is runnable with zero setup."""
    arg_names = [f"arg{i}" for i in range(len(task.args))] or ["*args"]
    signature = f"def {task.name}({', '.join(arg_names)}):"
    root_cause = (
        f"No capability named '{task.name}' exists in the registry."
        if error is None else
        f"'{task.name}' was attempted but raised {type(error).__name__}: {error}"
    )
    return CapabilitySpec(
        name=task.name,
        description=task.description,
        signature=signature,
        example_inputs=[list(task.args)] if task.args else [],
        expected_behavior=f"Should fulfil: {task.description}",
        root_cause=root_cause,
    )


class DiagnosticAgent:
    def __init__(self, client=None, model: str = config.ANTHROPIC_MODEL):
        self._client = client
        self.model = model

    def diagnose(self, task: Task, error: Optional[Exception] = None) -> CapabilitySpec:
        if not config.LLM_ENABLED:
            return _heuristic_spec(task, error)
        return self._diagnose_with_llm(task, error)

    def _diagnose_with_llm(self, task: Task, error: Optional[Exception]) -> CapabilitySpec:
        import anthropic
        client = self._client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        error_block = f"Previous attempt raised: {type(error).__name__}: {error}\n" if error else ""
        prompt = _DIAGNOSIS_PROMPT.format(
            name=task.name, description=task.description,
            args=list(task.args), kwargs=task.kwargs, error_block=error_block,
        )
        response = client.messages.create(
            model=self.model, max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Model didn't follow the format; fall back rather than crash the loop.
            return _heuristic_spec(task, error)
        return CapabilitySpec(
            name=data.get("name", task.name),
            description=data.get("description", task.description),
            signature=data.get("signature", f"def {task.name}(*args):"),
            example_inputs=data.get("example_inputs", []),
            expected_behavior=data.get("expected_behavior", ""),
            root_cause=data.get("root_cause", ""),
        )
