"""
capai.diagnostic_agent
=======================
Probes a task failure and produces a structured CapabilitySpec describing
the exact function that needs to be built.

Uses the unified llm_client (Groq or Anthropic). Falls back to a
deterministic heuristic when no LLM key is configured.
"""
from __future__ import annotations

import json
from typing import Optional

from . import config
from . import llm_client
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
  "name": MUST be exactly "{name}" — do not rename, do not add suffixes
  "description": one sentence describing what the function does
  "signature": must start with "def {name}(" — use the exact name
  "example_inputs": a list of 2-3 example argument lists that should work
  "expected_behavior": one or two sentences describing correct behaviour
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
    def diagnose(self, task: Task, error: Optional[Exception] = None) -> CapabilitySpec:
        if not config.LLM_ENABLED:
            return _heuristic_spec(task, error)

        error_block = f"Previous attempt raised: {type(error).__name__}: {error}\n" if error else ""
        prompt = _DIAGNOSIS_PROMPT.format(
            name=task.name, description=task.description,
            args=list(task.args), kwargs=task.kwargs, error_block=error_block,
        )
        try:
            text = llm_client.complete(prompt, max_tokens=600)
            # Strip markdown fences if model added them despite instructions
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            data = json.loads(text)
        except Exception:
            return _heuristic_spec(task, error)

        # Always use task.name — never let the LLM rename the function
        return CapabilitySpec(
            name=task.name,
            description=data.get("description", task.description),
            signature=data.get("signature", f"def {task.name}(*args):").replace(
                f"def {data.get('name', task.name)}(", f"def {task.name}("
            ),
            example_inputs=data.get("example_inputs", []),
            expected_behavior=data.get("expected_behavior", ""),
            root_cause=data.get("root_cause", ""),
        )
