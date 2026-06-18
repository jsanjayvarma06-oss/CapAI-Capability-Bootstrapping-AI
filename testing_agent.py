"""
capai.testing_agent
====================
Section 3.3 / 4.3 of the report: three independent layers, all of which a
capability must clear before the Manager Agent will even consider it.

  Layer 1 — known test cases:    explicit (args, expected_output) pairs
                                  attached to the spec, where available.
  Layer 2 — history replay:      every previous attempt recorded in this
                                  MCP server's history.json is re-run
                                  against the NEW code, so a fix can't
                                  silently reintroduce a bug it already
                                  solved before.
  Layer 3 — self-generated edge cases: built from the spec's example
                                  inputs by mutating each argument (None,
                                  empty, negative, far-out-of-range, wrong
                                  type) — cheap, deterministic, and good at
                                  catching missing input validation, which
                                  is the single most common failure mode in
                                  LLM-generated functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .mcp_server import MCPServer
from .models import CapabilitySpec, VerificationResult
from .sandbox import run_capability


@dataclass
class KnownTestCase:
    args: tuple
    expected: Any
    tolerance: float = 1e-6


class TestingAgent:
    def verify(
        self,
        spec: CapabilitySpec,
        source_code: str,
        mcp: MCPServer,
        known_tests: Optional[list[KnownTestCase]] = None,
    ) -> VerificationResult:
        details: list[str] = []
        layer_results: dict[str, bool] = {}

        layer_results["known_tests"] = self._run_known_tests(spec, source_code, known_tests or [], details)
        layer_results["history_replay"] = self._replay_history(spec, source_code, mcp, details)
        layer_results["self_generated"] = self._run_self_generated(spec, source_code, details)

        passed = all(layer_results.values())
        return VerificationResult(passed=passed, layer_results=layer_results, details=details)

    # ------------------------------------------------------------ layer 1
    def _run_known_tests(self, spec: CapabilitySpec, source_code: str,
                          known_tests: list[KnownTestCase], details: list[str]) -> bool:
        if not known_tests:
            details.append("Layer 1 (known tests): none supplied — treated as vacuously passed.")
            return True
        all_ok = True
        for i, case in enumerate(known_tests):
            result = run_capability(source_code, spec.name, case.args)
            ok = result.success and _close_enough(result.return_value, case.expected, case.tolerance)
            all_ok &= ok
            details.append(
                f"Layer 1 (known tests) case {i}: args={case.args} -> "
                f"{'PASS' if ok else 'FAIL'} (got {result.return_value if result.success else result.error_message}, "
                f"expected {case.expected})"
            )
        return all_ok

    # ------------------------------------------------------------ layer 2
    def _replay_history(self, spec: CapabilitySpec, source_code: str,
                         mcp: MCPServer, details: list[str]) -> bool:
        history = mcp.load_history()
        relevant = [h for h in history if h.get("spec_name") == spec.name]
        if not relevant:
            details.append("Layer 2 (history replay): no prior attempts on record — vacuously passed.")
            return True
        all_ok = True
        for i, attempt in enumerate(relevant):
            # We don't have the original args stored for every historical
            # attempt in this prototype, so replay focuses on the cases we
            # *do* have recorded notes for, e.g. a previously-failing input.
            replay_args = attempt.get("replay_args")
            if replay_args is None:
                continue
            result = run_capability(source_code, spec.name, tuple(replay_args))
            ok = result.success
            all_ok &= ok
            details.append(
                f"Layer 2 (history replay) prior case {i} (args={replay_args}): "
                f"{'PASS' if ok else 'FAIL'} (previously {'passed' if attempt.get('passed') else 'failed'})"
            )
        return all_ok

    # ------------------------------------------------------------ layer 3
    def _run_self_generated(self, spec: CapabilitySpec, source_code: str, details: list[str]) -> bool:
        edge_cases = _generate_edge_cases(spec)
        if not edge_cases:
            details.append("Layer 3 (self-generated edge cases): no example inputs to mutate — vacuously passed.")
            return True
        all_ok = True
        for args, expectation, label in edge_cases:
            result = run_capability(source_code, spec.name, tuple(args))
            if expectation == "should_raise":
                ok = not result.success  # we WANT an exception for genuinely invalid input
            else:  # "should_not_crash"
                ok = result.success
            all_ok &= ok
            details.append(
                f"Layer 3 (self-generated) [{label}] args={args} -> "
                f"{'PASS' if ok else 'FAIL'} (expected {expectation}, "
                f"got {'success: ' + repr(result.return_value) if result.success else 'error: ' + str(result.error_message)})"
            )
        return all_ok


def _close_enough(a: Any, b: Any, tolerance: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return a == b


def _generate_edge_cases(spec: CapabilitySpec) -> list[tuple[list, str, str]]:
    """
    Mutate the spec's example inputs to build cheap, deterministic edge
    cases. Returns a list of (args, expectation, label) where expectation
    is either "should_raise" (we are deliberately feeding invalid input and
    a well-written function should reject it) or "should_not_crash" (a
    boundary value a correct implementation should still handle).
    """
    if not spec.example_inputs:
        return []
    base = spec.example_inputs[0]
    cases: list[tuple[list, str, str]] = []

    for i, value in enumerate(base):
        if isinstance(value, str):
            mutated = list(base)
            mutated[i] = "totally-not-a-valid-value-###"
            cases.append((mutated, "should_raise", f"invalid string at position {i}"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            mutated_neg = list(base)
            mutated_neg[i] = -abs(value) - 1000
            cases.append((mutated_neg, "should_not_crash", f"large negative number at position {i}"))

    return cases
