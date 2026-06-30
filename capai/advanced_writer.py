"""
capai.advanced_writer
=======================
Handles bigger asks than a single utility function: multi-function
modules, classes, small algorithms, and multi-step pipelines that chain
existing capabilities together.

Pipeline (one-shot, no human-in-the-loop refinement):

  1. WRITE      — Groq writes the module/class/pipeline code
  2. AUTO-TEST  — Groq writes a separate assert-based test script for it
  3. RUN        — both run together in an isolated subprocess sandbox
  4. CRITIQUE   — if tests fail, Groq is shown the code + error and asked
                  to both diagnose and rewrite in a single pass
  5. RETRY      — steps 3-4 repeat up to CAPAI_ADVANCED_MAX_ITERATIONS
                  times before giving up

This is intentionally heavier than code_writer.py's heuristic-first path
— advanced_writer.py is for requests that look like more than one
function, e.g. "build a REST client class", "write a small ETL
pipeline that reads, validates, and aggregates records", or "implement
a binary search tree with insert, delete, and traversal".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import config
from . import llm_client
from .sandbox import run_module_with_tests

MAX_ITERATIONS = int(__import__("os").environ.get("CAPAI_ADVANCED_MAX_ITERATIONS", "3"))


@dataclass
class AdvancedBuildResult:
    success: bool
    code: str = ""
    test_code: str = ""
    iterations: int = 0
    log: list = None
    error: str = ""

    def __post_init__(self):
        if self.log is None:
            self.log = []


_WRITE_PROMPT = """\
You are a senior Python engineer. Write production-quality Python code for this request:

{description}

Requirements:
- Standard library only, no third-party imports unless explicitly requested.
- Include type hints on all public functions/methods.
- Validate inputs and raise clear exceptions (ValueError, TypeError) on bad input.
- If this involves multiple related operations, organise them into a class.
- If this involves a sequence of steps (a pipeline), implement each step as its
  own function/method plus one orchestrating function/method that runs them in order.
- No I/O side effects (no file writes, no network calls, no printing) unless the
  request explicitly asks for them.
- No explanations, no markdown fences, no usage examples outside the code itself.
  Return ONLY the Python source code.
"""

_TEST_PROMPT = """\
You are a QA engineer. Below is a Python module. Write a short test script that
uses plain `assert` statements (no unittest, no pytest) to verify at least 4
distinct behaviours: 2-3 normal/expected cases and 1-2 edge cases (invalid
input, boundary values, empty input — whatever is relevant to this code).

The test script will be executed via exec() in the SAME namespace as the
module below, so call the functions/classes directly by name — do not import
anything, do not redefine the module.

MODULE:
```python
{code}
```

Return ONLY the test script source code. No markdown fences, no commentary.
Every assertion must be a plain `assert <condition>, "<message>"` statement.
"""

_CRITIQUE_PROMPT = """\
The following Python module failed its test suite. Diagnose the bug and
return a CORRECTED full version of the module — not a diff, not an
explanation, the complete fixed source code only.

ORIGINAL REQUEST:
{description}

MODULE (has a bug):
```python
{code}
```

TEST SCRIPT THAT FAILED:
```python
{test_code}
```

FAILURE:
{error}

Return ONLY the corrected Python module source code. No markdown fences,
no commentary, no test code in your response — just the fixed module.
"""


def _strip_fences(text: str) -> str:
    match = re.match(r"^```(?:python)?\s*\n(.*?)\n```\s*$", text.strip(), re.DOTALL)
    return match.group(1) if match else text.strip()


def build(description: str, max_iterations: Optional[int] = None) -> AdvancedBuildResult:
    """
    One-shot build of a multi-function module, class, or pipeline from a
    plain-English description. Returns the final code (verified against
    self-generated tests) or the best attempt plus the failure log.
    """
    if not config.LLM_ENABLED:
        return AdvancedBuildResult(
            success=False,
            error="No LLM configured (set GROQ_API_KEY or ANTHROPIC_API_KEY) — "
                  "advanced_writer requires an LLM, there is no offline heuristic path.",
        )

    iterations = max_iterations or MAX_ITERATIONS
    log: list = []

    # Step 1 — write the module
    code = _strip_fences(llm_client.complete(_WRITE_PROMPT.format(description=description), max_tokens=2000))
    log.append(f"Wrote initial module ({len(code.splitlines())} lines).")

    # Step 2 — write tests for it
    test_code = _strip_fences(
        llm_client.complete(_TEST_PROMPT.format(code=code), max_tokens=1000)
    )
    log.append(f"Generated test script ({len(test_code.splitlines())} lines).")

    for attempt in range(1, iterations + 1):
        result = run_module_with_tests(code, test_code)

        if result.success:
            log.append(f"Attempt {attempt}: all tests passed.")
            return AdvancedBuildResult(
                success=True, code=code, test_code=test_code,
                iterations=attempt, log=log,
            )

        log.append(f"Attempt {attempt} failed: {result.error_message[:200]}")

        if attempt == iterations:
            break

        # Step 4 — critique and fix in one combined pass
        fixed = llm_client.complete(
            _CRITIQUE_PROMPT.format(
                description=description, code=code,
                test_code=test_code, error=result.error_message,
            ),
            max_tokens=2000,
        )
        code = _strip_fences(fixed)
        log.append(f"Rewrote module after attempt {attempt} failure.")

    return AdvancedBuildResult(
        success=False, code=code, test_code=test_code,
        iterations=iterations, log=log,
        error=f"Failed to produce passing code after {iterations} attempts. "
              f"Last error: {log[-2] if len(log) > 1 else 'unknown'}",
    )
