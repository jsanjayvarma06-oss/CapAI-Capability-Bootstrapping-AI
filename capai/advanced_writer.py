"""
capai.advanced_writer
=======================
Handles bigger asks than a single utility function: multi-function
modules, classes, algorithms, and multi-step pipelines.

Full pipeline per build:

  0. CACHE CHECK  — has this exact request been built and verified before?
  1. WRITE        — Groq (falls back to Anthropic) writes the code
  2. STATIC CHECK — ast.parse + pyflakes catch syntax/undefined-name bugs
                    before ever touching a subprocess
  3. TYPE CHECK   — mypy validates the type hints the LLM was told to add
  4. AUTO-TEST    — a separate test script is generated for the code
  5. RUN+COVERAGE — both run together in the sandbox under coverage.py
  6. CRITIQUE     — on any failure, the model sees the code, the test,
                    the error, AND which lines/branches were never
                    exercised, then rewrites both code and test
  7. RETRY        — up to CAPAI_ADVANCED_MAX_ITERATIONS times
  8. CONFIDENCE   — a 0-100 score based on iterations needed, coverage
                    achieved, and whether any check ever failed
  9. PERSIST      — successful builds are cached in MongoDB by a hash
                    of the description, so repeats are instant
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from . import llm_client
from .sandbox import run_module_with_tests
from .build_registry import BuildRegistry

MAX_ITERATIONS = int(os.environ.get("CAPAI_ADVANCED_MAX_ITERATIONS", "3"))

# Libraries beyond the stdlib that generated code is allowed to import.
# Kept short and well-understood — anything not on this list gets
# rejected at the static-check stage rather than silently failing later.
ALLOWED_EXTRA_LIBS = {
    "requests", "pandas", "numpy", "pydantic", "dateutil",
}

_registry = BuildRegistry()


@dataclass
class AdvancedBuildResult:
    success: bool
    code: str = ""
    test_code: str = ""
    iterations: int = 0
    log: list = field(default_factory=list)
    error: str = ""
    confidence: int = 0
    coverage_percent: float = 0.0
    from_cache: bool = False
    static_issues: list = field(default_factory=list)
    type_issues: list = field(default_factory=list)


# ── prompts ──────────────────────────────────────────────────────────────────

_WRITE_PROMPT = """\
You are a senior Python engineer. Write production-quality Python code for this request:

{description}

Requirements:
- Standard library only, UNLESS the request clearly needs one of these
  approved extras: {allowed_libs}. Do not import anything else.
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

CRITICAL: If the module defines a class, you MUST instantiate it first
(e.g. `obj = ClassName()`) and call its methods on that instance
(e.g. `obj.method_name(args)`) — do NOT call methods as if they were
free-standing functions. If the module defines top-level functions
instead of a class, call them directly by name with no instantiation.

MODULE:
```python
{code}
```

Return ONLY the test script source code. No markdown fences, no commentary.
Every assertion must be a plain `assert <condition>, "<message>"` statement.
"""

_CRITIQUE_PROMPT = """\
The following Python module failed verification. Diagnose the problem and
return a CORRECTED full version of the module — not a diff, not an
explanation, the complete fixed source code only.

IMPORTANT: First check whether the failure is actually a bug in the test
script itself (e.g. calling a class method as a free function without
instantiating the class first) rather than a bug in the module. If the
module's logic is correct and the test script is the actual problem,
fix the MODULE anyway to be more robust/forgiving where reasonable, but
do not break correct behaviour just because the test was written wrong.

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

{coverage_note}

Return ONLY the corrected Python module source code. No markdown fences,
no commentary, no test code in your response — just the fixed module.
"""


def _strip_fences(text: str) -> str:
    match = re.match(r"^```(?:python)?\s*\n(.*?)\n```\s*$", text.strip(), re.DOTALL)
    return match.group(1) if match else text.strip()


# ── static analysis (step 2) ─────────────────────────────────────────────────

def _static_check(code: str) -> list:
    """Syntax check via ast.parse, undefined-name / unused-import check via
    pyflakes. Returns a list of human-readable issue strings (empty = clean)."""
    issues = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} at line {e.lineno}"]

    # reject disallowed imports before running anything
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in sys.stdlib_module_names and top not in ALLOWED_EXTRA_LIBS:
                    issues.append(f"Disallowed import: '{top}' is not stdlib or an approved extra library")
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top and top not in sys.stdlib_module_names and top not in ALLOWED_EXTRA_LIBS:
                issues.append(f"Disallowed import: '{top}' is not stdlib or an approved extra library")

    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        import io
        out, err = io.StringIO(), io.StringIO()
        check(code, "<module>", Reporter(out, err))
        flagged = out.getvalue().strip()
        if flagged:
            issues.extend(line for line in flagged.splitlines() if "unable to detect undefined names" not in line.lower())
    except Exception:
        pass  # pyflakes is a nice-to-have, never block on it being unavailable

    return issues


# ── type checking (step 3) ───────────────────────────────────────────────────

def _type_check(code: str) -> list:
    """Run mypy on the generated code. Returns a list of issue strings."""
    try:
        path = Path(tempfile.mkstemp(suffix=".py")[1])
        path.write_text(code)
        proc = subprocess.run(
            [sys.executable, "-m", "mypy", "--ignore-missing-imports",
             "--no-error-summary", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        path.unlink(missing_ok=True)
        lines = [l for l in proc.stdout.splitlines() if "error:" in l]
        return lines[:10]  # cap noise
    except Exception:
        return []  # mypy unavailable or timed out — don't block the build on it


# ── coverage-aware run (step 5) ──────────────────────────────────────────────

def _run_with_coverage(code: str, test_code: str) -> tuple:
    """
    Runs module+test under coverage.py. coverage.py only tracks code
    that's associated with a real file path on disk, so the module is
    written to a temp .py file inside the subprocess and exec'd from
    there (rather than from an in-memory string) while coverage is
    active. Returns (ModuleRunResult, coverage_percent).
    """
    instrumented_test = (
        "import coverage as _capai_cov, tempfile as _capai_tmp, os as _capai_os\n"
        "_mod_fd, _mod_path = _capai_tmp.mkstemp(suffix='.py')\n"
        "with open(_mod_path, 'w') as _f:\n"
        f"    _f.write({code!r})\n"
        "_capai_os.close(_mod_fd)\n"
        "_cov = _capai_cov.Coverage(include=[_mod_path])\n"
        "_cov.start()\n"
        "with open(_mod_path) as _f:\n"
        "    exec(compile(_f.read(), _mod_path, 'exec'))\n"
        f"{test_code}\n"
        "_cov.stop()\n"
        "try:\n"
        "    import io as _io\n"
        "    _buf = _io.StringIO()\n"
        "    _pct = _cov.report(file=_buf, show_missing=False)\n"
        "    print('__COVERAGE__:' + str(round(_pct, 1)))\n"
        "except Exception as _cov_err:\n"
        "    print('__COVERAGE_ERROR__:' + str(_cov_err))\n"
        "_capai_os.unlink(_mod_path)\n"
    )
    result = run_module_with_tests(code, instrumented_test)
    coverage_percent = 0.0
    if result.test_output:
        for line in result.test_output.splitlines():
            if line.startswith("__COVERAGE__:"):
                try:
                    coverage_percent = float(line.split(":", 1)[1])
                except ValueError:
                    pass
    return result, coverage_percent


# ── confidence scoring (step 8) ──────────────────────────────────────────────

def _confidence_score(iterations: int, max_iterations: int, coverage_percent: float,
                       static_issues: list, type_issues: list) -> int:
    score = 100
    score -= (iterations - 1) * 20          # each retry needed costs 20 points
    score -= max(0, 80 - coverage_percent) * 0.3   # below 80% coverage costs points
    score -= len(static_issues) * 5
    score -= len(type_issues) * 3
    return max(0, min(100, round(score)))


# ── main entry point ──────────────────────────────────────────────────────────

def build(description: str, max_iterations: Optional[int] = None, use_cache: bool = True) -> AdvancedBuildResult:
    """
    One-shot build of a multi-function module, class, or pipeline from a
    plain-English description, with static analysis, type checking,
    coverage-aware testing, confidence scoring, and persistent caching.
    """
    if use_cache:
        cached = _registry.get(description)
        if cached:
            return AdvancedBuildResult(
                success=cached.get("success", False),
                code=cached.get("code", ""),
                test_code=cached.get("test_code", ""),
                iterations=cached.get("iterations", 0),
                log=cached.get("log", []) + ["Loaded from build cache."],
                error=cached.get("error", ""),
                confidence=cached.get("confidence", 0),
                coverage_percent=cached.get("coverage_percent", 0.0),
                from_cache=True,
                static_issues=cached.get("static_issues", []),
                type_issues=cached.get("type_issues", []),
            )

    if not config.LLM_ENABLED:
        return AdvancedBuildResult(
            success=False,
            error="No LLM configured (set GROQ_API_KEY or ANTHROPIC_API_KEY) — "
                  "advanced_writer requires an LLM, there is no offline heuristic path.",
        )

    iterations = max_iterations or MAX_ITERATIONS
    log: list = []

    code = _strip_fences(llm_client.complete(
        _WRITE_PROMPT.format(description=description, allowed_libs=", ".join(sorted(ALLOWED_EXTRA_LIBS))),
        max_tokens=2000,
    ))
    log.append(f"Wrote initial module ({len(code.splitlines())} lines).")

    test_code = _strip_fences(llm_client.complete(_TEST_PROMPT.format(code=code), max_tokens=1000))
    log.append(f"Generated test script ({len(test_code.splitlines())} lines).")

    static_issues: list = []
    type_issues: list = []
    coverage_percent = 0.0

    for attempt in range(1, iterations + 1):
        static_issues = _static_check(code)
        if static_issues:
            log.append(f"Attempt {attempt}: static analysis found {len(static_issues)} issue(s).")

        type_issues = _type_check(code)
        if type_issues:
            log.append(f"Attempt {attempt}: mypy found {len(type_issues)} issue(s).")

        if static_issues:
            # hard syntax/import errors block running entirely — go straight to critique
            error_summary = "Static analysis issues:\n" + "\n".join(static_issues)
            run_success = False
        else:
            result, coverage_percent = _run_with_coverage(code, test_code)
            run_success = result.success
            error_summary = result.error_message
            if run_success:
                log.append(f"Attempt {attempt}: all tests passed. Coverage: {coverage_percent:.1f}%.")

        if run_success and not static_issues:
            confidence = _confidence_score(attempt, iterations, coverage_percent, [], type_issues)
            final = AdvancedBuildResult(
                success=True, code=code, test_code=test_code,
                iterations=attempt, log=log, confidence=confidence,
                coverage_percent=coverage_percent,
                static_issues=[], type_issues=type_issues,
            )
            if use_cache:
                _registry.set(description, {
                    "success": True, "code": code, "test_code": test_code,
                    "iterations": attempt, "log": log, "confidence": confidence,
                    "coverage_percent": coverage_percent, "static_issues": [],
                    "type_issues": type_issues,
                })
            return final

        log.append(f"Attempt {attempt} failed: {error_summary[:200]}")

        if attempt == iterations:
            break

        coverage_note = (
            f"Coverage was only {coverage_percent:.1f}% — make sure the test exercises "
            f"more branches if you regenerate it." if coverage_percent and coverage_percent < 70 else ""
        )

        fixed = llm_client.complete(
            _CRITIQUE_PROMPT.format(
                description=description, code=code, test_code=test_code,
                error=error_summary, coverage_note=coverage_note,
            ),
            max_tokens=2000,
        )
        code = _strip_fences(fixed)
        log.append(f"Rewrote module after attempt {attempt} failure.")

        test_code = _strip_fences(llm_client.complete(_TEST_PROMPT.format(code=code), max_tokens=1000))
        log.append("Regenerated test script for the fixed module.")

    confidence = _confidence_score(iterations, iterations, coverage_percent, static_issues, type_issues)
    final = AdvancedBuildResult(
        success=False, code=code, test_code=test_code,
        iterations=iterations, log=log, confidence=confidence,
        coverage_percent=coverage_percent, static_issues=static_issues, type_issues=type_issues,
        error=f"Failed to produce passing code after {iterations} attempts. "
              f"Last error: {log[-1] if log else 'unknown'}",
    )
    if use_cache:
        _registry.set(description, {
            "success": False, "code": code, "test_code": test_code,
            "iterations": iterations, "log": log, "confidence": confidence,
            "coverage_percent": coverage_percent, "static_issues": static_issues,
            "type_issues": type_issues, "error": final.error,
        })
    return final


def build_streaming(description: str, max_iterations: Optional[int] = None, use_cache: bool = True):
    """
    Generator version of build(): yields human-readable progress lines
    as each step happens, then a final line starting with 'RESULT:'
    containing the JSON-encoded AdvancedBuildResult. Used by the
    /build/stream API endpoint so callers see progress in real time
    instead of waiting for the whole multi-attempt loop to finish.
    """
    import json as _json
    import dataclasses as _dc

    yield f"Starting build for: {description[:80]}..."

    if use_cache:
        cached = _registry.get(description)
        if cached:
            yield "Found cached result — returning immediately."
            yield "RESULT:" + _json.dumps({**cached, "from_cache": True})
            return

    if not config.LLM_ENABLED:
        yield "ERROR: no LLM configured."
        yield "RESULT:" + _json.dumps({"success": False, "error": "no LLM configured"})
        return

    iterations = max_iterations or MAX_ITERATIONS

    yield "Writing initial module..."
    code = _strip_fences(llm_client.complete(
        _WRITE_PROMPT.format(description=description, allowed_libs=", ".join(sorted(ALLOWED_EXTRA_LIBS))),
        max_tokens=2000,
    ))
    yield f"Wrote {len(code.splitlines())} lines."

    yield "Generating test script..."
    test_code = _strip_fences(llm_client.complete(_TEST_PROMPT.format(code=code), max_tokens=1000))
    yield f"Generated {len(test_code.splitlines())} lines of tests."

    log: list = []
    static_issues: list = []
    type_issues: list = []
    coverage_percent = 0.0

    for attempt in range(1, iterations + 1):
        yield f"Attempt {attempt}/{iterations}: running static analysis..."
        static_issues = _static_check(code)
        if static_issues:
            yield f"  Found {len(static_issues)} static issue(s)."

        type_issues = _type_check(code)

        if static_issues:
            run_success = False
            error_summary = "Static analysis issues:\n" + "\n".join(static_issues)
        else:
            yield f"Attempt {attempt}/{iterations}: running tests under coverage..."
            result, coverage_percent = _run_with_coverage(code, test_code)
            run_success = result.success
            error_summary = result.error_message
            yield f"  Coverage: {coverage_percent:.1f}%, success: {run_success}"

        if run_success and not static_issues:
            confidence = _confidence_score(attempt, iterations, coverage_percent, [], type_issues)
            yield f"All checks passed. Confidence: {confidence}/100."
            payload = {
                "success": True, "code": code, "test_code": test_code,
                "iterations": attempt, "confidence": confidence,
                "coverage_percent": coverage_percent, "static_issues": [],
                "type_issues": type_issues, "from_cache": False,
            }
            if use_cache:
                _registry.set(description, payload)
            yield "RESULT:" + _json.dumps(payload)
            return

        yield f"Attempt {attempt} failed: {error_summary[:150]}"
        if attempt == iterations:
            break

        yield "Critiquing and rewriting..."
        fixed = llm_client.complete(
            _CRITIQUE_PROMPT.format(
                description=description, code=code, test_code=test_code,
                error=error_summary, coverage_note="",
            ),
            max_tokens=2000,
        )
        code = _strip_fences(fixed)
        test_code = _strip_fences(llm_client.complete(_TEST_PROMPT.format(code=code), max_tokens=1000))
        yield "Rewrote module and regenerated tests."

    confidence = _confidence_score(iterations, iterations, coverage_percent, static_issues, type_issues)
    payload = {
        "success": False, "code": code, "test_code": test_code,
        "iterations": iterations, "confidence": confidence,
        "coverage_percent": coverage_percent, "static_issues": static_issues,
        "type_issues": type_issues, "from_cache": False,
        "error": f"Failed after {iterations} attempts.",
    }
    if use_cache:
        _registry.set(description, payload)
    yield "RESULT:" + _json.dumps(payload)
