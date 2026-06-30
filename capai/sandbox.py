"""
capai.sandbox
==============
The Safety Sandbox: runs a candidate capability's source code in a fresh
subprocess, never in CapAI's own process. A buggy or actively malicious
LLM-generated function can crash, hang, or do something unwise to ITS
process and nothing else — CapAI's own state, imports, and filesystem
access are untouched. This is what every layer of testing_agent.py calls
through.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import config


@dataclass
class CapabilityRunResult:
    success: bool
    return_value: Any = None
    error_message: str = ""


# Double braces are literal `{`/`}` once this template goes through
# str.format(); the inner f-string syntax (single braces) is what the
# CHILD process actually executes.
_RUNNER_TEMPLATE = """\
import json

CAPABILITY_SOURCE = {source!r}
FUNCTION_NAME = {function_name!r}
ARGS = json.loads({args_json!r})

namespace = {{}}
try:
    exec(CAPABILITY_SOURCE, namespace)
    func = namespace[FUNCTION_NAME]
    result = func(*ARGS)
    print(json.dumps({{"success": True, "return_value": result}}))
except Exception as e:
    print(json.dumps({{"success": False, "error_message": f"{{type(e).__name__}}: {{e}}"}}))
"""


def run_capability(
    source_code: str,
    function_name: str,
    args: tuple,
    timeout: Optional[float] = None,
) -> CapabilityRunResult:
    """
    Execute `function_name(*args)` as defined in `source_code`, isolated
    in its own subprocess with a hard timeout. Arguments and the return
    value travel as JSON, which is a deliberate restriction: a capability
    that can't express its inputs/outputs as JSON-safe values (numbers,
    strings, lists, dicts, booleans, None) isn't a good candidate for an
    autonomously-generated, autonomously-tested capability in the first
    place.
    """
    timeout = config.SANDBOX_TIMEOUT_SECONDS if timeout is None else timeout

    try:
        args_json = json.dumps(list(args))
    except TypeError:
        return CapabilityRunResult(
            success=False,
            error_message="Arguments are not JSON-serialisable; cannot run in the sandbox.",
        )

    script = _RUNNER_TEMPLATE.format(
        source=source_code, function_name=function_name, args_json=args_json
    )

    script_path = Path(tempfile.mkstemp(suffix=".py")[1])
    script_path.write_text(script)

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CapabilityRunResult(
            success=False, error_message=f"Execution timed out after {timeout} seconds."
        )
    finally:
        script_path.unlink(missing_ok=True)

    stdout = proc.stdout.strip()
    if not stdout:
        return CapabilityRunResult(
            success=False,
            error_message=proc.stderr.strip() or "Sandbox process produced no output (likely crashed).",
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return CapabilityRunResult(
            success=False,
            error_message=f"Could not parse sandbox output. stdout={stdout!r} stderr={proc.stderr!r}",
        )

    if payload.get("success"):
        return CapabilityRunResult(success=True, return_value=payload.get("return_value"))
    return CapabilityRunResult(
        success=False, error_message=payload.get("error_message", "Unknown sandbox failure.")
    )


# ──────────────────────────────────────────────────────────────────────
# Module-level sandbox: runs arbitrary source code plus a self-generated
# test script in the same subprocess. Used by advanced_writer.py for
# multi-function modules, classes, and pipelines where a single
# call_function(*args) -> value shape doesn't apply.
# ──────────────────────────────────────────────────────────────────────

_MODULE_RUNNER_TEMPLATE = """\\
import json, traceback

MODULE_SOURCE = {source!r}
TEST_SOURCE = {test_source!r}

namespace = {{}}
report = {{"success": False, "error_message": "", "test_output": ""}}

try:
    exec(MODULE_SOURCE, namespace)
except Exception as e:
    report["error_message"] = f"Module failed to load: {{type(e).__name__}}: {{e}}"
    print(json.dumps(report))
    raise SystemExit(0)

import io, contextlib
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(TEST_SOURCE, namespace)
    report["success"] = True
    report["test_output"] = buf.getvalue()
except AssertionError as e:
    report["error_message"] = f"Test assertion failed: {{e}}"
    report["test_output"] = buf.getvalue()
except Exception as e:
    report["error_message"] = f"{{type(e).__name__}}: {{e}}\\n{{traceback.format_exc()}}"
    report["test_output"] = buf.getvalue()

print(json.dumps(report))
"""


@dataclass
class ModuleRunResult:
    success: bool
    error_message: str = ""
    test_output: str = ""


def run_module_with_tests(
    module_source: str,
    test_source: str,
    timeout: Optional[float] = None,
) -> ModuleRunResult:
    """
    Load `module_source` (which may define multiple functions, classes,
    or a small pipeline) into a fresh subprocess, then execute
    `test_source` against it. `test_source` should use plain `assert`
    statements — any AssertionError or exception is reported back as a
    structured failure rather than crashing the parent process.
    """
    timeout = config.SANDBOX_TIMEOUT_SECONDS * 2 if timeout is None else timeout

    script = _MODULE_RUNNER_TEMPLATE.format(source=module_source, test_source=test_source)
    script_path = Path(tempfile.mkstemp(suffix=".py")[1])
    script_path.write_text(script)

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ModuleRunResult(success=False, error_message=f"Execution timed out after {timeout}s.")
    finally:
        script_path.unlink(missing_ok=True)

    stdout = proc.stdout.strip()
    if not stdout:
        return ModuleRunResult(
            success=False,
            error_message=proc.stderr.strip() or "Sandbox produced no output (likely crashed).",
        )

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return ModuleRunResult(
            success=False,
            error_message=f"Could not parse sandbox output. stdout={stdout!r} stderr={proc.stderr!r}",
        )

    return ModuleRunResult(
        success=payload.get("success", False),
        error_message=payload.get("error_message", ""),
        test_output=payload.get("test_output", ""),
    )
