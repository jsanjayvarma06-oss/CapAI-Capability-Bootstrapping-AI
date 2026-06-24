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
import os
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

    script_fd, script_name = tempfile.mkstemp(suffix=".py")
    os.close(script_fd)
    script_path = Path(script_name)
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
