"""
capai.code_writer
===================
Section 3.3 / 4.4 of the report: turns a CapabilitySpec (the Diagnostic
Agent's output) into an actual, callable Python function.

When ANTHROPIC_API_KEY is configured, this asks Claude to write the
function. When it isn't, it falls back to a small built-in library of
common capabilities so the *rest* of the acquisition loop — MCP creation,
git versioning, three-layer testing, Manager Agent promotion, registry
reuse — can be exercised and demoed with zero external dependencies and
zero cost. The heuristic fallback is deliberately narrow: it exists so
`python -m capai.demo` works out of the box, not as a substitute for real
code generation. Anything it doesn't recognise gets an honestly-labelled
generic stub rather than a silent wrong answer.
"""
from __future__ import annotations

import re

from . import config
from .models import CapabilitySpec

_CODEGEN_PROMPT = """\
You are the Code Writer inside CapAI. Write ONE pure Python function that exactly matches this \
specification.

Requirements:
- No side effects: no file I/O, no network calls, no printing.
- Validate inputs and raise ValueError or TypeError for invalid input rather than silently \
returning a wrong answer.
- Standard library only — no third-party imports.
- Exactly one top-level function definition, nothing else.

Function name: {name}
Description: {description}
Required signature: {signature}
Example inputs that should work: {example_inputs}
Expected behaviour: {expected_behavior}

Respond with ONLY the Python source code for this function. No markdown fences, no commentary, \
no usage examples, no other functions.
"""


class CodeWriter:
    def __init__(self, client=None, model: str = None):
        self._client = client
        self.model = model or config.ANTHROPIC_MODEL

    def write(self, spec: CapabilitySpec) -> str:
        if not config.LLM_ENABLED:
            return _heuristic_source(spec)
        return self._write_with_llm(spec)

    def _write_with_llm(self, spec: CapabilitySpec) -> str:
        import anthropic

        client = self._client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        prompt = _CODEGEN_PROMPT.format(
            name=spec.name,
            description=spec.description,
            signature=spec.signature,
            example_inputs=spec.example_inputs,
            expected_behavior=spec.expected_behavior,
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return _strip_markdown_fences(text)


def _strip_markdown_fences(text: str) -> str:
    match = re.match(r"^```(?:python)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return match.group(1) if match else text


# ──────────────────────────────────────────────────────────────────────
# Offline fallback library. Recognises a small set of common capabilities
# by keyword so the demo and tests can run with zero API key. Real,
# arbitrary capability generation is the LLM path above — this is not a
# replacement for it.
# ──────────────────────────────────────────────────────────────────────

def _heuristic_source(spec: CapabilitySpec) -> str:
    key = f"{spec.name} {spec.description}".lower()
    name = spec.name

    if "celsius" in key and "fahrenheit" in key:
        return (
            f"def {name}(celsius):\n"
            f"    if not isinstance(celsius, (int, float)) or isinstance(celsius, bool):\n"
            f"        raise TypeError('celsius must be a number')\n"
            f"    return celsius * 9 / 5 + 32\n"
        )

    if "fahrenheit" in key and "celsius" not in key:
        return (
            f"def {name}(fahrenheit):\n"
            f"    if not isinstance(fahrenheit, (int, float)) or isinstance(fahrenheit, bool):\n"
            f"        raise TypeError('fahrenheit must be a number')\n"
            f"    return (fahrenheit - 32) * 5 / 9\n"
        )

    if "slug" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    import re\n"
            f"    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
            f"    if not slug:\n"
            f"        raise ValueError('text has no sluggable characters')\n"
            f"    return slug\n"
        )

    if "reverse" in key and "string" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    return text[::-1]\n"
        )

    if "is_even" in key or ("even" in key and "odd" not in key):
        return (
            f"def {name}(number):\n"
            f"    if not isinstance(number, int) or isinstance(number, bool):\n"
            f"        raise TypeError('number must be an integer')\n"
            f"    return number % 2 == 0\n"
        )

    if "is_prime" in key or "prime" in key:
        return (
            f"def {name}(number):\n"
            f"    if not isinstance(number, int) or isinstance(number, bool):\n"
            f"        raise TypeError('number must be an integer')\n"
            f"    if number < 2:\n"
            f"        return False\n"
            f"    for divisor in range(2, int(number ** 0.5) + 1):\n"
            f"        if number % divisor == 0:\n"
            f"            return False\n"
            f"    return True\n"
        )

    # Honestly-labelled generic fallback: validates that an input was
    # supplied and echoes it back. Keeps the surrounding loop runnable
    # offline for capabilities the heuristic library doesn't know; it is
    # not a stand-in for genuine code generation.
    return (
        f"def {name}(value=None):\n"
        f"    if value is None:\n"
        f"        raise ValueError('{name} requires an input value')\n"
        f"    return value\n"
    )
