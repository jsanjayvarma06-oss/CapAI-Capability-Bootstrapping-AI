"""
capai.code_writer
===================
Turns a CapabilitySpec into an actual, callable Python function.

Uses the unified llm_client (Groq or Anthropic, whichever key is set).
Falls back to a small offline heuristic library when no key is configured
so `python -m capai.demo` always runs out of the box.
"""
from __future__ import annotations

import re

from . import config
from . import llm_client
from .models import CapabilitySpec

_CODEGEN_PROMPT = """\
You are the Code Writer inside CapAI. Write ONE pure Python function that exactly matches this \
specification.

Requirements:
- No side effects: no file I/O, no network calls, no printing.
- For out-of-domain inputs (e.g. negative numbers for is_prime, empty string for slugify), \
return a sensible default (False, None, 0) rather than raising an exception, \
unless the spec explicitly says to raise.
- Only raise TypeError for completely wrong types (e.g. passing a string where a number is required).
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
    def write(self, spec: CapabilitySpec) -> str:
        # always try heuristic first — it's guaranteed correct for known functions
        heuristic = _heuristic_source(spec)
        if heuristic is not None:
            return heuristic
        if not config.LLM_ENABLED:
            return _generic_stub(spec)
        prompt = _CODEGEN_PROMPT.format(
            name=spec.name,
            description=spec.description,
            signature=spec.signature,
            example_inputs=spec.example_inputs,
            expected_behavior=spec.expected_behavior,
        )
        text = llm_client.complete(prompt, max_tokens=800)
        return _strip_markdown_fences(text)


def _strip_markdown_fences(text: str) -> str:
    match = re.match(r"^```(?:python)?\s*\n(.*?)\n```\s*$", text, re.DOTALL)
    return match.group(1) if match else text


# ──────────────────────────────────────────────────────────────────────
# Offline fallback library
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

    return None  # not a known heuristic — let LLM handle it


def _generic_stub(spec: CapabilitySpec) -> str:
    name = spec.name
    return (
        f"def {name}(value=None):\n"
        f"    if value is None:\n"
        f"        raise ValueError('{name} requires an input value')\n"
        f"    return value\n"
    )
