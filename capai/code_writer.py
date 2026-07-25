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
        if not getattr(config, 'SKIP_HEURISTICS', False):
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

    if "flatten" in key and "json" in key:
        return (
            f"def {name}(data, parent_key='', sep='.'):\n"
            f"    if not isinstance(data, dict):\n"
            f"        raise TypeError('Input must be a dict')\n"
            f"    items = {{}}\n"
            f"    for k, v in data.items():\n"
            f"        new_key = parent_key + sep + k if parent_key else k\n"
            f"        if isinstance(v, dict):\n"
            f"            items.update({name}(v, new_key, sep=sep))\n"
            f"        else:\n"
            f"            items[new_key] = v\n"
            f"    return items\n"
        )

    if "json_keys" in key or ("json" in key and "key" in key and "count" not in key and "csv" not in key):
        return (
            f"def {name}(data):\n"
            f"    if not isinstance(data, dict):\n"
            f"        raise TypeError('Input must be a dict')\n"
            f"    return sorted(data.keys())\n"
        )

    if "json_to_csv" in key or ("json" in key and "csv" in key):
        return (
            f"def {name}(data):\n"
            f"    if not isinstance(data, dict):\n"
            f"        raise TypeError('Input must be a dict')\n"
            f"    return ','.join(f'{{k}}={{v}}' for k, v in data.items())\n"
        )

    if "count_json_keys" in key or ("count" in key and "json" in key and "key" in key):
        return (
            f"def {name}(data):\n"
            f"    if not isinstance(data, dict):\n"
            f"        raise TypeError('Input must be a dict')\n"
            f"    count = 0\n"
            f"    for k, v in data.items():\n"
            f"        count += 1\n"
            f"        if isinstance(v, dict):\n"
            f"            count += {name}(v)\n"
            f"    return count\n"
        )

    # ── temperature ──────────────────────────────────────────────────────
    if "km" in key and "mile" in key:
        return (
            f"def {name}(km):\n"
            f"    if not isinstance(km, (int, float)) or isinstance(km, bool):\n"
            f"        raise TypeError('km must be a number')\n"
            f"    return km * 0.621371\n"
        )

    if "mile" in key and "km" in key:
        return (
            f"def {name}(miles):\n"
            f"    if not isinstance(miles, (int, float)) or isinstance(miles, bool):\n"
            f"        raise TypeError('miles must be a number')\n"
            f"    return miles / 0.621371\n"
        )

    # ── string utils ─────────────────────────────────────────────────────
    if "reverse" in key and ("string" in key or "str" in key):
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    return text[::-1]\n"
        )

    if "word_count" in key or ("word" in key and "count" in key):
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    return len(text.split())\n"
        )

    if "camel" in key and "snake" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    import re\n"
            f"    return re.sub(r'(?<!^)(?=[A-Z])', '_', text).lower()\n"
        )

    if "snake" in key and "camel" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    parts = text.split('_')\n"
            f"    return parts[0] + ''.join(p.capitalize() for p in parts[1:])\n"
        )

    if "palindrome" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    clean = text.lower().replace(' ', '')\n"
            f"    return clean == clean[::-1]\n"
        )

    if "truncate" in key:
        return (
            f"def {name}(text, length=100, suffix='...'):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    return text if len(text) <= length else text[:length] + suffix\n"
        )

    if "count_vowel" in key or ("count" in key and "vowel" in key):
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    return sum(1 for c in text.lower() if c in 'aeiou')\n"
        )

    # ── hashing ──────────────────────────────────────────────────────────
    if "md5" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    import hashlib\n"
            f"    return hashlib.md5(text.encode()).hexdigest()\n"
        )

    if "sha256" in key or "sha_256" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    import hashlib\n"
            f"    return hashlib.sha256(text.encode()).hexdigest()\n"
        )

    if "sha512" in key or "sha_512" in key:
        return (
            f"def {name}(text):\n"
            f"    if not isinstance(text, str):\n"
            f"        raise TypeError('text must be a string')\n"
            f"    import hashlib\n"
            f"    return hashlib.sha512(text.encode()).hexdigest()\n"
        )

    # ── password ─────────────────────────────────────────────────────────
    if "strong_password" in key or ("strong" in key and "password" in key):
        return (
            f"def {name}(password):\n"
            f"    if not isinstance(password, str):\n"
            f"        raise TypeError('password must be a string')\n"
            f"    if len(password) < 8: return False\n"
            f"    if not any(c.isupper() for c in password): return False\n"
            f"    if not any(c.islower() for c in password): return False\n"
            f"    if not any(c.isdigit() for c in password): return False\n"
            f"    if not any(c in '!@#$%^&*()_+-=[]{{}}|;:,.<>?' for c in password): return False\n"
            f"    return True\n"
        )

    if "password_strength" in key or ("password" in key and "strength" in key and "score" in key):
        return (
            f"def {name}(password):\n"
            f"    if not isinstance(password, str):\n"
            f"        raise TypeError('password must be a string')\n"
            f"    score = 0\n"
            f"    if len(password) >= 8: score += 1\n"
            f"    if any(c.isupper() for c in password): score += 1\n"
            f"    if any(c.islower() for c in password): score += 1\n"
            f"    if any(c.isdigit() for c in password): score += 1\n"
            f"    if any(c in '!@#$%^&*()_+-=[]{{}}|;:,.<>?' for c in password): score += 1\n"
            f"    return score\n"
        )

    # ── url / email ───────────────────────────────────────────────────────
    if "valid_url" in key or ("valid" in key and "url" in key):
        return (
            f"def {name}(url):\n"
            f"    if not isinstance(url, str):\n"
            f"        raise TypeError('url must be a string')\n"
            f"    return url.startswith('http://') or url.startswith('https://')\n"
        )

    if "extract_domain" in key or ("domain" in key and "extract" in key):
        return (
            f"def {name}(url):\n"
            f"    if not isinstance(url, str):\n"
            f"        raise TypeError('url must be a string')\n"
            f"    from urllib.parse import urlparse\n"
            f"    return urlparse(url).netloc\n"
        )

    if "valid_email" in key or ("valid" in key and "email" in key):
        return (
            f"def {name}(email):\n"
            f"    if not isinstance(email, str):\n"
            f"        raise TypeError('email must be a string')\n"
            f"    import re\n"
            f"    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{{2,}}$', email))\n"
        )

    # ── maths ─────────────────────────────────────────────────────────────
    if "factorial" in key:
        return (
            f"def {name}(n):\n"
            f"    if not isinstance(n, int) or isinstance(n, bool) or n < 0:\n"
            f"        raise ValueError('n must be a non-negative integer')\n"
            f"    import math\n"
            f"    return math.factorial(n)\n"
        )

    if "fibonacci" in key:
        return (
            f"def {name}(n):\n"
            f"    if not isinstance(n, int) or isinstance(n, bool) or n < 0:\n"
            f"        raise ValueError('n must be a non-negative integer')\n"
            f"    a, b = 0, 1\n"
            f"    for _ in range(n):\n"
            f"        a, b = b, a + b\n"
            f"    return a\n"
        )

    if "bmi" in key or ("body" in key and "mass" in key):
        return (
            f"def {name}(weight_kg, height_m):\n"
            f"    if not isinstance(weight_kg, (int, float)) or isinstance(weight_kg, bool):\n"
            f"        raise TypeError('weight_kg must be a number')\n"
            f"    if not isinstance(height_m, (int, float)) or isinstance(height_m, bool):\n"
            f"        raise TypeError('height_m must be a number')\n"
            f"    if height_m <= 0:\n"
            f"        raise ValueError('height_m must be positive')\n"
            f"    return weight_kg / (height_m ** 2)\n"
        )

    if "is_even" in key or ("even" in key and "odd" not in key):
        return (
            f"def {name}(number):\n"
            f"    if not isinstance(number, int) or isinstance(number, bool):\n"
            f"        raise TypeError('number must be an integer')\n"
            f"    return number % 2 == 0\n"
        )

    if "is_odd" in key or ("odd" in key and "even" not in key):
        return (
            f"def {name}(number):\n"
            f"    if not isinstance(number, int) or isinstance(number, bool):\n"
            f"        raise TypeError('number must be an integer')\n"
            f"    return number % 2 != 0\n"
        )

    if "clamp" in key:
        return (
            f"def {name}(value, min_val, max_val):\n"
            f"    return max(min_val, min(max_val, value))\n"
        )

    if "percent" in key or "percentage" in key:
        return (
            f"def {name}(part, total):\n"
            f"    if total == 0:\n"
            f"        raise ValueError('total cannot be zero')\n"
            f"    return (part / total) * 100\n"
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
