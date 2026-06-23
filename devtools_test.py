"""
CapAI Developer Tools Stress Test
===================================
Tests 20 developer tool capabilities across 5 categories:
  - JSON tools
  - Slug generation
  - Password / hashing
  - URL validation
  - String utilities

Run:
    python devtools_test.py
"""
import sys
import time
sys.path.insert(0, '.')
from capai_plugin import CapAIPlugin, CapAIError

ai = CapAIPlugin()

passed = failed = 0
results = []

def test(name, description, args, expected=None, check=None, label=None):
    global passed, failed
    display = label or f"{name}({', '.join(repr(a) for a in args)})"
    try:
        t0 = time.monotonic()
        result = ai.run(name, description, *args)
        ms = (time.monotonic() - t0) * 1000

        if expected is not None:
            assert result == expected, f"expected {expected!r}, got {result!r}"
        if check is not None:
            assert check(result), f"check failed for result: {result!r}"

        print(f"  ✓  {display:<55} = {str(result):<25} ({ms:.0f}ms)")
        passed += 1
        results.append((name, True, result, ms))
    except CapAIError as e:
        print(f"  ✗  {display:<55} → {e}")
        failed += 1
        results.append((name, False, str(e), 0))
    except AssertionError as e:
        print(f"  ✗  {display:<55} → {e}")
        failed += 1
        results.append((name, False, str(e), 0))


def section(title):
    print(f"\n{'─'*75}")
    print(f"  {title}")
    print(f"{'─'*75}")


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*75)
print("  CapAI Developer Tools Stress Test")
print("═"*75)

h = ai.health()
print(f"\n  Server  : {h.get('status')}")
print(f"  Provider: {h.get('provider')} / {h.get('model')}")

# ── 1. JSON Tools ────────────────────────────────────────────────────────────
section("1. JSON TOOLS")

test("flatten_json",
     "Flatten a nested JSON dict into dot-notation keys",
     [{"user": {"name": "Sanjay", "age": 22}, "city": "Hyderabad"}],
     expected={"user.name": "Sanjay", "user.age": 22, "city": "Hyderabad"})

test("json_keys",
     "Return all top-level keys of a JSON object as a sorted list",
     [{"z": 1, "a": 2, "m": 3}],
     expected=["a", "m", "z"])

test("json_depth",
     "Return the maximum nesting depth of a JSON object",
     [{"a": {"b": {"c": 1}}}],
     expected=3)

test("json_to_csv_row",
     "Convert a flat JSON dict to a CSV row string with comma separation",
     [{"name": "Sanjay", "age": "22", "city": "Hyderabad"}],
     check=lambda r: "Sanjay" in r and "22" in r and "Hyderabad" in r)

test("count_json_keys",
     "Count total number of keys in a nested JSON object recursively",
     [{"a": 1, "b": {"c": 2, "d": {"e": 3}}}],
     expected=5)

# ── 2. Slug Generation ────────────────────────────────────────────────────────
section("2. SLUG GENERATION")

test("slugify",
     "Convert a string to a URL-friendly slug",
     ["Hello World!"],
     expected="hello-world")

test("slugify",
     "Convert a string to a URL-friendly slug",
     ["  Python is Awesome!!! "],
     check=lambda r: r == "python-is-awesome",
     label="slugify('  Python is Awesome!!! ')")

test("slugify",
     "Convert a string to a URL-friendly slug",
     ["CapAI -- Capability Bootstrapping AI"],
     check=lambda r: "capai" in r and "-" in r,
     label="slugify('CapAI -- Capability...')")

test("camel_to_snake",
     "Convert camelCase string to snake_case",
     ["helloWorldFoo"],
     expected="hello_world_foo")

test("snake_to_camel",
     "Convert snake_case string to camelCase",
     ["hello_world_foo"],
     expected="helloWorldFoo")

# ── 3. Password / Hashing ─────────────────────────────────────────────────────
section("3. PASSWORD & HASHING")

test("md5_hash",
     "Return MD5 hex digest of a string",
     ["hello"],
     expected="5d41402abc4b2a76b9719d911017c592")

test("sha256_hash",
     "Return SHA-256 hex digest of a string",
     ["hello"],
     expected="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")

test("is_strong_password",
     "Check if a password is strong: min 8 chars, has uppercase, lowercase, digit, special char",
     ["Str0ng!Pass"],
     expected=True)

test("is_strong_password",
     "Check if a password is strong: min 8 chars, has uppercase, lowercase, digit, special char",
     ["weakpass"],
     expected=False,
     label="is_strong_password('weakpass')")

test("password_strength_score",
     "Return a password strength score from 0 to 5 based on length, uppercase, lowercase, digits, special chars",
     ["Str0ng!Pass"],
     check=lambda r: isinstance(r, (int, float)) and r >= 4)

# ── 4. URL Validation ─────────────────────────────────────────────────────────
section("4. URL VALIDATION")

test("is_valid_url",
     "Check if a string is a valid URL starting with http or https",
     ["https://www.google.com"],
     expected=True)

test("is_valid_url",
     "Check if a string is a valid URL starting with http or https",
     ["not-a-url"],
     expected=False,
     label="is_valid_url('not-a-url')")

test("extract_domain",
     "Extract the domain name from a URL",
     ["https://www.google.com/search?q=capai"],
     expected="www.google.com")

test("is_valid_email",
     "Check if a string is a valid email address",
     ["sanjay@example.com"],
     expected=True)

test("is_valid_email",
     "Check if a string is a valid email address",
     ["not-an-email"],
     expected=False,
     label="is_valid_email('not-an-email')")

# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*75}")
print(f"  RESULTS: {passed} passed   {failed} failed   {passed+failed} total")
print(f"{'═'*75}")

# timing summary
times = [(n, ms) for n, ok, _, ms in results if ok and ms > 0]
if times:
    avg = sum(ms for _, ms in times) / len(times)
    slowest = max(times, key=lambda x: x[1])
    fastest = min(times, key=lambda x: x[1])
    print(f"\n  Avg response : {avg:.0f}ms")
    print(f"  Fastest      : {fastest[0]} ({fastest[1]:.0f}ms)")
    print(f"  Slowest      : {slowest[0]} ({slowest[1]:.0f}ms)")

print(f"\n  Capabilities now in registry:")
for cap in ai.capabilities():
    print(f"    • {cap}")
print()

sys.exit(0 if failed == 0 else 1)
