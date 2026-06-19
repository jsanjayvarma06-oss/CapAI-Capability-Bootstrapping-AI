"""
Run with:   python -m capai.demo

Shows the full acquisition loop end-to-end: a capability that does not
exist yet gets diagnosed, written, tested, and promoted, then a second
call to the same capability is served instantly from the registry. Needs
zero API keys — uses the offline heuristic Diagnostic Agent / Code Writer
fallback. Set ANTHROPIC_API_KEY first to see it run with real LLM-backed
diagnosis and code generation instead.
"""
from __future__ import annotations

from capai import CapAI


def main() -> None:
    def on_event(event):
        print(f"  {event}")

    ai = CapAI(on_event=on_event)

    print("=" * 72)
    print("CALL 1 — 'celsius_to_fahrenheit' is NOT in the registry yet")
    print("=" * 72)
    result = ai.run("celsius_to_fahrenheit", "Convert a Celsius temperature to Fahrenheit", 100)
    print(f"\n  -> result = {result}\n")

    print("=" * 72)
    print("CALL 2 — same capability, now served instantly from the registry")
    print("=" * 72)
    result = ai.run("celsius_to_fahrenheit", "Convert a Celsius temperature to Fahrenheit", 0)
    print(f"\n  -> result = {result}\n")

    print("=" * 72)
    print("CALL 3 — a brand new capability triggers the loop again")
    print("=" * 72)
    result = ai.run("is_prime", "Check whether an integer is a prime number", 97)
    print(f"\n  -> result = {result}\n")

    print("Registry now contains:", ai.capabilities())


if __name__ == "__main__":
    main()
