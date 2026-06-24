"""
capai_plugin.py
================
Production-ready CapAI plugin. Drop this single file into any Python project.

USAGE — three ways to use it:

1. Direct call:
    from capai_plugin import capai
    result = capai("add_numbers", "Add two integers", 3, 4)

2. Decorator — auto-fallback to CapAI if function raises NotImplementedError:
    from capai_plugin import capai_fallback

    @capai_fallback
    def reverse_string(text: str) -> str:
        raise NotImplementedError

    print(reverse_string("hello"))  # CapAI builds and runs it automatically

3. Lazy capability — define intent, call later:
    from capai_plugin import CapAIPlugin

    ai = CapAIPlugin()
    bmi = ai.capability("calculate_bmi", "Calculate BMI from weight(kg) and height(m)")
    print(bmi(70, 1.75))  # 22.86

Zero dependencies — pure Python stdlib.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
import urllib.request
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(format="[CapAI] %(levelname)s %(message)s")
logger = logging.getLogger("capai")
logger.setLevel(os.environ.get("CAPAI_LOG_LEVEL", "INFO"))


# ── configuration ─────────────────────────────────────────────────────────────

CAPAI_URL        = os.environ.get("CAPAI_URL", "https://capai-capability-bootstrapping-ai-fu58.onrender.com")
CAPAI_TIMEOUT    = int(os.environ.get("CAPAI_TIMEOUT", "60"))
CAPAI_RETRIES    = int(os.environ.get("CAPAI_RETRIES", "3"))
CAPAI_BACKOFF    = float(os.environ.get("CAPAI_BACKOFF", "2.0"))   # seconds, doubles each retry
CAPAI_CACHE      = os.environ.get("CAPAI_CACHE", "true").lower() == "true"
CAPAI_CB_THRESH  = int(os.environ.get("CAPAI_CB_THRESH", "5"))     # circuit breaker threshold


# ── exceptions ────────────────────────────────────────────────────────────────

class CapAIError(Exception):
    """Base error for all CapAI plugin failures."""

class CapAIUnavailable(CapAIError):
    """Server is unreachable or circuit breaker is open."""

class CapAIBuildError(CapAIError):
    """Server could not build the requested capability."""


# ── circuit breaker ───────────────────────────────────────────────────────────

class _CircuitBreaker:
    """
    Stops hammering a down server.
    Opens after CAPAI_CB_THRESH consecutive failures.
    Tries again after 60 seconds (half-open probe).
    """
    def __init__(self, threshold: int = CAPAI_CB_THRESH, recovery: float = 60.0):
        self._failures = 0
        self._threshold = threshold
        self._recovery = recovery
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self._recovery:
                # half-open: allow one probe
                return False
            return True

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = time.monotonic()
                logger.warning(
                    f"Circuit breaker opened after {self._failures} failures. "
                    f"Will retry in {self._recovery}s."
                )


# ── local cache ───────────────────────────────────────────────────────────────

class _LocalCache:
    """
    Thread-safe in-memory cache of capability results.
    Key: (name, args_repr) → value: result
    Separate registry cache: name → True (means server has it, skip build phase)
    """
    def __init__(self):
        self._results: dict[str, Any] = {}
        self._registry: set[str] = set()
        self._lock = threading.Lock()

    def get(self, name: str, args: tuple) -> tuple[bool, Any]:
        key = f"{name}:{repr(args)}"
        with self._lock:
            if key in self._results:
                return True, self._results[key]
        return False, None

    def set(self, name: str, args: tuple, value: Any):
        key = f"{name}:{repr(args)}"
        with self._lock:
            self._results[key] = value
            self._registry.add(name)

    def invalidate(self, name: Optional[str] = None):
        with self._lock:
            if name:
                self._registry.discard(name)
                keys = [k for k in self._results if k.startswith(f"{name}:")]
                for k in keys:
                    del self._results[k]
            else:
                self._results.clear()
                self._registry.clear()

    def known(self, name: str) -> bool:
        with self._lock:
            return name in self._registry


# ── core plugin ───────────────────────────────────────────────────────────────

class CapAIPlugin:
    """
    Production CapAI client with caching, retries, circuit breaker, and observability.

    Environment variables:
        CAPAI_URL         Server URL (default: your Render deployment)
        CAPAI_TIMEOUT     Request timeout in seconds (default: 60)
        CAPAI_RETRIES     Max retry attempts (default: 3)
        CAPAI_BACKOFF     Initial backoff in seconds, doubles each retry (default: 2.0)
        CAPAI_CACHE       Enable local result cache (default: true)
        CAPAI_CB_THRESH   Circuit breaker failure threshold (default: 5)
        CAPAI_LOG_LEVEL   Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    """

    def __init__(self, url: str = CAPAI_URL):
        self.url = url.rstrip("/")
        self._cache = _LocalCache()
        self._cb = _CircuitBreaker()

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, name: str, description: str, *args, **kwargs) -> Any:
        """
        Run a capability by name. Builds it automatically if it doesn't exist.

        Args:
            name:        snake_case function name e.g. "celsius_to_fahrenheit"
            description: plain English description e.g. "Convert Celsius to Fahrenheit"
            *args:       positional arguments to pass to the function
            **kwargs:    keyword arguments to pass to the function

        Returns:
            Whatever the capability returns.

        Raises:
            CapAIUnavailable: server is down or circuit breaker is open
            CapAIBuildError:  server failed to build the capability after retries
        """
        t0 = time.monotonic()

        # local cache hit — no network call at all
        if CAPAI_CACHE:
            hit, cached = self._cache.get(name, args)
            if hit:
                logger.debug(f"cache hit: {name}{args} = {cached!r}")
                return cached

        result = self._run_with_retry(name, description, args, kwargs)

        elapsed = (time.monotonic() - t0) * 1000
        logger.info(f"{name}{args} → {result!r}  ({elapsed:.0f}ms)")

        if CAPAI_CACHE:
            self._cache.set(name, args, result)

        return result

    def capability(self, name: str, description: str) -> Callable:
        """
        Return a callable that runs a named capability.
        Useful for defining intent once and calling many times.

        Example:
            bmi = ai.capability("calculate_bmi", "Calculate BMI from weight(kg) and height(m)")
            print(bmi(70, 1.75))
        """
        @functools.wraps(lambda *a, **kw: None)
        def _call(*args, **kwargs):
            return self.run(name, description, *args, **kwargs)
        _call.__name__ = name
        _call.__doc__ = description
        return _call

    def health(self) -> dict:
        """Check server health. Returns dict with status, provider, model."""
        try:
            return self._get("/health")
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def capabilities(self) -> list[str]:
        """List all capability names currently in the server registry."""
        try:
            caps = self._get("/capabilities")
            return [c["name"] for c in caps]
        except Exception:
            return []

    def reset(self):
        """Wipe the server registry and local cache. Useful in tests."""
        self._cache.invalidate()
        try:
            result = self._post("/reset", {})
            logger.info(f"Registry reset: {result}")
            return result
        except Exception as e:
            logger.warning(f"Reset failed: {e}")

    def invalidate(self, name: Optional[str] = None):
        """Invalidate local cache for a capability (or all if name is None)."""
        self._cache.invalidate(name)
        logger.debug(f"Cache invalidated: {name or 'all'}")

    # ── retry loop ────────────────────────────────────────────────────────────

    def _run_with_retry(self, name: str, description: str, args: tuple, kwargs: dict) -> Any:
        if self._cb.open:
            raise CapAIUnavailable(
                f"CapAI circuit breaker is open — server had too many recent failures. "
                f"Try again in ~60s or check {self.url}/health"
            )

        last_error: Optional[Exception] = None
        backoff = CAPAI_BACKOFF

        for attempt in range(1, CAPAI_RETRIES + 1):
            try:
                payload = {"name": name, "description": description,
                           "args": list(args), "kwargs": kwargs}
                response = self._post("/run", payload)

                if not response.get("success"):
                    raise CapAIBuildError(response.get("error", "unknown error"))

                self._cb.record_success()
                return response["result"]

            except CapAIBuildError:
                # build failures don't benefit from retrying immediately
                self._cb.record_failure()
                raise

            except (CapAIUnavailable, URLError, TimeoutError, OSError) as e:
                last_error = e
                self._cb.record_failure()
                if attempt < CAPAI_RETRIES:
                    logger.warning(
                        f"Attempt {attempt}/{CAPAI_RETRIES} failed for '{name}': {e}. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2

            except Exception as e:
                last_error = e
                self._cb.record_failure()
                if attempt < CAPAI_RETRIES:
                    logger.warning(f"Attempt {attempt}/{CAPAI_RETRIES} failed for '{name}': {e}. Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2

        raise CapAIUnavailable(
            f"CapAI server unreachable after {CAPAI_RETRIES} attempts. Last error: {last_error}"
        )

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _post(self, path: str, data: dict) -> dict:
        url = self.url + path
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=CAPAI_TIMEOUT) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail", str(e))
            except Exception:
                detail = str(e)
            raise CapAIBuildError(f"HTTP {e.code}: {detail}") from e
        except URLError as e:
            raise CapAIUnavailable(f"Cannot reach CapAI server at {url}: {e.reason}") from e

    def _get(self, path: str) -> Any:
        url = self.url + path
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=CAPAI_TIMEOUT) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            raise CapAIBuildError(f"HTTP {e.code}") from e
        except URLError as e:
            raise CapAIUnavailable(f"Cannot reach CapAI server: {e.reason}") from e


# ── module-level singleton ────────────────────────────────────────────────────

_default = CapAIPlugin()


def capai(name: str, description: str, *args, **kwargs) -> Any:
    """
    Module-level shortcut. Uses a shared CapAIPlugin instance.

    from capai_plugin import capai
    result = capai("celsius_to_fahrenheit", "Convert Celsius to Fahrenheit", 100)
    """
    return _default.run(name, description, *args, **kwargs)


def capai_fallback(fn: Callable) -> Callable:
    """
    Decorator. If the function raises NotImplementedError, CapAI builds and
    runs it automatically using the function name and docstring as the spec.

    Example:
        @capai_fallback
        def reverse_string(text: str) -> str:
            \"\"\"Reverse a string\"\"\"
            raise NotImplementedError

        print(reverse_string("hello"))  # "olleh"
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError:
            description = fn.__doc__ or f"Implement {fn.__name__}"
            logger.info(f"@capai_fallback: '{fn.__name__}' not implemented, delegating to CapAI")
            return _default.run(fn.__name__, description, *args, **kwargs)
    return wrapper


# ── demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ai = CapAIPlugin()

    print("=" * 55)
    print("  CapAI Plugin — Production Demo")
    print("=" * 55)

    # health check
    h = ai.health()
    print(f"\n  Server : {h.get('status')}")
    print(f"  Provider: {h.get('provider')} / {h.get('model')}")

    tests = [
        ("celsius_to_fahrenheit", "Convert Celsius to Fahrenheit", (100,)),
        ("celsius_to_fahrenheit", "Convert Celsius to Fahrenheit", (0,)),    # cache hit
        ("is_prime",              "Check if a number is prime",    (17,)),
        ("is_prime",              "Check if a number is prime",    (4,)),     # cache hit
        ("slugify",               "Convert a string to a URL slug", ("Hello World!",)),
        ("reverse_string",        "Reverse a string",              ("hello",)),
        ("word_count",            "Count words in a sentence",     ("the quick brown fox",)),
        ("km_to_miles",           "Convert kilometres to miles",   (10,)),
    ]

    print(f"\n{'─'*55}")
    passed = failed = 0
    for name, desc, args in tests:
        try:
            result = ai.run(name, desc, *args)
            print(f"  ✓  {name}{args} = {result!r}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {name}{args} → {e}")
            failed += 1

    print(f"{'─'*55}")
    print(f"\n  {passed} passed  {failed} failed")

    # decorator demo
    print(f"\n{'─'*55}")
    print("  @capai_fallback decorator demo")
    print(f"{'─'*55}")

    @capai_fallback
    def calculate_bmi(weight_kg: float, height_m: float) -> float:
        """Calculate Body Mass Index from weight in kg and height in metres"""
        raise NotImplementedError

    try:
        bmi = calculate_bmi(70, 1.75)
        print(f"  ✓  calculate_bmi(70, 1.75) = {bmi!r}")
    except Exception as e:
        print(f"  ✗  calculate_bmi → {e}")

    # capability() demo
    print(f"\n{'─'*55}")
    print("  ai.capability() lazy binding demo")
    print(f"{'─'*55}")

    to_fahrenheit = ai.capability("celsius_to_fahrenheit", "Convert Celsius to Fahrenheit")
    for c in [0, 100, 37]:
        try:
            print(f"  ✓  {c}°C = {to_fahrenheit(c)}°F  (cache hit)" if ai._cache.known("celsius_to_fahrenheit") else f"  ✓  {c}°C = {to_fahrenheit(c)}°F")
        except Exception as e:
            print(f"  ✗  {c}°C → {e}")

    print(f"\n{'─'*55}")
    print("  Capabilities in server registry:")
    for cap in ai.capabilities():
        print(f"    • {cap}")
    print(f"{'─'*55}\n")
    sys.exit(0 if failed == 0 else 1)
