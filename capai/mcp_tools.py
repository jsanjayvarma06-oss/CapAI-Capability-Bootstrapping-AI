"""
capai.mcp_tools
=================
MCP (Model Context Protocol) tool definitions, mounted directly inside
the main FastAPI app at /mcp. Unlike capai_mcp_server.py (a standalone
proxy that talks to CapAI over HTTP), this version calls the in-process
CapAI instance directly — no self-HTTP round trip, lower latency, and
it deploys as part of the SAME Render service rather than a separate one.

Any MCP-compatible client (Claude Desktop, agent frameworks) connects to:
    https://<your-capai-service>.onrender.com/mcp
"""
from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="CapAI",
    streamable_http_path="/",
    instructions=(
        "CapAI gives you skills you don't natively have. When you need to do "
        "something computational that you can't do reliably yourself — math, "
        "validation, parsing, string transforms, hashing, or even a small "
        "class or pipeline — describe what you need in plain English and "
        "call one of these tools instead of guessing or hallucinating an "
        "answer. Use capai_run for a single small function. Use capai_build "
        "for anything that needs multiple functions, a class, or a sequence "
        "of steps. Use capai_auto if you're not sure which one applies."
    ),
)

# Set by api.py right after CapAI() is instantiated, so these tools call
# the SAME registry/orchestrator the REST endpoints use — one shared
# state, no duplicate MongoDB connections, no HTTP round trip to itself.
_capai_instance = None


def bind(capai_instance) -> None:
    global _capai_instance
    _capai_instance = capai_instance


@mcp.tool()
def capai_run(name: str, description: str, args: Optional[list] = None) -> dict:
    """
    Acquire and run a single small capability — a function CapAI either
    already knows or will build on the spot. Best for one focused task:
    a calculation, a validator, a converter, a string transform, a hash.

    Args:
        name: a short snake_case identifier, e.g. "is_prime", "calculate_gst".
        description: plain English description of exactly what it should do.
        args: positional arguments to pass to the function, e.g. [17] or [70, 1.75].
    """
    try:
        result = _capai_instance.run(name, description, *(args or []))
        return {"success": True, "result": result, "capability_name": name}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def capai_build(description: str) -> dict:
    """
    Build something bigger than a single function — a class with several
    methods, a small algorithm, or a multi-step pipeline (e.g. extract,
    transform, load). The code is written, statically analysed, type
    checked, tested under coverage, and self-corrected on failure before
    being returned.

    Args:
        description: plain English description of the class/module/pipeline,
                      including the methods or steps it needs.
    """
    from .advanced_writer import build
    result = build(description)
    return {
        "success": result.success,
        "code": result.code,
        "confidence": result.confidence,
        "coverage_percent": result.coverage_percent,
        "iterations": result.iterations,
        "from_cache": result.from_cache,
        "error": result.error,
    }


@mcp.tool()
def capai_auto(description: str, args: Optional[list] = None) -> dict:
    """
    Let CapAI decide whether this is a simple single-function request or
    a bigger multi-function/class/pipeline request, and route it
    automatically. Use this when unsure which of capai_run or
    capai_build applies.

    Args:
        description: plain English description of what you need.
        args: positional arguments, only relevant for the simple route.
    """
    complex_signals = [
        "class", "pipeline", "multiple functions", "and a function that",
        "orchestrat", "several methods", "step 1", "step one", "module with",
    ]
    is_complex = (
        len(description.split()) > 25
        or any(sig in description.lower() for sig in complex_signals)
    )

    if is_complex:
        from .advanced_writer import build
        result = build(description)
        return {
            "route": "build", "success": result.success, "code": result.code,
            "confidence": result.confidence, "coverage_percent": result.coverage_percent,
            "iterations": result.iterations, "from_cache": result.from_cache,
            "error": result.error,
        }

    import re as _re
    STOPWORDS = {"a", "an", "the", "is", "if", "of", "to", "for", "and", "or",
                 "given", "from", "with", "in", "on", "this", "that", "it"}
    words = [w for w in _re.findall(r"[a-zA-Z]+", description.lower()) if w not in STOPWORDS][:5]
    name = "_".join(words) if words else "auto_capability"
    try:
        result = _capai_instance.run(name, description, *(args or []))
        return {"route": "run", "success": True, "result": result, "capability_name": name}
    except Exception as e:
        return {"route": "run", "success": False, "error": str(e)}


@mcp.tool()
def capai_list_capabilities() -> list:
    """
    List every single-function capability CapAI has already built and
    verified. Calling capai_run with one of these names returns
    instantly since it's already in the permanent registry.
    """
    return [
        {"name": c.name, "description": c.description, "verified": c.verified, "approved": c.approved}
        for c in _capai_instance.registry.list_active()
    ]


@mcp.tool()
def capai_health() -> dict:
    """
    Check whether CapAI is reachable and which LLM provider is backing it.
    """
    from . import config
    return {
        "status": "ok",
        "llm_enabled": config.LLM_ENABLED,
        "provider": "groq" if config.GROQ_API_KEY else ("anthropic" if config.ANTHROPIC_API_KEY else "offline"),
    }
