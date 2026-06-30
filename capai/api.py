"""
capai.api
==========
FastAPI REST layer — exposes CapAI as an HTTP service so it can be hosted
on Render (or any other platform) and called from any language.

Run locally:
    uvicorn capai.api:app --reload --port 8000

Endpoints:
    POST /run          — acquire + run a capability for a task
    GET  /capabilities — list all active capabilities in the registry
    GET  /health       — liveness check
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .__init__ import CapAI

_capai = CapAI()

# ── MCP server setup ──────────────────────────────────────────────────────────
# Exposes the same CapAI capabilities as native MCP tools at /mcp, so any
# MCP-compatible client (Claude Desktop, agent frameworks, etc.) can connect
# to THIS SAME Render service instead of needing a separate one.
_mcp_server = None
try:
    from .mcp_tools import mcp as _mcp_server, bind as _mcp_bind
    _mcp_bind(_capai)
except Exception as e:
    print(f"[api] MCP server could not be initialised ({e}) — REST API still works normally.")


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Combined startup/shutdown for the main app AND the mounted MCP
    server. The MCP streamable-http transport requires its session
    manager's run() context to be active for the whole process lifetime
    — mounting the sub-app alone does NOT start this automatically, so
    it must be entered here as part of the parent app's own lifespan.
    """
    # startup: clear any broken registry entries left over from a
    # previous failed acquisition attempt
    broken = [
        cap.name for cap in _capai.registry._capabilities.values()
        if not getattr(cap, "source_code", None)
    ]
    for name in broken:
        _capai.registry._capabilities.pop(name, None)
    if broken:
        print(f"[startup] Cleared {len(broken)} broken registry entries: {broken}")

    if _mcp_server is not None:
        async with _mcp_server.session_manager.run():
            print("[api] MCP session manager started.")
            yield
    else:
        yield


app = FastAPI(
    title="CapAI",
    description="Self-expanding capability acquisition layer for AI systems.",
    version="0.1.0",
    lifespan=_lifespan,
)

if _mcp_server is not None:
    app.mount("/mcp", _mcp_server.streamable_http_app())
    print("[api] MCP server mounted at /mcp")


# ── request / response models ────────────────────────────────────────

class RunRequest(BaseModel):
    name: str
    description: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


class RunResponse(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    capability_name: str


class CapabilityInfo(BaseModel):
    name: str
    description: str
    verified: bool
    approved: bool


# ── endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_enabled": config.LLM_ENABLED,
        "provider": "groq" if config.GROQ_API_KEY else ("anthropic" if config.ANTHROPIC_API_KEY else "offline"),
        "model": config.GROQ_MODEL if config.GROQ_API_KEY else config.ANTHROPIC_MODEL,
    }


@app.get("/capabilities", response_model=List[CapabilityInfo])
def list_capabilities():
    caps = _capai.registry.list_active()
    return [
        CapabilityInfo(
            name=c.name,
            description=c.description,
            verified=c.verified,
            approved=c.approved,
        )
        for c in caps
    ]


@app.post("/reset")
def reset_registry():
    """Wipe in-memory registry only. MongoDB is preserved so capabilities reload on restart."""
    all_names = list(_capai.registry._capabilities.keys())
    _capai.registry._capabilities.clear()
    return {"cleared": len(all_names), "removed": all_names, "message": "Memory wiped — capabilities will reload from MongoDB on next call"}


@app.post("/reset/{name}")
def reset_capability(name: str):
    """Delete a single capability from memory and MongoDB. Use 'hard' to wipe everything."""
    if name == "hard":
        # full wipe — memory + MongoDB
        all_names = list(_capai.registry._capabilities.keys())
        _capai.registry._capabilities.clear()
        if _capai.registry._collection is not None:
            try:
                _capai.registry._collection.delete_many({})
            except Exception as e:
                print(f"[reset] MongoDB wipe failed: {e}")
        return {"cleared": len(all_names), "removed": all_names, "message": "Memory and MongoDB wiped — all capabilities will be rebuilt from scratch"}
    # single capability delete
    _capai.registry._capabilities.pop(name, None)
    if _capai.registry._collection is not None:
        try:
            _capai.registry._collection.delete_one({"name": name})
        except Exception as e:
            print(f"[reset] MongoDB delete failed for '{name}': {e}")
    return {"deleted": name, "message": f"'{name}' removed — will be rebuilt on next call"}


@app.post("/run", response_model=RunResponse)
def run_task(req: RunRequest):
    try:
        result = _capai.run(req.name, req.description, *req.args, **req.kwargs)
        return RunResponse(success=True, result=result, capability_name=req.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class BuildRequest(BaseModel):
    description: str
    max_iterations: Optional[int] = None
    use_cache: bool = True


class BuildResponse(BaseModel):
    success: bool
    code: str = ""
    test_code: str = ""
    iterations: int = 0
    log: List[str] = []
    error: str = ""
    confidence: int = 0
    coverage_percent: float = 0.0
    from_cache: bool = False
    static_issues: List[str] = []
    type_issues: List[str] = []


@app.post("/build", response_model=BuildResponse)
def build_advanced(req: BuildRequest):
    """
    Build bigger-than-a-single-function code: multi-function modules,
    classes, algorithms, or pipelines. Writes the code, statically
    analyses it, type-checks it, auto-generates a coverage-tracked test
    suite, self-corrects on failure, scores confidence, and caches the
    verified result in MongoDB so repeat requests are instant.
    """
    from .advanced_writer import build
    result = build(req.description, max_iterations=req.max_iterations, use_cache=req.use_cache)
    return BuildResponse(
        success=result.success,
        code=result.code,
        test_code=result.test_code,
        iterations=result.iterations,
        log=result.log,
        error=result.error,
        confidence=result.confidence,
        coverage_percent=result.coverage_percent,
        from_cache=result.from_cache,
        static_issues=result.static_issues,
        type_issues=result.type_issues,
    )


@app.post("/build/stream")
async def build_advanced_stream(req: BuildRequest):
    """
    Same as /build, but streams progress log lines as they happen
    (server-sent text lines) instead of waiting for the entire
    write -> test -> critique loop to finish before responding.
    """
    from fastapi.responses import StreamingResponse
    from .advanced_writer import build_streaming

    async def event_stream():
        for line in build_streaming(req.description, max_iterations=req.max_iterations, use_cache=req.use_cache):
            yield line + "\n"

    return StreamingResponse(event_stream(), media_type="text/plain")


@app.get("/usage")
def usage_summary():
    """Token usage and cost-tracking summary across Groq and Anthropic calls."""
    from .llm_client import get_usage_summary
    return get_usage_summary()


class AutoRequest(BaseModel):
    description: str
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}


@app.post("/auto")
def auto_route(req: AutoRequest):
    """
    Difficulty-aware single entry point. Heuristically decides whether a
    request is a simple single-function call (routes to /run-style
    acquisition, fast heuristic path) or a multi-function/class/pipeline
    request (routes to /build's full write-test-critique loop), so
    callers don't have to choose the endpoint themselves.
    """
    complex_signals = [
        "class", "pipeline", "multiple functions", "and a function that",
        "orchestrat", "several methods", "step 1", "step one", "module with",
    ]
    is_complex = (
        len(req.description.split()) > 25
        or any(sig in req.description.lower() for sig in complex_signals)
    )

    if is_complex:
        from .advanced_writer import build
        result = build(req.description)
        return {
            "route": "build",
            "success": result.success,
            "code": result.code,
            "confidence": result.confidence,
            "coverage_percent": result.coverage_percent,
            "iterations": result.iterations,
            "from_cache": result.from_cache,
            "error": result.error,
        }

    # simple path — derive a meaningful snake_case name from the description
    # by dropping common stopwords rather than blindly taking the first N words,
    # so "Check if a number is prime" -> "check_number_prime" not "check_if_a_number"
    import re as _re
    STOPWORDS = {"a", "an", "the", "is", "if", "of", "to", "for", "and", "or",
                 "given", "from", "with", "in", "on", "this", "that", "it"}
    words = [w for w in _re.findall(r"[a-zA-Z]+", req.description.lower()) if w not in STOPWORDS][:5]
    name = "_".join(words) if words else "auto_capability"
    try:
        result = _capai.run(name, req.description, *req.args, **req.kwargs)
        return {"route": "run", "success": True, "result": result, "capability_name": name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
