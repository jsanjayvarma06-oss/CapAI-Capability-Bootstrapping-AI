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

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .__init__ import CapAI

app = FastAPI(
    title="CapAI",
    description="Self-expanding capability acquisition layer for AI systems.",
    version="0.1.0",
)

_capai = CapAI()


@app.on_event("startup")
async def startup_cleanup():
    """On startup, remove any registry entries that have no source code (broken entries)."""
    broken = [
        cap.name for cap in _capai.registry._capabilities.values()
        if not getattr(cap, "source_code", None)
    ]
    for name in broken:
        _capai.registry._capabilities.pop(name, None)
    if broken:
        print(f"[startup] Cleared {len(broken)} broken registry entries: {broken}")


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


class BuildResponse(BaseModel):
    success: bool
    code: str = ""
    test_code: str = ""
    iterations: int = 0
    log: List[str] = []
    error: str = ""


@app.post("/build", response_model=BuildResponse)
def build_advanced(req: BuildRequest):
    """
    Build bigger-than-a-single-function code: multi-function modules,
    classes, algorithms, or pipelines. Writes the code, auto-generates a
    test suite, runs both in a sandbox, and self-corrects on failure —
    all in one request, no human-in-the-loop refinement.
    """
    from .advanced_writer import build
    result = build(req.description, max_iterations=req.max_iterations)
    return BuildResponse(
        success=result.success,
        code=result.code,
        test_code=result.test_code,
        iterations=result.iterations,
        log=result.log,
        error=result.error,
    )
