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

# One shared CapAI instance per worker process.
# The registry is persisted to disk so capabilities survive restarts
# (requires a Render Persistent Disk or equivalent mounted at CAPAI_HOME).
_capai = CapAI()


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


@app.post("/run", response_model=RunResponse)
def run_task(req: RunRequest):
    try:
        result = _capai.run(
            name=req.name,
            description=req.description,
            args=req.args,
            kwargs=req.kwargs,
        )
        return RunResponse(success=True, result=result, capability_name=req.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
