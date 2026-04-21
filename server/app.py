"""
FastAPI application — OrgOS OpenEnv HTTP API.

Endpoints (OpenEnv-compatible):
  GET  /health       — liveness probe
  GET  /metadata     — env description
  GET  /schema       — action / observation schema
  POST /reset        — start new episode
  POST /step         — take one action
  GET  /state        — current episode metadata
  POST /state        — same (backward compat)
  GET  /schema/apps  — per-app operation catalogue (used by UI)
  GET  /             — serve the demo dashboard UI
  GET  /ui/run-agent — SSE stream of one inference episode (for UI)
"""

import json
import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import OrgOSAction, OrgOSObservation, OrgOSState
from server.environment import OrgOSEnvironment


# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(
    title="OrgOS — Multi-App Enterprise RL Environment",
    description=(
        "A Salesforce + Zendesk + Jira + Workday simulator for training agents "
        "that handle real enterprise workflows under schema drift and policy changes."
    ),
    version="2.0.0",
)

# Mount static assets (JS, CSS) if the ui/ directory exists
_UI_STATIC = os.path.join(os.path.dirname(__file__), "..", "ui", "static")
if os.path.isdir(_UI_STATIC):
    app.mount("/static", StaticFiles(directory=_UI_STATIC), name="static")

# Single shared environment instance (stateful per-process)
env = OrgOSEnvironment()


# ------------------------------------------------------------------
# Request / response helpers
# ------------------------------------------------------------------

class ResetRequest(BaseModel):
    workflow_id: Optional[str] = None   # "A", "B", "C", or None for round-robin


class StepResponse(BaseModel):
    observation: OrgOSObservation
    reward: float
    done: bool
    info: dict = {}


# ------------------------------------------------------------------
# Core OpenEnv routes
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "env": "orgos", "version": "2.0.0"}


@app.get("/metadata")
def metadata():
    return {
        "name":        "orgos-openenv",
        "description": (
            "OrgOS: multi-app enterprise RL environment. "
            "The agent completes cross-app business workflows (triage, onboarding, churn) "
            "across Jira, Zendesk, Salesforce, and Workday simulators. "
            "Schema drift and policy changes challenge the agent to generalise."
        ),
        "version": "2.0.0",
        "tags": ["openenv", "enterprise", "multi-app", "schema-drift", "rl"],
        "workflows": [
            {
                "id":         "A",
                "name":       "Customer Bug Fix",
                "difficulty": "medium",
                "apps":       ["zendesk", "jira", "salesforce", "workday"],
            },
            {
                "id":         "B",
                "name":       "Employee Onboarding",
                "difficulty": "medium",
                "apps":       ["workday", "salesforce", "zendesk"],
            },
            {
                "id":         "C",
                "name":       "Churn Risk Alert",
                "difficulty": "hard",
                "apps":       ["salesforce", "zendesk", "jira"],
            },
        ],
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "app":       {"type": "string", "enum": ["jira", "zendesk", "salesforce", "workday"]},
                "operation": {"type": "string", "description": "App-specific operation name"},
                "args":      {"type": "object", "description": "Operation arguments"},
            },
            "required": ["app", "operation"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "done":            {"type": "boolean"},
                "reward":          {"type": "number"},
                "current_score":   {"type": "number"},
                "workflow_id":     {"type": "string"},
                "step_count":      {"type": "integer"},
                "app_states":      {"type": "object"},
                "workflow_goal":   {"type": "string"},
                "completed_steps": {"type": "array"},
                "pending_steps":   {"type": "array"},
                "schema_hints":    {"type": "object"},
                "active_rules":    {"type": "object"},
                "rule_violations": {"type": "array"},
                "reward_breakdown":{"type": "object"},
                "message":         {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id":           {"type": "string"},
                "workflow_id":          {"type": "string"},
                "schema_versions":      {"type": "object"},
                "step_count":           {"type": "integer"},
                "max_steps":            {"type": "integer"},
                "rule_violation_count": {"type": "integer"},
                "workflow_completion":  {"type": "number"},
                "rule_compliance_rate": {"type": "number"},
                "policy_drift_active":  {"type": "boolean"},
            },
        },
    }


@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest = Body(default=ResetRequest())):
    try:
        obs = env.reset(workflow_id=req.workflow_id)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(observation=obs, reward=obs.reward, done=False)


@app.post("/step", response_model=StepResponse)
async def step(body: Dict[str, Any] = Body(...)):
    """
    Accept both openenv-core wrapped format:
        {"action": {"app": "...", "operation": "...", "args": {...}}, "timeout_s": 15}
    and direct format:
        {"app": "...", "operation": "...", "args": {...}}
    """
    action_data = body.get("action", body)
    try:
        action = OrgOSAction(**action_data)
        obs    = env.step(action)
    except (TypeError, KeyError, Exception) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return StepResponse(observation=obs, reward=obs.reward, done=obs.done)


@app.get("/state", response_model=OrgOSState)
def state_get():
    """GET /state — openenv-core spec."""
    return env.state()


@app.post("/state", response_model=OrgOSState)
def state_post():
    """POST /state — backward compatibility."""
    return env.state()


# ------------------------------------------------------------------
# UI helper routes
# ------------------------------------------------------------------

@app.get("/schema/apps")
def app_schemas():
    """Return per-app operation catalogue. Used by the dashboard UI."""
    from server.apps.jira import JiraApp
    from server.apps.zendesk import ZendeskApp
    from server.apps.salesforce import SalesforceApp
    from server.apps.workday import WorkdayApp
    return {
        "jira":       {"operations": JiraApp.OPERATIONS},
        "zendesk":    {"operations": ZendeskApp.OPERATIONS},
        "salesforce": {"operations": SalesforceApp.OPERATIONS},
        "workday":    {"operations": WorkdayApp.OPERATIONS},
    }


@app.get("/ui/run-agent")
async def run_agent_sse(workflow_id: str = "A", model: str = "gpt-4o-mini"):
    """
    Server-Sent Events stream.
    Runs one inference episode and streams step events to the UI.
    Each event is: data: <json>\n\n
    """
    import asyncio

    async def _event_stream():
        import json as _json
        from inference import run_workflow_generator
        try:
            async for event in run_workflow_generator(workflow_id=workflow_id, env_ref=env):
                yield f"data: {_json.dumps(event)}\n\n"
                await asyncio.sleep(0)   # yield control
        except Exception as exc:
            yield f"data: {_json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve the OrgOS demo dashboard."""
    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    # Minimal inline fallback if ui/ hasn't been built yet
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>OrgOS Dashboard</title>
<style>body{font-family:monospace;background:#0f172a;color:#94a3b8;padding:2rem}
h1{color:#38bdf8}a{color:#38bdf8}</style></head>
<body>
<h1>OrgOS — Enterprise RL Environment</h1>
<p>The full dashboard UI is at <code>ui/index.html</code>.</p>
<p>API docs: <a href="/docs">/docs</a> &nbsp;|&nbsp;
   Health: <a href="/health">/health</a></p>
</body></html>
""")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
