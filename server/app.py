"""
FastAPI application exposing the OpenEnv-compatible HTTP API.
Endpoints: GET /health, GET /metadata, GET /schema,
           POST /reset, POST /step, GET /state, POST /state, GET /docs
"""

from typing import Any, Dict, Optional
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from server.environment import DataCleaningEnvironment

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description="A real-world data cleaning environment for AI agent training.",
    version="0.1.0",
)

# Single shared environment instance (stateful server)
env = DataCleaningEnvironment()


# New reset body accepts workflow_id
class ResetRequest(BaseModel):
    workflow_id: Optional[str] = None  # "A", "B", "C", or None for round-robin


class StepResponse(BaseModel):
    observation: DataCleaningObservation
    reward: float
    done: bool
    info: dict = {}


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "data-cleaning-env",
        "description": (
            "A real-world data cleaning environment where an AI agent fixes "
            "missing values, duplicate rows, format inconsistencies, outliers, "
            "and dtype errors across three progressively harder tasks."
        ),
        "version": "0.1.0",
        "tags": ["openenv", "data-cleaning", "rl", "real-world"],
        "tasks": [
            {"id": "task1", "name": "Fill Missing Values", "difficulty": "easy"},
            {"id": "task2", "name": "Fix Formats and Remove Duplicates", "difficulty": "medium"},
            {"id": "task3", "name": "Full Cleaning Pipeline", "difficulty": "hard"},
        ],
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "fill_missing",
                        "drop_duplicates",
                        "fix_format",
                        "replace_value",
                        "drop_outliers",
                        "fix_dtype",
                    ],
                },
                "column": {"type": "string", "nullable": True},
                "params": {"type": "object", "nullable": True},
            },
            "required": ["operation"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "done":             {"type": "boolean"},
                "reward":           {"type": "number"},
                "data_preview":     {"type": "string"},
                "data_shape":       {"type": "array", "items": {"type": "integer"}},
                "missing_counts":   {"type": "object"},
                "duplicate_count":  {"type": "integer"},
                "dtype_issues":     {"type": "object"},
                "task_description": {"type": "string"},
                "message":          {"type": "string"},
                "step_count":       {"type": "integer"},
                "current_score":    {"type": "number"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id":       {"type": "string"},
                "task_id":          {"type": "integer"},
                "step_count":       {"type": "integer"},
                "max_steps":        {"type": "integer"},
                "total_errors":     {"type": "integer"},
                "errors_remaining": {"type": "integer"},
            },
        },
    }


@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=obs.reward, done=False)


@app.post("/step", response_model=StepResponse)
async def step(body: Dict[str, Any] = Body(...)):
    """
    Accept both openenv-core wrapped format:
        {"action": {"operation": "...", ...}, "timeout_s": 15}
    and direct format (for backward compat with our own client/inference):
        {"operation": "...", "column": "...", "params": {...}}
    """
    action_data = body.get("action", body)
    try:
        action = DataCleaningAction(**action_data)
        obs = env.step(action)
    except (TypeError, KeyError, Exception) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=obs.reward, done=obs.done)


@app.get("/state", response_model=DataCleaningState)
def state_get():
    """GET /state — openenv-core spec."""
    return env.state()


@app.post("/state", response_model=DataCleaningState)
def state_post():
    """POST /state — backward compatibility."""
    return env.state()



@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve the demo dashboard."""
    return FileResponse("ui/index.html")

@app.get("/schema/apps")
def app_schemas():
    """Return the canonical action space per app — used by the UI."""
    return {...}  # maps app → list of operations + their arg schemas


# ------------------------------------------------------------------
# Entry point (required by openenv-core and [project.scripts])
# ------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
