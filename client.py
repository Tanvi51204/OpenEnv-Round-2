"""
Synchronous HTTP client for the OrgOS OpenEnv environment.

Usage
-----
    from client import OrgOSEnvClient
    from models import OrgOSAction

    client = OrgOSEnvClient(base_url="http://localhost:8000")

    # Start a new episode (workflow_id "A"/"B"/"C" or None for round-robin)
    result = client.reset(workflow_id="A")
    print(result.observation.workflow_goal)

    # Take a step
    action = OrgOSAction(
        app="zendesk",
        operation="acknowledge_ticket",
        args={"ticket_number": "ZD-001"},
    )
    result = client.step(action)
    print(result.observation.current_score, result.reward, result.done)

    # Inspect state
    state = client.state()
    print(state.episode_id, state.workflow_completion)
"""

from typing import Optional
import httpx
from pydantic import BaseModel

from models import OrgOSAction, OrgOSObservation, OrgOSState


class StepResult(BaseModel):
    """Returned by reset() and step()."""
    observation: OrgOSObservation
    reward: float
    done: bool
    info: dict = {}


class OrgOSEnvClient:
    """
    Thin synchronous wrapper around the OrgOS HTTP API.

    All methods raise httpx.HTTPStatusError on non-2xx responses.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url  = base_url.rstrip("/")
        self._client   = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, workflow_id: Optional[str] = None) -> StepResult:
        """
        Start a new episode.

        Parameters
        ----------
        workflow_id : str | None
            "A" = Customer Bug Fix  (support role)
            "B" = Employee Onboarding  (manager role)
            "C" = Churn Risk Alert  (support role)
            None = round-robin (A → B → C → A …)
        """
        payload = {"workflow_id": workflow_id} if workflow_id is not None else {}
        resp    = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        return StepResult(**resp.json())

    def step(self, action: OrgOSAction) -> StepResult:
        """
        Take one action in the environment.

        Parameters
        ----------
        action : OrgOSAction
            app       : str   – "jira" | "zendesk" | "salesforce" | "workday"
            operation : str   – app-specific operation name
            args      : dict  – operation arguments
        """
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> OrgOSState:
        """Return current episode metadata without modifying state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return OrgOSState(**resp.json())

    def health(self) -> dict:
        """Ping the server. Returns {"status": "healthy"} if healthy."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def app_schemas(self) -> dict:
        """Return per-app operation catalogue."""
        resp = self._client.get("/schema/apps")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self._client.close()
