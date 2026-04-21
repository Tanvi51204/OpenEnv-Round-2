"""
Synchronous HTTP client for the Data Cleaning OpenEnv environment.

Usage
-----
    from client import DataCleaningEnvClient, DataCleaningAction

    client = DataCleaningEnvClient(base_url="http://localhost:8000")

    # Start a new episode (task_id 1/2/3 or omit for round-robin)
    result = client.reset(task_id=1)
    print(result.observation.task_description)

    # Take a step
    action = DataCleaningAction(
        operation="fill_missing",
        column="salary",
        params={"strategy": "median"},
    )
    result = client.step(action)
    print(result.observation.current_score, result.reward, result.done)

    # Inspect state
    state = client.state()
    print(state.episode_id, state.errors_remaining)
"""

from typing import Optional
import httpx
from pydantic import BaseModel

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState


class StepResult(BaseModel):
    """Returned by reset() and step()."""
    observation: DataCleaningObservation
    reward: float
    done: bool
    info: dict = {}


class DataCleaningEnvClient:
    """
    Thin synchronous wrapper around the DataCleaning HTTP API.

    All methods raise httpx.HTTPStatusError on non-2xx responses.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client   = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[int] = None) -> StepResult:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : int | None
            1 = Easy   (fill missing values)
            2 = Medium (fix formats + duplicates)
            3 = Hard   (full pipeline)
            None = round-robin (1 → 2 → 3 → 1 …)
        """
        payload = {"task_id": task_id} if task_id is not None else {}
        resp    = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        return StepResult(**resp.json())

    def step(self, action: DataCleaningAction) -> StepResult:
        """
        Apply one cleaning operation and return the updated observation.

        Parameters
        ----------
        action : DataCleaningAction
            operation : str   – one of fill_missing / drop_duplicates /
                                fix_format / replace_value / drop_outliers / fix_dtype
            column    : str   – target column (optional for drop_duplicates)
            params    : dict  – operation-specific parameters
        """
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> DataCleaningState:
        """Return current episode metadata without modifying state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return DataCleaningState(**resp.json())

    def health(self) -> dict:
        """Ping the server. Returns {"status": "ok"} if healthy."""
        resp = self._client.get("/health")
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
