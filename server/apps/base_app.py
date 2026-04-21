"""Abstract base class for all OrgOS app modules."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from server.schema_drift import SchemaDriftEngine


class BaseApp(ABC):
    APP_NAME: str = ""

    def __init__(self, drift: SchemaDriftEngine):
        self._drift = drift

    # ------------------------------------------------------------------
    # Core interface — every app must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, records: List[Dict]) -> None:
        """Load synthetic records into in-memory state."""

    @abstractmethod
    def execute(self, operation: str, args: Dict) -> Dict:
        """
        Execute an operation.
        Returns dict with at minimum:
          {"success": bool, "message": str}
        May also include:
          {"data": ..., "schema_error": str, "schema_adapted": bool, "ticket": dict}
        """

    @abstractmethod
    def get_state_view(self, max_rows: int = 5) -> str:
        """Return agent-visible snapshot as a compact multi-line string."""

    @abstractmethod
    def count_open_items(self) -> int:
        """Count pending/open work items (used by grader)."""

    # ------------------------------------------------------------------
    # Shared helpers available to all concrete apps
    # ------------------------------------------------------------------

    def _check_schema_drift(self, args: Dict) -> Tuple[Optional[str], bool]:
        """
        Delegate to the drift engine to check if args use stale canonical names.
        Returns (schema_error_field_or_None, schema_adapted_bool).
        """
        return self._drift.check_args_for_drift(args, self.APP_NAME)

    def _to_agent_view(self, record: Dict) -> Dict:
        """Translate a canonical record to the agent-visible drifted representation."""
        return self._drift.translate_record(record, self.APP_NAME)

    def _compact(self, record: Dict, fields: List[str]) -> Dict:
        """Return only the specified fields from a (possibly drifted) record."""
        return {k: v for k, v in record.items() if k in fields and v is not None}
