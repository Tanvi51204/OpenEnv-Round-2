class BaseApp(ABC):
    APP_NAME: str = ""

    # --- Core interface every app must implement ---
    @abstractmethod
    def initialize(self, records: List[Dict]) -> None:
        """Load synthetic records into in-memory state."""

    @abstractmethod
    def execute(self, operation: str, args: Dict) -> Dict:
        """Execute an operation. Returns {"success": bool, "data": ..., "message": str}"""

    @abstractmethod
    def get_state_view(self, max_rows: int = 5) -> str:
        """Return agent-visible snapshot as a compact string."""

    @abstractmethod
    def count_open_items(self) -> int:
        """Count pending/open work items (used by grader)."""