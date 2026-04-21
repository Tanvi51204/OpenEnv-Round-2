"""Workday-like app — HR and people operations."""

from typing import Dict, List, Optional
from server.apps.base_app import BaseApp
from server.schema_drift import SchemaDriftEngine


class WorkdayApp(BaseApp):
    APP_NAME = "workday"

    OPERATIONS = [
        "get_employee", "list_employees", "provision_access",
        "log_sla_event", "request_budget_approval",
        "create_onboarding_task", "complete_task",
    ]

    def __init__(self, drift: SchemaDriftEngine):
        super().__init__(drift)
        self._records: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # BaseApp interface
    # ------------------------------------------------------------------

    def initialize(self, records: List[Dict]) -> None:
        self._records = {r["employee_id"]: r for r in records}

    def execute(self, operation: str, args: Dict) -> Dict:
        method = getattr(self, f"_op_{operation}", None)
        if method is None:
            return {
                "success": False,
                "message": f"Unknown operation '{operation}'. Available: {', '.join(self.OPERATIONS)}",
            }
        try:
            return method(**args)
        except TypeError as exc:
            return {"success": False, "message": f"Bad args for '{operation}': {exc}"}

    def get_state_view(self, max_rows: int = 5) -> str:
        pending = [r for r in self._records.values()
                   if r.get("status") == "pending"][:max_rows]
        sample = pending or list(self._records.values())[:max_rows]
        if not sample:
            return "No employee records loaded."
        lines = []
        for rec in sample:
            view = self._to_agent_view(rec)
            keep = ["employee_id", "name",
                    "level", "job_level", "seniority",
                    "manager_id", "reports_to", "direct_manager",
                    "status", "request_status", "approval_state",
                    "department", "territory", "email"]
            compact = {k: v for k, v in view.items() if k in keep and v is not None}
            lines.append(str(compact))
        return "\n".join(lines)

    def count_open_items(self) -> int:
        return sum(1 for r in self._records.values()
                   if r.get("status") == "pending")

    # ------------------------------------------------------------------
    # Workflow completion state checks
    # ------------------------------------------------------------------

    def sla_logged(self) -> bool:
        """True once log_sla_event was called (Workflow A step A5)."""
        return any(r.get("_sla_logged") for r in self._records.values())

    def employee_created(self) -> bool:
        """True once create_onboarding_task was called for EMP-NEW-001 (Workflow B step B1)."""
        return bool(self._records.get("EMP-NEW-001", {}).get("_onboarding_created"))

    def access_provisioned(self, app_name: str) -> bool:
        """True once provision_access was called for the given app (Workflow B step B2)."""
        return any(
            r.get("_access_provisioned", {}).get(app_name)
            for r in self._records.values()
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _op_get_employee(self, employee_id: str) -> Dict:
        rec = self._records.get(employee_id)
        if not rec:
            return {"success": False,
                    "message": f"Employee {employee_id} not found. Use list_employees to browse."}
        return {"success": True, "data": self._to_agent_view(rec),
                "message": f"Retrieved employee {employee_id} ({rec.get('name', '')})"}

    def _op_list_employees(self, department: Optional[str] = None,
                           status: Optional[str] = None,
                           limit: int = 10) -> Dict:
        matching = [
            r for r in self._records.values()
            if (department is None or r.get("department") == department)
            and (status is None or r.get("status") == status)
        ][:limit]
        drifted = [self._to_agent_view(r) for r in matching]
        keep = ["employee_id", "name",
                "level", "job_level", "seniority",
                "manager_id", "reports_to", "direct_manager",
                "status", "request_status", "approval_state",
                "department", "territory"]
        compact = [{k: v for k, v in r.items() if k in keep and v is not None}
                   for r in drifted]
        return {"success": True, "data": compact,
                "message": f"Found {len(compact)} employees"
                           + (f" in {department}" if department else "")}

    def _op_provision_access(self, employee_id: str, app_name: str,
                             **kwargs) -> Dict:
        """Grant app access to an employee (Workflow B step B2)."""
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: use current field name, not '{schema_error}'"}

        rec = self._records.get(employee_id)
        if not rec:
            return {"success": False, "message": f"Employee {employee_id} not found"}

        rec.setdefault("_access_provisioned", {})[app_name] = True
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"Provisioned {app_name} access for {employee_id} ({rec.get('name', '')})"}

    def _op_log_sla_event(self, ticket_id: str, sla_met: bool = True,
                          elapsed_minutes: Optional[float] = None) -> Dict:
        """Log an SLA compliance event (Workflow A step A5)."""
        # Find an employee record to attach the log to
        first = next(iter(self._records.values()), None)
        if first is None:
            return {"success": False, "message": "No Workday records loaded"}

        first["_sla_logged"] = True
        status = "MET" if sla_met else "BREACHED"
        detail = (f" ({elapsed_minutes:.1f} min elapsed)" if elapsed_minutes else "")
        return {
            "success": True,
            "message": f"SLA event logged for {ticket_id}: {status}{detail}",
        }

    def _op_request_budget_approval(self, employee_id: str,
                                    amount: float = 0, reason: str = "") -> Dict:
        """Request budget approval (triggers RBAC / approval threshold check upstream)."""
        rec = self._records.get(employee_id)
        if not rec:
            return {"success": False, "message": f"Employee {employee_id} not found"}
        return {
            "success": True,
            "message": f"Budget approval request submitted for {employee_id}: ${amount:,.0f}",
        }

    def _op_create_onboarding_task(self, employee_id: str, **kwargs) -> Dict:
        """Create onboarding record for a new employee (Workflow B step B1)."""
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: use current field name, not '{schema_error}'"}

        rec = self._records.get(employee_id)
        if not rec:
            # Auto-create a stub record if it doesn't exist yet
            rec = {
                "employee_id":         employee_id,
                "name":                kwargs.get("name", "New Employee"),
                "level":               kwargs.get("level") or kwargs.get("job_level") or kwargs.get("seniority", "IC1"),
                "manager_id":          kwargs.get("manager_id") or kwargs.get("reports_to") or kwargs.get("direct_manager"),
                "status":              "pending",
                "department":          kwargs.get("department", "support"),
                "territory":           kwargs.get("territory", "west"),
                "email":               kwargs.get("email", f"{employee_id.lower()}@company.com"),
                "_access_provisioned": {},
                "_sla_logged":         False,
                "_onboarding_created": True,
            }
            self._records[employee_id] = rec
        else:
            rec["_onboarding_created"] = True

        rec.setdefault("_onboarding_tasks", []).append("onboarding_checklist")
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"Onboarding task created for {employee_id} ({rec.get('name', '')})"}

    def _op_complete_task(self, employee_id: str, task: str) -> Dict:
        rec = self._records.get(employee_id)
        if not rec:
            return {"success": False, "message": f"Employee {employee_id} not found"}
        tasks = rec.get("_onboarding_tasks", [])
        if task in tasks:
            tasks.remove(task)
        return {"success": True,
                "message": f"Completed task '{task}' for {employee_id}"}
