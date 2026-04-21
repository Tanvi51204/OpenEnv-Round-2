"""Jira-like app — engineering ticket management."""

from typing import Dict, List, Optional
from server.apps.base_app import BaseApp
from server.schema_drift import SchemaDriftEngine


class JiraApp(BaseApp):
    APP_NAME = "jira"

    OPERATIONS = [
        "get_issue", "create_issue", "update_status", "set_priority",
        "assign_owner", "add_label", "link_zendesk_ticket", "close_issue", "list_issues",
    ]

    def __init__(self, drift: SchemaDriftEngine):
        super().__init__(drift)
        self._records: Dict[str, Dict] = {}
        # Workflow completion state tracking
        self._linked_issues: set = set()    # issue_ids linked to a Zendesk ticket
        self._assigned_issues: set = set()  # issue_ids with a non-null assignee
        self._bugs_checked: bool = False    # list_issues was called (Workflow C)

    # ------------------------------------------------------------------
    # BaseApp interface
    # ------------------------------------------------------------------

    def initialize(self, records: List[Dict]) -> None:
        self._records = {r["issue_id"]: r for r in records}
        self._linked_issues.clear()
        self._assigned_issues.clear()
        self._bugs_checked = False
        # Seed state from loaded data
        for issue_id, rec in self._records.items():
            if rec.get("assignee"):
                self._assigned_issues.add(issue_id)
            if rec.get("linked_zendesk"):
                self._linked_issues.add(issue_id)

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
        open_issues = [r for r in self._records.values()
                       if r.get("status") not in ("closed",)][:max_rows]
        if not open_issues:
            return "No open issues."
        lines = []
        for rec in open_issues:
            view = self._to_agent_view(rec)
            keep = ["issue_id", "title",
                    "priority", "severity", "urgency_level",
                    "assignee", "owner", "assigned_to",
                    "status", "state", "current_state",
                    "customer_id", "linked_zendesk"]
            compact = {k: v for k, v in view.items() if k in keep and v is not None}
            lines.append(str(compact))
        return "\n".join(lines)

    def count_open_items(self) -> int:
        return sum(1 for r in self._records.values() if r.get("status") != "closed")

    # ------------------------------------------------------------------
    # Workflow completion state checks
    # ------------------------------------------------------------------

    def has_linked_issue(self) -> bool:
        """True once any issue is linked to a Zendesk ticket (Workflow A step A2)."""
        return len(self._linked_issues) > 0

    def issue_assigned(self) -> bool:
        """True once JIRA-001 (primary bug) has an assignee (Workflow A step A4)."""
        return bool(self._records.get("JIRA-001", {}).get("assignee"))

    def bugs_checked(self) -> bool:
        """True once list_issues has been called (Workflow C step C3)."""
        return self._bugs_checked

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _op_get_issue(self, issue_id: str) -> Dict:
        rec = self._records.get(issue_id)
        if not rec:
            return {"success": False, "message": f"Issue {issue_id} not found. Use list_issues to browse."}
        return {"success": True, "data": self._to_agent_view(rec),
                "message": f"Retrieved {issue_id}"}

    def _op_create_issue(self, title: str, **kwargs) -> Dict:
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            return {
                "success": False,
                "schema_error": schema_error,
                "message": (f"Schema error: field '{schema_error}' is not in the current schema. "
                            f"Check schema_hints for the correct field name."),
            }

        issue_id = f"JIRA-{len(self._records) + 1:03d}"
        # Accept both canonical and drifted names for priority / assignee
        priority = (kwargs.get("priority") or kwargs.get("severity")
                    or kwargs.get("urgency_level", "p2"))
        linked   = kwargs.get("linked_zendesk") or kwargs.get("zendesk_ticket")

        rec = {
            "issue_id":       issue_id,
            "title":          title,
            "priority":       priority,
            "assignee":       kwargs.get("assignee") or kwargs.get("owner") or kwargs.get("assigned_to"),
            "status":         "open",
            "reporter":       kwargs.get("reporter", "agent"),
            "customer_id":    kwargs.get("customer_id"),
            "linked_zendesk": linked,
            "labels":         [],
            "created_at":     "2026-04-21T09:00:00",
        }
        self._records[issue_id] = rec

        if linked:
            self._linked_issues.add(issue_id)
        if rec["assignee"]:
            self._assigned_issues.add(issue_id)

        return {
            "success": True,
            "data": {"issue_id": issue_id},
            "schema_adapted": schema_adapted,
            "message": f"Created {issue_id}: '{title}'"
                       + (f" linked to {linked}" if linked else ""),
        }

    def _op_update_status(self, issue_id: str, **kwargs) -> Dict:
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: use current field name, not '{schema_error}'"}

        rec = self._records.get(issue_id)
        if not rec:
            return {"success": False, "message": f"Issue {issue_id} not found"}

        new_status = (kwargs.get("status") or kwargs.get("state")
                      or kwargs.get("current_state"))
        if not new_status:
            return {"success": False, "message": "Provide status/state/current_state value"}

        rec["status"] = new_status
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"{issue_id} status → '{new_status}'"}

    def _op_set_priority(self, issue_id: str, **kwargs) -> Dict:
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: '{schema_error}' is a stale field name"}

        rec = self._records.get(issue_id)
        if not rec:
            return {"success": False, "message": f"Issue {issue_id} not found"}

        new_priority = (kwargs.get("priority") or kwargs.get("severity")
                        or kwargs.get("urgency_level"))
        if not new_priority:
            return {"success": False,
                    "message": "Provide priority / severity / urgency_level value"}

        rec["priority"] = new_priority
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"{issue_id} priority → '{new_priority}'"}

    def _op_assign_owner(self, issue_id: str, **kwargs) -> Dict:
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            hint = self._drift.translate_field("assignee", self.APP_NAME)
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: use '{hint}' instead of '{schema_error}'"}

        rec = self._records.get(issue_id)
        if not rec:
            return {"success": False, "message": f"Issue {issue_id} not found"}

        assignee = (kwargs.get("assignee") or kwargs.get("owner")
                    or kwargs.get("assigned_to"))
        if not assignee:
            return {"success": False,
                    "message": "Provide assignee / owner / assigned_to value"}

        rec["assignee"] = assignee
        self._assigned_issues.add(issue_id)
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"{issue_id} assigned to '{assignee}'"}

    def _op_add_label(self, issue_id: str, label: str) -> Dict:
        rec = self._records.get(issue_id)
        if not rec:
            return {"success": False, "message": f"Issue {issue_id} not found"}
        rec.setdefault("labels", []).append(label)
        return {"success": True, "message": f"Added label '{label}' to {issue_id}"}

    def _op_link_zendesk_ticket(self, issue_id: str, zendesk_ticket_number: str) -> Dict:
        rec = self._records.get(issue_id)
        if not rec:
            return {"success": False, "message": f"Issue {issue_id} not found"}
        rec["linked_zendesk"] = zendesk_ticket_number
        self._linked_issues.add(issue_id)
        return {"success": True,
                "message": f"Linked {issue_id} ↔ Zendesk {zendesk_ticket_number}"}

    def _op_close_issue(self, issue_id: str) -> Dict:
        rec = self._records.get(issue_id)
        if not rec:
            return {"success": False, "message": f"Issue {issue_id} not found"}
        rec["status"] = "closed"
        return {"success": True, "message": f"Closed {issue_id}"}

    def _op_list_issues(self, status: str = "open", customer_id: Optional[str] = None,
                        limit: int = 10) -> Dict:
        self._bugs_checked = True
        matching = [
            r for r in self._records.values()
            if (status == "all" or r.get("status") == status)
            and (customer_id is None or r.get("customer_id") == customer_id)
        ][:limit]
        drifted = [self._to_agent_view(r) for r in matching]
        keep = ["issue_id", "title", "priority", "severity", "urgency_level",
                "assignee", "owner", "assigned_to",
                "status", "state", "current_state",
                "customer_id", "linked_zendesk"]
        compact = [{k: v for k, v in r.items() if k in keep and v is not None}
                   for r in drifted]
        return {"success": True, "data": compact,
                "message": f"Found {len(compact)} {status} issues"
                           + (f" for {customer_id}" if customer_id else "")}
