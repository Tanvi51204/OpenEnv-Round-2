"""Zendesk-like app — customer support ticket management."""

from typing import Dict, List, Optional
from server.apps.base_app import BaseApp
from server.schema_drift import SchemaDriftEngine


class ZendeskApp(BaseApp):
    APP_NAME = "zendesk"

    OPERATIONS = [
        "get_ticket", "acknowledge_ticket", "set_urgency", "assign_agent",
        "escalate_to_jira", "resolve_ticket", "add_note", "list_tickets",
    ]

    def __init__(self, drift: SchemaDriftEngine):
        super().__init__(drift)
        self._records: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # BaseApp interface
    # ------------------------------------------------------------------

    def initialize(self, records: List[Dict]) -> None:
        self._records = {r["ticket_number"]: r for r in records}

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
        open_tickets = [r for r in self._records.values()
                        if r.get("state") not in ("resolved", "closed")][:max_rows]
        if not open_tickets:
            return "No open tickets."
        lines = []
        for rec in open_tickets:
            view = self._to_agent_view(rec)
            keep = ["ticket_number", "title",
                    "urgency", "priority", "impact_level",
                    "agent_email", "handler", "assigned_agent",
                    "state", "ticket_state", "resolution_status",
                    "customer_id"]
            compact = {k: v for k, v in view.items() if k in keep and v is not None}
            lines.append(str(compact))
        return "\n".join(lines)

    def count_open_items(self) -> int:
        return sum(1 for r in self._records.values()
                   if r.get("state") not in ("resolved", "closed"))

    # ------------------------------------------------------------------
    # Workflow completion state checks
    # ------------------------------------------------------------------

    def ticket_acknowledged(self) -> bool:
        """True once ZD-001 has been acknowledged (Workflow A step A1)."""
        return bool(self._records.get("ZD-001", {}).get("_acknowledged"))

    def support_queried(self, account_id: str) -> bool:
        """True once tickets for account_id were listed (Workflow C step C2)."""
        return account_id in self._records.get("ZD-001", {}).get("_queried_accounts", []) or \
               any(account_id in r.get("_queried_accounts", []) for r in self._records.values())

    def profile_created(self) -> bool:
        """True once a new agent profile was created (Workflow B step B4)."""
        return any(r.get("_profile_created") for r in self._records.values())

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _op_get_ticket(self, ticket_number: str, customer_id: Optional[str] = None) -> Dict:
        # If customer_id provided, look up all tickets for that customer
        if customer_id:
            matching = [r for r in self._records.values()
                        if r.get("customer_id") == customer_id]
            # Mark as queried for Workflow C
            for r in matching:
                r.setdefault("_queried_accounts", [])
                if customer_id not in r["_queried_accounts"]:
                    r["_queried_accounts"].append(customer_id)
            if not matching:
                return {"success": True, "data": [],
                        "message": f"No tickets found for customer {customer_id}"}
            return {
                "success": True,
                "data": [self._to_agent_view(r) for r in matching[:5]],
                "message": f"Found {len(matching)} tickets for {customer_id}",
            }

        rec = self._records.get(ticket_number)
        if not rec:
            return {"success": False,
                    "message": f"Ticket {ticket_number} not found. Use list_tickets to browse."}
        rec.setdefault("_queried_accounts", [])
        cid = rec.get("customer_id")
        if cid and cid not in rec["_queried_accounts"]:
            rec["_queried_accounts"].append(cid)

        return {"success": True, "data": self._to_agent_view(rec),
                "ticket": rec,
                "message": f"Retrieved {ticket_number}"}

    def _op_acknowledge_ticket(self, ticket_number: str) -> Dict:
        rec = self._records.get(ticket_number)
        if not rec:
            return {"success": False, "message": f"Ticket {ticket_number} not found"}
        rec["_acknowledged"] = True
        if rec.get("state") == "new":
            rec["state"] = "open"
        return {"success": True, "ticket": rec,
                "message": f"Acknowledged {ticket_number} — status → open"}

    def _op_set_urgency(self, ticket_number: str, **kwargs) -> Dict:
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            hint = self._drift.translate_field("urgency", self.APP_NAME)
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: use '{hint}' not '{schema_error}'"}

        rec = self._records.get(ticket_number)
        if not rec:
            return {"success": False, "message": f"Ticket {ticket_number} not found"}

        new_urgency = (kwargs.get("urgency") or kwargs.get("priority")
                       or kwargs.get("impact_level"))
        if not new_urgency:
            return {"success": False,
                    "message": "Provide urgency / priority / impact_level value"}

        rec["urgency"] = new_urgency
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"{ticket_number} urgency → '{new_urgency}'"}

    def _op_assign_agent(self, ticket_number: str, **kwargs) -> Dict:
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            hint = self._drift.translate_field("agent_email", self.APP_NAME)
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: use '{hint}' not '{schema_error}'"}

        rec = self._records.get(ticket_number)
        # For Workflow B profile creation: allow creating a new agent entry
        if not rec:
            # Create a minimal profile record for the new agent
            email = (kwargs.get("agent_email") or kwargs.get("handler")
                     or kwargs.get("assigned_agent"))
            if not email:
                return {"success": False, "message": f"Ticket {ticket_number} not found"}
            # Create a synthetic profile ticket
            profile_rec = {
                "ticket_number":    ticket_number,
                "title":            "Agent profile",
                "urgency":          "p3",
                "agent_email":      email,
                "state":            "closed",
                "customer_id":      None,
                "_acknowledged":    False,
                "_queried_accounts": [],
                "_profile_created": True,
            }
            self._records[ticket_number] = profile_rec
            return {"success": True, "schema_adapted": schema_adapted,
                    "message": f"Created Zendesk profile for agent '{email}'"}

        email = (kwargs.get("agent_email") or kwargs.get("handler")
                 or kwargs.get("assigned_agent"))
        if not email:
            return {"success": False,
                    "message": "Provide agent_email / handler / assigned_agent value"}

        rec["agent_email"] = email
        rec["_profile_created"] = True
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"{ticket_number} assigned to '{email}'"}

    def _op_escalate_to_jira(self, ticket_number: str,
                              jira_issue_id: Optional[str] = None) -> Dict:
        rec = self._records.get(ticket_number)
        if not rec:
            return {"success": False, "message": f"Ticket {ticket_number} not found"}
        rec["state"] = "pending"
        rec["escalated_to_jira"] = jira_issue_id or "pending"
        return {"success": True,
                "message": f"{ticket_number} escalated to Jira"
                           + (f" ({jira_issue_id})" if jira_issue_id else "")}

    def _op_resolve_ticket(self, ticket_number: str) -> Dict:
        rec = self._records.get(ticket_number)
        if not rec:
            return {"success": False, "message": f"Ticket {ticket_number} not found"}
        rec["state"] = "resolved"
        return {"success": True, "message": f"{ticket_number} resolved"}

    def _op_add_note(self, ticket_number: str, note: str) -> Dict:
        rec = self._records.get(ticket_number)
        if not rec:
            return {"success": False, "message": f"Ticket {ticket_number} not found"}
        rec.setdefault("notes", []).append(note)
        return {"success": True, "message": f"Note added to {ticket_number}"}

    def _op_list_tickets(self, state: str = "open", customer_id: Optional[str] = None,
                         limit: int = 10) -> Dict:
        matching = [
            r for r in self._records.values()
            if (state == "all" or r.get("state") == state)
            and (customer_id is None or r.get("customer_id") == customer_id)
        ][:limit]
        # Mark accounts as queried
        if customer_id:
            for r in matching:
                r.setdefault("_queried_accounts", [])
                if customer_id not in r["_queried_accounts"]:
                    r["_queried_accounts"].append(customer_id)

        drifted = [self._to_agent_view(r) for r in matching]
        keep = ["ticket_number", "title",
                "urgency", "priority", "impact_level",
                "agent_email", "handler", "assigned_agent",
                "state", "ticket_state", "resolution_status",
                "customer_id"]
        compact = [{k: v for k, v in r.items() if k in keep and v is not None}
                   for r in drifted]
        return {
            "success": True,
            "data": compact,
            "message": f"Found {len(compact)} {state} tickets"
                       + (f" for {customer_id}" if customer_id else ""),
        }
