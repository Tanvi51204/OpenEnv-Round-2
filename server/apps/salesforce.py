"""Salesforce-like app — CRM account and pipeline management."""

from typing import Dict, List, Optional
from server.apps.base_app import BaseApp
from server.schema_drift import SchemaDriftEngine


class SalesforceApp(BaseApp):
    APP_NAME = "salesforce"

    OPERATIONS = [
        "get_account", "list_accounts", "update_deal_stage", "flag_churn_risk",
        "assign_account_owner", "log_interaction", "get_opportunity",
    ]

    def __init__(self, drift: SchemaDriftEngine):
        super().__init__(drift)
        self._records: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # BaseApp interface
    # ------------------------------------------------------------------

    def initialize(self, records: List[Dict]) -> None:
        self._records = {r["account_id"]: r for r in records}

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
        at_risk = [r for r in self._records.values()
                   if r.get("health") in ("red", "yellow")][:max_rows]
        sample = at_risk or list(self._records.values())[:max_rows]
        if not sample:
            return "No accounts loaded."
        lines = []
        for rec in sample:
            view = self._to_agent_view(rec)
            keep = ["account_id", "company_name",
                    "deal_stage", "pipeline_stage", "stage",
                    "health", "account_health", "risk_score",
                    "owner", "owner_name", "account_owner", "rep_email",
                    "arr", "annual_recurring_revenue",
                    "is_paying", "territory"]
            compact = {k: v for k, v in view.items() if k in keep and v is not None}
            lines.append(str(compact))
        return "\n".join(lines)

    def count_open_items(self) -> int:
        return sum(1 for r in self._records.values()
                   if r.get("health") in ("red", "yellow") or
                   r.get("deal_stage") in ("prospect", "qualification", "negotiation"))

    # ------------------------------------------------------------------
    # Workflow completion state checks
    # ------------------------------------------------------------------

    def account_checked(self) -> bool:
        """True once get_account was called for the Workflow A customer (Workflow A step A3)."""
        return any(
            r.get("_is_workflow_a_account") and r.get("_account_checked")
            for r in self._records.values()
        )

    def churn_flagged(self) -> bool:
        """True once flag_churn_risk was called for the at-risk account (Workflow C step C1)."""
        return any(
            r.get("_is_churn_target") and r.get("_churn_flagged")
            for r in self._records.values()
        )

    def team_assigned(self) -> bool:
        """Legacy free-pass check — kept for backwards compatibility.
        Workflow B no longer uses this; see new_hire_assigned_in_territory()."""
        return any(r.get("_team_assigned") for r in self._records.values())

    def new_hire_assigned_in_territory(self, employee_id: str, territory: str) -> bool:
        """True once an SF account in `territory` has `employee_id` as its owner
        (Workflow B step B3 — tightened from the free-pass team_assigned check).
        Forces real cross-app data flow: the agent must use the employee_id and territory
        discovered in B1 to satisfy this check."""
        if not employee_id or not territory:
            return False
        return any(
            r.get("territory") == territory
            and r.get("owner") == employee_id
            and r.get("_team_assigned")
            for r in self._records.values()
        )

    def intervention_assigned(self) -> bool:
        """True once assign_account_owner called on the churn-risk account (Workflow C step C4)."""
        return any(
            r.get("_is_churn_target") and r.get("_intervention_assigned")
            for r in self._records.values()
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _op_get_account(self, account_id: str) -> Dict:
        rec = self._records.get(account_id)
        if not rec:
            return {"success": False,
                    "message": f"Account {account_id} not found. Use list_accounts to browse."}
        rec["_account_checked"] = True
        return {"success": True, "data": self._to_agent_view(rec),
                "message": f"Retrieved account {account_id} ({rec.get('company_name', '')})"}

    def _op_list_accounts(self, health: Optional[str] = None,
                          territory: Optional[str] = None,
                          limit: int = 10) -> Dict:
        matching = [
            r for r in self._records.values()
            if (health is None or r.get("health") == health)
            and (territory is None or r.get("territory") == territory)
        ][:limit]
        drifted = [self._to_agent_view(r) for r in matching]
        keep = ["account_id", "company_name",
                "deal_stage", "pipeline_stage", "stage",
                "health", "account_health", "risk_score",
                "owner", "owner_name", "account_owner", "rep_email",
                "arr", "annual_recurring_revenue",
                "is_paying", "territory"]
        compact = [{k: v for k, v in r.items() if k in keep and v is not None}
                   for r in drifted]
        return {"success": True, "data": compact,
                "message": f"Found {len(compact)} accounts"
                           + (f" (health={health})" if health else "")}

    def _op_update_deal_stage(self, account_id: str, amount: float = 0, **kwargs) -> Dict:
        """Note: requires manager approval if amount > threshold (checked by BusinessRuleEngine)."""
        schema_error, schema_adapted = self._check_schema_drift(kwargs)
        if schema_error:
            hint = self._drift.translate_field("deal_stage", self.APP_NAME)
            return {"success": False, "schema_error": schema_error,
                    "message": f"Schema error: use '{hint}' not '{schema_error}'"}

        rec = self._records.get(account_id)
        if not rec:
            return {"success": False, "message": f"Account {account_id} not found"}

        new_stage = (kwargs.get("deal_stage") or kwargs.get("pipeline_stage")
                     or kwargs.get("stage"))
        if not new_stage:
            return {"success": False,
                    "message": "Provide deal_stage / pipeline_stage / stage value"}

        rec["deal_stage"] = new_stage
        return {"success": True, "schema_adapted": schema_adapted,
                "message": f"{account_id} deal stage → '{new_stage}'"}

    def _op_flag_churn_risk(self, account_id: str, reason: Optional[str] = None) -> Dict:
        rec = self._records.get(account_id)
        if not rec:
            return {"success": False, "message": f"Account {account_id} not found"}
        rec["_churn_flagged"] = True
        rec["health"] = "red"
        return {
            "success": True,
            "message": f"Flagged {account_id} ({rec.get('company_name', '')}) as churn risk"
                       + (f": {reason}" if reason else ""),
        }

    def _op_assign_account_owner(self, account_id: str, **kwargs) -> Dict:
            schema_error, schema_adapted = self._check_schema_drift(kwargs)
            if schema_error:
                hint = self._drift.translate_field("owner", self.APP_NAME)
                return {"success": False, "schema_error": schema_error,
                        "message": f"Schema error: use '{hint}' not '{schema_error}'"}

            rec = self._records.get(account_id)
            if not rec:
                return {"success": False, "message": f"Account {account_id} not found"}

            new_owner = (kwargs.get("owner") or kwargs.get("owner_name")
                        or kwargs.get("account_owner") or kwargs.get("rep_email"))
            if not new_owner:
                correct_field = self._drift.translate_field("owner", self.APP_NAME)
                return {"success": False,
                        "message": f"Missing owner field. Use '{correct_field}' as the arg key for this episode."}

            rec["owner"] = new_owner
            rec["_team_assigned"] = True
            # Semantic-marker driven: any churn target getting an owner is an intervention.
            # Replaces the old `account_id == "ACME-003"` hardcoded ID check.
            if rec.get("_is_churn_target"):
                rec["_intervention_assigned"] = True

            return {"success": True, "schema_adapted": schema_adapted,
                    "message": f"{account_id} owner → '{new_owner}'"}
                    
    def _op_log_interaction(self, account_id: str, note: str = "") -> Dict:
        rec = self._records.get(account_id)
        if not rec:
            return {"success": False, "message": f"Account {account_id} not found"}
        rec["_interaction_logged"] = True
        rec.setdefault("interactions", []).append(note)
        return {"success": True,
                "message": f"Logged interaction for {account_id}"}

    def _op_get_opportunity(self, account_id: str) -> Dict:
        rec = self._records.get(account_id)
        if not rec:
            return {"success": False, "message": f"Account {account_id} not found"}
        opp = {
            "account_id":   account_id,
            "company_name": rec.get("company_name"),
            "arr":          rec.get("arr"),
            "deal_stage":   rec.get("deal_stage"),
            "health":       rec.get("health"),
            "is_paying":    rec.get("is_paying"),
        }
        return {"success": True, "data": self._to_agent_view(opp),
                "message": f"Retrieved opportunity for {account_id}"}
