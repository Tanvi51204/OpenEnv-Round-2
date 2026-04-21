"""Business rule engine — RBAC, SLA checks, approval thresholds, policy drift."""

from typing import Dict, List, Tuple

from models import OrgOSAction


DEFAULT_RULES: Dict = {
    "sla_p0_minutes":       30,       # P0 tickets: acknowledge within 30 min
    "sla_p1_hours":         4,        # P1 tickets: first response within 4 h
    "approval_threshold":   10_000,   # $ above which manager approval is needed
    "max_tickets_per_agent": 10,      # RBAC: agent capacity cap
    "gdpr_max_days":        30,       # GDPR ticket resolution SLA
    "rbac": {
        # Support engineers — can complete Workflows A and C
        "support": {
            "zendesk":    ["*"],      # full ticket lifecycle
            "jira":       ["*"],      # full issue lifecycle
            "salesforce": [
                "get_account", "list_accounts", "get_opportunity",
                "log_interaction", "flag_churn_risk", "assign_account_owner",
            ],
            "workday":    [
                "get_employee", "list_employees", "log_sla_event",
            ],
        },
        # Engineers — focused on Jira + limited Zendesk/Salesforce reads
        "engineer": {
            "jira":       ["*"],
            "zendesk":    ["get_ticket", "list_tickets", "add_note", "resolve_ticket"],
            "salesforce": ["get_account", "list_accounts"],
            "workday":    ["get_employee"],
        },
        # Managers — full access to all apps (Workflow B)
        "manager": {"*": ["*"]},
    },
}

POLICY_DRIFT_EVENTS: Dict = {
    "sla_tighten":       {"sla_p0_minutes": 15, "sla_p1_hours": 2},
    "approval_tighten":  {"approval_threshold": 5_000},
    "gdpr_expedite":     {"gdpr_max_days": 7},
}


class BusinessRuleEngine:
    def __init__(self):
        import copy
        self.rules = copy.deepcopy(DEFAULT_RULES)
        self._violation_log: List[str] = []

    # ------------------------------------------------------------------
    # Policy drift
    # ------------------------------------------------------------------

    def apply_policy_drift(self, event: str) -> None:
        """Called mid-episode or at episode start to change rules."""
        if event in POLICY_DRIFT_EVENTS:
            self.rules.update(POLICY_DRIFT_EVENTS[event])

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def check_action(self, action: OrgOSAction, context: Dict) -> Tuple[bool, str, float]:
        """
        Returns (allowed, reason, penalty).

        penalty values:
          -0.25  RBAC violation
          -0.10  approval threshold exceeded without manager approval
        """
        role = context.get("agent_role", "support")
        app_perms = self.rules["rbac"].get(role, {})

        # Wildcard role (manager) → always allowed
        if "*" in app_perms and "*" in app_perms.get("*", []):
            pass  # fall through to approval check
        else:
            allowed_ops = app_perms.get(action.app, app_perms.get("*", []))
            if "*" not in allowed_ops and action.operation not in allowed_ops:
                reason = f"RBAC: '{role}' cannot run '{action.operation}' on '{action.app}'"
                self._violation_log.append(reason)
                return False, reason, -0.25

        # Approval threshold check
        if action.operation in ("request_budget_approval", "update_deal_stage"):
            amount = action.args.get("amount", 0)
            if amount > self.rules["approval_threshold"] and not context.get("manager_approved"):
                reason = (
                    f"Approval required: ${amount:,.0f} exceeds "
                    f"${self.rules['approval_threshold']:,.0f} threshold"
                )
                self._violation_log.append(reason)
                return False, reason, -0.10

        return True, "", 0.0

    # ------------------------------------------------------------------
    # SLA checks
    # ------------------------------------------------------------------

    def check_sla(self, ticket: Dict, elapsed_minutes: float) -> Tuple[bool, float]:
        """Returns (sla_met, penalty)."""
        priority = ticket.get("priority", ticket.get("urgency", "p2"))
        if priority in ("p0", "critical") and elapsed_minutes > self.rules["sla_p0_minutes"]:
            return False, -0.15
        if priority in ("p1", "high") and elapsed_minutes > self.rules["sla_p1_hours"] * 60:
            return False, -0.10
        return True, 0.0

    # ------------------------------------------------------------------
    # Violation log
    # ------------------------------------------------------------------

    def get_violations_this_step(self) -> List[str]:
        """Return and clear the per-step violation log."""
        v = self._violation_log.copy()
        self._violation_log.clear()
        return v

    def get_active_rules_summary(self) -> Dict:
        """Return scalar rules for inclusion in observation."""
        return {
            "sla_p0_minutes":    self.rules["sla_p0_minutes"],
            "sla_p1_hours":      self.rules["sla_p1_hours"],
            "approval_threshold": self.rules["approval_threshold"],
            "gdpr_max_days":     self.rules["gdpr_max_days"],
        }
