DEFAULT_RULES = {
    "sla_p0_minutes": 30,          # P0 tickets: acknowledge within 30 min
    "sla_p1_hours": 4,             # P1 tickets: first response within 4h
    "approval_threshold": 10_000,  # $ above which manager approval needed
    "max_tickets_per_agent": 10,   # RBAC: agent capacity cap
    "gdpr_max_days": 30,           # compliance: GDPR ticket resolution
    "rbac": {
        "support": {"salesforce": ["read"], "jira": ["read", "create_issue"]},
        "engineer": {"jira": ["*"], "zendesk": ["read"]},
        "manager": {"*": ["*"]},
    }
}

POLICY_DRIFT_EVENTS = {
    "sla_tighten":          {"sla_p0_minutes": 15, "sla_p1_hours": 2},
    "approval_tighten":     {"approval_threshold": 5_000},
    "gdpr_expedite":        {"gdpr_max_days": 7},
}

class BusinessRuleEngine:
    def __init__(self):
        self.rules = DEFAULT_RULES.copy()
        self._violation_log: List[str] = []

    def apply_policy_drift(self, event: str) -> None:
        """Called mid-episode or at episode start to change rules."""
        if event in POLICY_DRIFT_EVENTS:
            self.rules.update(POLICY_DRIFT_EVENTS[event])

    def check_action(self, action: OrgOSAction, context: Dict) -> Tuple[bool, str, float]:
        """Returns (allowed, reason, penalty)."""
        violations = []

        # RBAC check
        role = context.get("agent_role", "support")
        app_perms = self.rules["rbac"].get(role, {})
        allowed_ops = app_perms.get(action.app, app_perms.get("*", []))
        if "*" not in allowed_ops and action.operation not in allowed_ops:
            violations.append(f"RBAC: {role} cannot {action.operation} on {action.app}")
            return False, violations[0], -0.25

        # Approval threshold check
        if action.operation in ("request_budget_approval", "update_deal_stage"):
            amount = action.args.get("amount", 0)
            if amount > self.rules["approval_threshold"] and not context.get("manager_approved"):
                violations.append(f"Approval required: ${amount} > ${self.rules['approval_threshold']}")
                return False, violations[0], -0.10

        self._violation_log.extend(violations)
        return True, "", 0.0

    def check_sla(self, ticket: Dict, elapsed_minutes: float) -> Tuple[bool, float]:
        """Returns (sla_met, penalty)."""
        priority = ticket.get("priority", ticket.get("urgency", "p2"))
        if priority in ("p0", "critical") and elapsed_minutes > self.rules["sla_p0_minutes"]:
            return False, -0.15
        return True, 0.0

    def get_violations_this_step(self) -> List[str]:
        v = self._violation_log.copy()
        self._violation_log.clear()
        return v