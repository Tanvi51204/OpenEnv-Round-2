# Canonical → actual field names per app per schema version
SCHEMA_MAP = {
    "jira": {
        "v1": {"priority": "priority",      "assignee": "assignee",       "status": "status"},
        "v2": {"priority": "severity",       "assignee": "owner",          "status": "state"},
        "v3": {"priority": "urgency_level",  "assignee": "assigned_to",    "status": "current_state",
               "sla_deadline": "due_by"},  # v3 adds a new field
    },
    "zendesk": {
        "v1": {"urgency": "urgency",         "agent_email": "agent_email", "state": "state"},
        "v2": {"urgency": "priority",        "agent_email": "handler",     "state": "ticket_state"},
        "v3": {"urgency": "impact_level",    "agent_email": "assigned_agent","state": "resolution_status"},
    },
    "salesforce": {
        "v1": {"deal_stage": "deal_stage",   "health": "health",           "owner": "owner_name"},
        "v2": {"deal_stage": "pipeline_stage","health": "account_health",  "owner": "account_owner"},
        "v3": {"deal_stage": "stage",        "health": "risk_score",       "owner": "rep_email",
               "arr": "annual_recurring_revenue"},
    },
    "workday": {
        "v1": {"level": "level",             "manager_id": "manager_id",   "status": "resolution"},
        "v2": {"level": "job_level",         "manager_id": "reports_to",   "status": "request_status"},
        "v3": {"level": "seniority",         "manager_id": "direct_manager","status": "approval_state"},
    },
}

class SchemaDriftEngine:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._versions: Dict[str, str] = {}  # app → "v1"/"v2"/"v3"

    def sample_for_episode(self, episode_num: int) -> None:
        """Sample schema versions deterministically per episode."""
        rng = random.Random(self._seed + episode_num)
        self._versions = {app: rng.choice(["v1", "v2", "v3"]) for app in SCHEMA_MAP}

    def translate_record(self, record: Dict, app: str) -> Dict:
        """Rename canonical field names → current schema's field names."""
        version = self._versions.get(app, "v1")
        mapping = SCHEMA_MAP[app][version]
        return {mapping.get(k, k): v for k, v in record.items()}

    def get_hints(self) -> Dict[str, str]:
        """Return partial schema hints visible in observation.
        Only reveal 1 random field per app (agent must probe for the rest)."""
        hints = {}
        rng = random.Random(self._seed)
        for app, version in self._versions.items():
            mapping = SCHEMA_MAP[app][version]
            # Reveal only fields that actually changed (v2/v3)
            changed = {f"{app}.{k}": v for k, v in mapping.items() if k != v}
            if changed:
                key = rng.choice(list(changed.keys()))
                hints[key] = changed[key]
        return hints