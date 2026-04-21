"""Schema drift engine — manages per-episode field-name versioning across all 4 apps."""

import random
from typing import Dict, Optional

# Canonical field → actual field name, per app, per schema version
SCHEMA_MAP = {
    "jira": {
        "v1": {"priority": "priority",     "assignee": "assignee",       "status": "status"},
        "v2": {"priority": "severity",      "assignee": "owner",          "status": "state"},
        "v3": {"priority": "urgency_level", "assignee": "assigned_to",    "status": "current_state",
               "sla_deadline": "due_by"},
    },
    "zendesk": {
        "v1": {"urgency": "urgency",        "agent_email": "agent_email", "state": "state"},
        "v2": {"urgency": "priority",       "agent_email": "handler",     "state": "ticket_state"},
        "v3": {"urgency": "impact_level",   "agent_email": "assigned_agent", "state": "resolution_status"},
    },
    "salesforce": {
        "v1": {"deal_stage": "deal_stage",  "health": "health",           "owner": "owner_name"},
        "v2": {"deal_stage": "pipeline_stage", "health": "account_health","owner": "account_owner"},
        "v3": {"deal_stage": "stage",       "health": "risk_score",       "owner": "rep_email",
               "arr": "annual_recurring_revenue"},
    },
    "workday": {
        "v1": {"level": "level",            "manager_id": "manager_id",   "status": "resolution"},
        "v2": {"level": "job_level",        "manager_id": "reports_to",   "status": "request_status"},
        "v3": {"level": "seniority",        "manager_id": "direct_manager", "status": "approval_state"},
    },
}


class SchemaDriftEngine:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._versions: Dict[str, str] = {app: "v1" for app in SCHEMA_MAP}

    def sample_for_episode(self, episode_num: int) -> None:
        """Sample schema versions deterministically per episode."""
        rng = random.Random(self._seed + episode_num)
        self._versions = {app: rng.choice(["v1", "v2", "v3"]) for app in SCHEMA_MAP}

    def translate_record(self, record: Dict, app: str) -> Dict:
        """Rename canonical field names → current schema's field names (for output to agent)."""
        version = self._versions.get(app, "v1")
        mapping = SCHEMA_MAP.get(app, {}).get(version, {})
        return {mapping.get(k, k): v for k, v in record.items()
                if not k.startswith("_")}  # strip internal state-tracking fields

    def translate_field(self, canonical_field: str, app: str) -> str:
        """Get the current drifted name for a canonical field."""
        version = self._versions.get(app, "v1")
        mapping = SCHEMA_MAP.get(app, {}).get(version, {})
        return mapping.get(canonical_field, canonical_field)

    def check_args_for_drift(self, args: Dict, app: str):
        """
        Check whether action args use canonical (stale) vs drifted (correct) field names.
        Returns (schema_error: Optional[str], schema_adapted: bool).
          - schema_error: the canonical field name the agent incorrectly used, or None
          - schema_adapted: True if agent correctly used a drifted field name
        """
        version = self._versions.get(app, "v1")
        if version == "v1":
            return None, False  # v1 is canonical — no drift, no credit/penalty

        mapping = SCHEMA_MAP.get(app, {}).get(version, {})
        changed = {k: v for k, v in mapping.items() if k != v}      # canonical → drifted
        reverse = {v: k for k, v in changed.items()}                 # drifted → canonical

        for key in args:
            if key in changed:
                return key, False   # Agent used old canonical name on drifted schema → error
            if key in reverse:
                return None, True   # Agent correctly used drifted name → adaptation bonus

        return None, False

    def get_hints(self) -> Dict[str, str]:
        """Return partial schema hints visible in observation.
        Reveals 1 changed field per app that has drift (agent must discover the rest)."""
        hints = {}
        rng = random.Random(self._seed)
        for app, version in self._versions.items():
            mapping = SCHEMA_MAP.get(app, {}).get(version, {})
            changed = {f"{app}.{k}": v for k, v in mapping.items() if k != v}
            if changed:
                key = rng.choice(list(changed.keys()))
                hints[key] = changed[key]
        return hints

    def get_all_changes(self) -> Dict[str, Dict[str, str]]:
        """Return all field changes for every app (used by UI schema drift viewer)."""
        result = {}
        for app, version in self._versions.items():
            mapping = SCHEMA_MAP.get(app, {}).get(version, {})
            result[app] = {k: v for k, v in mapping.items() if k != v}
        return result
