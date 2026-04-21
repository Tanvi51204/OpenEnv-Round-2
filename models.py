# models.py

class OrgOSAction(BaseModel):
    app: str           # "jira" | "zendesk" | "salesforce" | "workday"
    operation: str     # app-specific operation name
    args: Dict[str, Any] = {}

class RewardBreakdown(BaseModel):
    workflow_completion: float = 0.0   # 0.30 weight
    rule_compliance: float = 0.0       # 0.25 weight
    schema_adaptation: float = 0.0     # 0.20 weight
    efficiency: float = 0.0            # 0.15 weight
    policy_drift_handling: float = 0.0 # 0.10 weight

class OrgOSObservation(BaseModel):
    done: bool
    reward: float
    current_score: float
    workflow_id: str               # "A", "B", or "C"
    step_count: int
    # Per-app state views (what the agent sees)
    app_states: Dict[str, str]     # app_name → CSV/JSON string preview
    # Workflow progress
    workflow_goal: str
    completed_steps: List[str]
    pending_steps: List[str]
    # Schema drift info (partial — agent must probe to discover rest)
    schema_hints: Dict[str, str]   # e.g. {"jira.priority": "severity"}
    # Business rules in effect this episode
    active_rules: Dict[str, Any]   # {"sla_p0_minutes": 15, "approval_threshold": 5000}
    # Per-step feedback
    rule_violations: List[str]     # violations that just occurred
    reward_breakdown: RewardBreakdown
    message: str

class OrgOSState(BaseModel):
    episode_id: str
    workflow_id: str
    schema_versions: Dict[str, str]     # {"jira": "v2", "zendesk": "v1", ...}
    step_count: int
    max_steps: int
    rule_violation_count: int
    workflow_completion: float
    rule_compliance_rate: float
    policy_drift_active: bool