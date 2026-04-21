"""Workflow engine — defines and evaluates multi-app workflow completion."""

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class WorkflowStep:
    step_id: str
    description: str
    app: str
    operation: str
    # Callable that checks if this step was completed given the app states
    completion_check: Callable[[Dict], bool]


# ---------------------------------------------------------------------------
# Workflow A: Customer Bug Fix  (Zendesk → Jira → Salesforce → Workday)
# Agent role: support
# ---------------------------------------------------------------------------
WORKFLOW_A_STEPS = [
    WorkflowStep(
        "A1", "Acknowledge the incoming Zendesk ticket (ZD-001)",
        "zendesk", "acknowledge_ticket",
        lambda apps: apps["zendesk"].ticket_acknowledged(),
    ),
    WorkflowStep(
        "A2", "Escalate to Jira — create a new issue linked to ZD-001",
        "jira", "create_issue",
        lambda apps: apps["jira"].has_linked_issue(),
    ),
    WorkflowStep(
        "A3", "Verify the customer's account status in Salesforce (ACME-001)",
        "salesforce", "get_account",
        lambda apps: apps["salesforce"].account_checked(),
    ),
    WorkflowStep(
        "A4", "Assign the Jira issue to an engineer (JIRA-001)",
        "jira", "assign_owner",
        lambda apps: apps["jira"].issue_assigned(),
    ),
    WorkflowStep(
        "A5", "Log the SLA compliance event in Workday",
        "workday", "log_sla_event",
        lambda apps: apps["workday"].sla_logged(),
    ),
]

# ---------------------------------------------------------------------------
# Workflow B: Employee Onboarding  (Workday → Workday → Salesforce → Zendesk)
# Agent role: manager
# ---------------------------------------------------------------------------
WORKFLOW_B_STEPS = [
    WorkflowStep(
        "B1", "Create the new employee's onboarding record in Workday (EMP-NEW-001)",
        "workday", "create_onboarding_task",
        lambda apps: apps["workday"].employee_created(),
    ),
    WorkflowStep(
        "B2", "Provision Jira access for the new employee via Workday",
        "workday", "provision_access",
        lambda apps: apps["workday"].access_provisioned("jira"),
    ),
    WorkflowStep(
        "B3", "Assign the new employee to the correct Salesforce territory team",
        "salesforce", "assign_account_owner",
        lambda apps: apps["salesforce"].team_assigned(),
    ),
    WorkflowStep(
        "B4", "Create a Zendesk support agent profile for the new employee",
        "zendesk", "assign_agent",
        lambda apps: apps["zendesk"].profile_created(),
    ),
]

# ---------------------------------------------------------------------------
# Workflow C: Churn Risk Alert  (Salesforce → Zendesk → Jira → Salesforce)
# Agent role: support
# ---------------------------------------------------------------------------
WORKFLOW_C_STEPS = [
    WorkflowStep(
        "C1", "Flag at-risk account ACME-003 as churn risk in Salesforce",
        "salesforce", "flag_churn_risk",
        lambda apps: apps["salesforce"].churn_flagged(),
    ),
    WorkflowStep(
        "C2", "Query recent support ticket volume for ACME-003 in Zendesk",
        "zendesk", "get_ticket",
        lambda apps: apps["zendesk"].support_queried("ACME-003"),
    ),
    WorkflowStep(
        "C3", "Check outstanding Jira bugs linked to ACME-003",
        "jira", "list_issues",
        lambda apps: apps["jira"].bugs_checked(),
    ),
    WorkflowStep(
        "C4", "Assign an intervention owner to ACME-003 in Salesforce",
        "salesforce", "assign_account_owner",
        lambda apps: apps["salesforce"].intervention_assigned(),
    ),
]

# ---------------------------------------------------------------------------
# Goal descriptions shown to the agent at reset
# ---------------------------------------------------------------------------
WORKFLOW_GOALS: Dict[str, str] = {
    "A": (
        "Workflow A — Customer Bug Fix: "
        "A P1 bug has been reported via Zendesk (ticket ZD-001) by customer ACME-001. "
        "Steps required: "
        "(1) acknowledge Zendesk ticket ZD-001, "
        "(2) create a new Jira issue linked to ZD-001, "
        "(3) verify ACME-001's account status in Salesforce, "
        "(4) assign the Jira issue (JIRA-001) to an engineer, "
        "(5) log the SLA compliance event in Workday. "
        "Use list operations if you need to discover record IDs."
    ),
    "B": (
        "Workflow B — Employee Onboarding: "
        "A new support engineer has joined the West team. "
        "Employee ID: EMP-NEW-001, Name: Alex Rivera, department: support, territory: west. "
        "Steps required: "
        "(1) create an onboarding record in Workday for EMP-NEW-001, "
        "(2) provision Jira access for EMP-NEW-001 via Workday, "
        "(3) assign EMP-NEW-001 to the correct Salesforce territory (use any ACME-* account in the west region), "
        "(4) create a Zendesk agent profile for EMP-NEW-001. "
        "You have manager-level access."
    ),
    "C": (
        "Workflow C — Churn Risk Alert: "
        "Account ACME-003 (GlobalTech) is showing churn signals. "
        "Steps required: "
        "(1) flag ACME-003 as a churn risk in Salesforce, "
        "(2) query recent support tickets for ACME-003 in Zendesk (use customer_id=ACME-003), "
        "(3) list open Jira bugs related to ACME-003, "
        "(4) assign an intervention owner to ACME-003 in Salesforce. "
        "Focus account: ACME-003."
    ),
}

# Role each workflow expects the agent to act as
WORKFLOW_ROLES: Dict[str, str] = {
    "A": "support",
    "B": "manager",
    "C": "support",
}


class WorkflowEngine:
    WORKFLOWS = {
        "A": WORKFLOW_A_STEPS,
        "B": WORKFLOW_B_STEPS,
        "C": WORKFLOW_C_STEPS,
    }

    def __init__(self):
        self._steps: List[WorkflowStep] = []
        self._completed: List[str] = []
        self._workflow_id: str = "A"

    def start(self, workflow_id: str) -> None:
        """Initialise engine for the given workflow."""
        self._workflow_id = workflow_id
        self._steps = self.WORKFLOWS[workflow_id].copy()
        self._completed = []

    def evaluate(self, apps: Dict) -> float:
        """Check all steps and return completion ratio (0.0–1.0)."""
        if not self._steps:
            return 0.0
        completed = sum(1 for s in self._steps if s.completion_check(apps))
        self._completed = [s.step_id for s in self._steps if s.completion_check(apps)]
        return completed / len(self._steps)

    def get_pending(self) -> List[str]:
        """Return descriptions of not-yet-completed steps."""
        return [s.description for s in self._steps if s.step_id not in self._completed]

    def get_completed(self) -> List[str]:
        """Return step IDs that have been completed."""
        return list(self._completed)

    def get_goal(self) -> str:
        """Return the natural-language goal description for the active workflow."""
        return WORKFLOW_GOALS.get(self._workflow_id, "Complete the assigned workflow.")

    def get_role(self) -> str:
        """Return the expected agent role for RBAC checks."""
        return WORKFLOW_ROLES.get(self._workflow_id, "support")
