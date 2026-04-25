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
        "A1", "Find and acknowledge the new P1 support ticket in Zendesk",
        "zendesk", "acknowledge_ticket",
        lambda apps: apps["zendesk"].ticket_acknowledged(),
    ),
    WorkflowStep(
        "A2", "Create a new Jira issue linked to that Zendesk ticket",
        "jira", "create_issue",
        lambda apps: apps["jira"].has_linked_issue(),
    ),
    WorkflowStep(
        "A3", "Verify the customer's account status in Salesforce",
        "salesforce", "get_account",
        lambda apps: apps["salesforce"].account_checked(),
    ),
    WorkflowStep(
        "A4", "Assign the pre-existing Jira bug for this customer to an engineer",
        "jira", "assign_owner",
        lambda apps: apps["jira"].issue_assigned(),
    ),
    WorkflowStep(
        "A5", "Log the SLA compliance event in Workday using the ticket ID",
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
        "B1", "Find the pending new hire in Workday and create their onboarding record",
        "workday", "create_onboarding_task",
        lambda apps: apps["workday"].employee_created(),
    ),
    WorkflowStep(
        "B2", "Provision Jira access for the new employee via Workday",
        "workday", "provision_access",
        lambda apps: apps["workday"].access_provisioned("jira"),
    ),
    WorkflowStep(
        "B3", "Assign the new employee to a Salesforce territory account (west region)",
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
        "C1", "Identify and flag the at-risk account as churn risk in Salesforce",
        "salesforce", "flag_churn_risk",
        lambda apps: apps["salesforce"].churn_flagged(),
    ),
    WorkflowStep(
        "C2", "Query recent support tickets for the at-risk account in Zendesk",
        "zendesk", "get_ticket",
        lambda apps: apps["zendesk"].support_queried("ACME-003"),
    ),
    WorkflowStep(
        "C3", "List open Jira bugs linked to the at-risk account",
        "jira", "list_issues",
        lambda apps: apps["jira"].bugs_checked(),
    ),
    WorkflowStep(
        "C4", "Assign an intervention owner to the at-risk account in Salesforce",
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
        "A P1 bug has been escalated through the support queue. "
        "Investigate the open ticket, escalate it to the engineering tracker, "
        "verify the affected customer's account health, ensure the issue has an assigned owner, "
        "and record SLA compliance. "
        "Use list operations to discover relevant record IDs before acting."
    ),
    "B": (
        "Workflow B — Employee Onboarding: "
        "A new support engineer has joined the West team and needs to be fully set up. "
        "Ensure their employment record exists, provision the appropriate tooling access, "
        "assign them to the correct territory in your CRM, and create their support profile. "
        "Query the relevant systems to identify the new employee and required accounts."
    ),
    "C": (
        "Workflow C — Churn Risk Alert: "
        "An enterprise account is showing churn signals and requires immediate attention. "
        "Flag the at-risk account, assess their recent support ticket volume and open bug history, "
        "and assign an intervention owner. "
        "Use discovery operations to identify the account before taking action."
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
