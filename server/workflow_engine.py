"""Workflow engine — defines and evaluates multi-app workflow completion."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class WorkflowStep:
    step_id: str
    description: str
    app: str
    operation: str
    # Callable that checks if this step was completed given the app states
    completion_check: Callable[[Dict], bool]


# ---------------------------------------------------------------------------
# Helpers — look up semantic-marker targets at evaluation time so completion
# checks are not coupled to specific record IDs in the data generator.
# ---------------------------------------------------------------------------

def _churn_target_account_id(apps: Dict) -> Optional[str]:
    """Return the SF account_id flagged as the churn target this episode."""
    for aid, rec in apps["salesforce"]._records.items():
        if rec.get("_is_churn_target"):
            return aid
    return None


def _new_hire_assigned_sf(apps: Dict) -> bool:
    """Workflow B step B3: the new hire (from Workday) must be the SF owner of an account
    in their own territory. Threads employee_id + territory across apps automatically."""
    new_hire = apps["workday"].get_new_hire()
    if not new_hire:
        return False
    return apps["salesforce"].new_hire_assigned_in_territory(
        new_hire.get("employee_id", ""),
        new_hire.get("territory", ""),
    )


def _new_hire_assigned_jira(apps: Dict) -> bool:
    """Workflow B step B4: an open Jira issue must be assigned to the new hire's employee_id."""
    new_hire = apps["workday"].get_new_hire()
    if not new_hire:
        return False
    return apps["jira"].new_hire_assigned_to_issue(new_hire.get("employee_id", ""))


def _support_queried_for_churn_target(apps: Dict) -> bool:
    """Workflow C step C2: Zendesk must have been queried for the churn target's account."""
    aid = _churn_target_account_id(apps)
    return bool(aid) and apps["zendesk"].support_queried(aid)


def _bugs_checked_for_churn_target(apps: Dict) -> bool:
    """Workflow C step C3: Jira list_issues must have been called with customer_id=<churn target>."""
    aid = _churn_target_account_id(apps)
    return bool(aid) and apps["jira"].bugs_checked_for(aid)


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
        "A4", "Assign the Jira issue you just created (linked to the Zendesk ticket) to an engineer",
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
        "B1",
        "Find the pending new hire in Workday (list_employees status=pending returns one result) "
        "and create their onboarding record",
        "workday", "create_onboarding_task",
        lambda apps: apps["workday"].employee_created(),
    ),
    WorkflowStep(
        "B2",
        "Provision Jira access for THAT new hire (use the employee_id from B1)",
        "workday", "provision_access",
        lambda apps: apps["workday"].access_provisioned("jira"),
    ),
    WorkflowStep(
        "B3",
        "Assign the new hire as Salesforce account owner for an account in their own territory "
        "(use employee_id and territory from B1)",
        "salesforce", "assign_account_owner",
        _new_hire_assigned_sf,
    ),
    WorkflowStep(
        "B4",
        "Assign an open Jira issue to the new hire (use employee_id from B1 as the assignee)",
        "jira", "assign_owner",
        _new_hire_assigned_jira,
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
        "C2", "Query recent support tickets for the at-risk account in Zendesk "
              "(use the account_id from C1, not a hardcoded value)",
        "zendesk", "get_ticket",
        _support_queried_for_churn_target,
    ),
    WorkflowStep(
        "C3", "List open Jira bugs for the at-risk account "
              "(call list_issues with customer_id=<churn account>)",
        "jira", "list_issues",
        _bugs_checked_for_churn_target,
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
        "Workflow B — Manager-Aware Onboarding: "
        "One employee in Workday is currently pending onboarding (status=pending). "
        "Find that pending employee, create their onboarding record, provision their Jira access, "
        "assign them as the Salesforce account owner for an account in THEIR OWN territory, "
        "and assign them ownership of an open Jira issue. "
        "Each step's output feeds the next — capture the employee_id and territory from step 1 "
        "and reuse them in subsequent steps."
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
