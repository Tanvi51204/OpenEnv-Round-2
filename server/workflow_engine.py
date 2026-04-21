@dataclass
class WorkflowStep:
    step_id: str
    description: str
    app: str
    operation: str
    # Callable that checks if this step was completed given the app states
    completion_check: Callable[[Dict[str, "BaseApp"]], bool]

# Workflow A: Customer Bug → Engineering Fix
WORKFLOW_A_STEPS = [
    WorkflowStep("A1", "Acknowledge ticket in Zendesk",
                 "zendesk", "acknowledge_ticket",
                 lambda apps: apps["zendesk"].ticket_acknowledged()),

    WorkflowStep("A2", "Escalate to Jira — create linked issue",
                 "jira", "create_issue",
                 lambda apps: apps["jira"].has_linked_issue()),

    WorkflowStep("A3", "Check if customer is paying (Salesforce lookup)",
                 "salesforce", "get_account",
                 lambda apps: apps["salesforce"].account_checked()),

    WorkflowStep("A4", "Assign correct engineer in Jira based on priority",
                 "jira", "assign_owner",
                 lambda apps: apps["jira"].issue_assigned()),

    WorkflowStep("A5", "Log SLA status in Workday",
                 "workday", "log_sla_event",
                 lambda apps: apps["workday"].sla_logged()),
]

# Workflow B: Employee Onboarding
WORKFLOW_B_STEPS = [
    WorkflowStep("B1", "Create employee record in Workday", ...),
    WorkflowStep("B2", "Provision Jira access based on role", ...),
    WorkflowStep("B3", "Add to Salesforce team by territory", ...),
    WorkflowStep("B4", "Create Zendesk support profile if customer-facing", ...),
]

# Workflow C: Churn Risk Alert
WORKFLOW_C_STEPS = [
    WorkflowStep("C1", "Flag at-risk account in Salesforce", ...),
    WorkflowStep("C2", "Query recent support volume in Zendesk", ...),
    WorkflowStep("C3", "Check outstanding bugs in Jira", ...),
    WorkflowStep("C4", "Synthesize churn score and assign intervention owner", ...),
]

class WorkflowEngine:
    WORKFLOWS = {"A": WORKFLOW_A_STEPS, "B": WORKFLOW_B_STEPS, "C": WORKFLOW_C_STEPS}

    def start(self, workflow_id: str) -> None:
        self._steps = self.WORKFLOWS[workflow_id].copy()
        self._completed: List[str] = []

    def evaluate(self, apps: Dict) -> float:
        """Check all steps and return completion ratio (0.0-1.0)."""
        completed = sum(1 for s in self._steps if s.completion_check(apps))
        self._completed = [s.step_id for s in self._steps if s.completion_check(apps)]
        return completed / len(self._steps)

    def get_pending(self) -> List[str]:
        return [s.description for s in self._steps if s.step_id not in self._completed]