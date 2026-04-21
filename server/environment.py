"""OrgOS environment — the single stateful RL environment object."""

import uuid
from typing import Dict, Optional

from models import OrgOSAction, OrgOSObservation, OrgOSState, RewardBreakdown
from server.apps.jira import JiraApp
from server.apps.zendesk import ZendeskApp
from server.apps.salesforce import SalesforceApp
from server.apps.workday import WorkdayApp
from server.business_rules import BusinessRuleEngine
from server.data_generator import generate_episode_data
from server.schema_drift import SchemaDriftEngine
from server.workflow_engine import WorkflowEngine


class OrgOSEnvironment:
    MAX_STEPS = {"A": 15, "B": 20, "C": 18}
    WORKFLOWS  = ["A", "B", "C"]

    def __init__(self):
        self._drift    = SchemaDriftEngine(seed=42)
        self._rules    = BusinessRuleEngine()
        self._workflow = WorkflowEngine()
        self._apps: Dict[str, object] = {
            "jira":       JiraApp(self._drift),
            "zendesk":    ZendeskApp(self._drift),
            "salesforce": SalesforceApp(self._drift),
            "workday":    WorkdayApp(self._drift),
        }
        self._episode_num          = 0
        self._episode_id           = ""
        self._workflow_id          = "A"
        self._step_count           = 0
        self._last_score           = 0.001
        self._policy_drift_applied = False

        # Reward component trackers
        self._wf_score     = 0.0   # workflow completion
        self._rule_score   = 1.0   # compliance (starts perfect, penalized on violation)
        self._schema_score = 0.0   # schema adaptation successes
        self._efficiency   = 1.0   # degrades with failed/no-op actions
        self._policy_score = 0.0   # policy drift handling bonus

    # ------------------------------------------------------------------
    # OpenEnv core API
    # ------------------------------------------------------------------

    def reset(self, workflow_id: Optional[str] = None) -> OrgOSObservation:
        self._episode_num += 1
        self._episode_id   = str(uuid.uuid4())
        self._workflow_id  = workflow_id or self.WORKFLOWS[(self._episode_num - 1) % 3]
        self._step_count   = 0
        self._last_score   = 0.001
        self._rule_score   = 1.0
        self._wf_score     = 0.0
        self._schema_score = 0.0
        self._efficiency   = 1.0
        self._policy_score = 0.0
        self._policy_drift_applied = False

        # Sample schema versions for this episode
        self._drift.sample_for_episode(self._episode_num)

        # Possibly activate policy drift (every 3rd episode)
        self._rules = BusinessRuleEngine()
        if self._episode_num % 3 == 0:
            self._rules.apply_policy_drift("sla_tighten")
            self._policy_drift_applied = True

        # Load fresh synthetic data into each app
        records = generate_episode_data(self._workflow_id, seed=42 + self._episode_num)
        for app_name, app in self._apps.items():
            app.initialize(records[app_name])

        # Start workflow tracking
        self._workflow.start(self._workflow_id)

        return self._build_obs(
            reward=0.001,
            done=False,
            message="Episode started. Study the workflow goal and schema hints before acting.",
        )

    def step(self, action: OrgOSAction) -> OrgOSObservation:
        self._step_count += 1
        old_score    = self._last_score
        extra_penalty = 0.0

        # 1. Validate app exists
        if action.app not in self._apps:
            return self._build_obs(
                reward=-0.05,
                done=False,
                message=f"Unknown app '{action.app}'. Valid apps: {list(self._apps)}",
            )

        # 2. Business rule check (RBAC, approvals)
        agent_role = self._workflow.get_role()
        ctx        = {"agent_role": agent_role, "manager_approved": False}
        allowed, reason, rule_penalty = self._rules.check_action(action, ctx)
        if not allowed:
            self._rule_score = max(0.0, self._rule_score - 0.08)
            extra_penalty    = rule_penalty
            return self._build_obs(
                reward=extra_penalty,
                done=False,
                message=f"Rule violation: {reason}",
            )

        # 3. Execute on app
        result = self._apps[action.app].execute(action.operation, action.args)

        # 4. Check schema drift FIRST — apps return success:False when schema_error is set
        if result.get("schema_error"):
            self._efficiency -= 0.02
            return self._build_obs(
                reward=-0.20,
                done=False,
                message=(
                    f"Stale schema: field '{result['schema_error']}' is no longer valid. "
                    "Check schema_hints for the current field name. "
                    f"Hint: {result.get('message', '')}"
                ),
            )

        if not result.get("success"):
            self._efficiency -= 0.02   # penalize failed/no-op actions
            return self._build_obs(
                reward=-0.01,
                done=False,
                message=result.get("message", "Operation failed"),
            )

        # Schema adaptation bonus (agent used correct drifted field name)
        if result.get("schema_adapted"):
            self._schema_score = min(1.0, self._schema_score + 0.10)
            self._policy_score = min(1.0, self._policy_score + 0.05)

        # 5. Re-evaluate workflow completion
        self._wf_score = self._workflow.evaluate(self._apps)

        # 6. SLA check (only if a ticket was touched)
        sla_ok, sla_pen = self._rules.check_sla(
            result.get("ticket", {}),
            self._step_count * 2.5,   # approximate 2.5 min per step
        )
        if not sla_ok:
            extra_penalty   += sla_pen
            self._rule_score = max(0.0, self._rule_score - 0.05)

        # 7. Compute composite score
        new_score = self._compute_score()
        delta     = new_score - old_score + extra_penalty
        self._last_score = max(0.001, min(0.999, new_score))

        # 8. Terminal condition
        done = (
            self._wf_score >= 0.95
            or self._step_count >= self.MAX_STEPS[self._workflow_id]
        )
        if done and self._wf_score >= 0.95:
            delta += 0.20   # terminal completion bonus

        return self._build_obs(
            reward=delta,
            done=done,
            message=result.get("message", "OK"),
        )

    # ------------------------------------------------------------------
    # State endpoint
    # ------------------------------------------------------------------

    def state(self) -> OrgOSState:
        return OrgOSState(
            episode_id           = self._episode_id,
            workflow_id          = self._workflow_id,
            schema_versions      = self._drift._versions,
            step_count           = self._step_count,
            max_steps            = self.MAX_STEPS.get(self._workflow_id, 15),
            rule_violation_count = len(self._rules._violation_log),
            workflow_completion  = self._wf_score,
            rule_compliance_rate = self._rule_score,
            policy_drift_active  = self._policy_drift_applied,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self) -> float:
        raw = (
            0.30 * self._wf_score   +
            0.25 * self._rule_score +
            0.20 * self._schema_score +
            0.15 * self._efficiency +
            0.10 * self._policy_score
        )
        return max(0.001, min(0.999, raw))

    def _build_obs(self, reward: float, done: bool, message: str) -> OrgOSObservation:
        """Construct a fully-populated observation from current environment state."""
        # Per-app state previews
        app_states = {
            name: app.get_state_view(max_rows=3)
            for name, app in self._apps.items()
        }

        flat_hints = self._drift.get_hints()

        # # Schema hints (partial — agent must probe to discover full mapping)
        # schema_hints = self._drift.get_hints()
        # # Flatten to dot-notation: {"jira.priority": "severity", ...}
        # flat_hints: Dict[str, str] = {}
        # for app_name, field_map in schema_hints.items():
        #     for canonical, drifted in field_map.items():
        #         if canonical != drifted:
        #             flat_hints[f"{app_name}.{canonical}"] = drifted

        # Workflow progress
        completed_steps = self._workflow.get_completed()
        pending_steps   = self._workflow.get_pending()
        workflow_goal   = self._workflow.get_goal()

        # Reward breakdown snapshot
        breakdown = RewardBreakdown(
            workflow_completion   = self._wf_score,
            rule_compliance       = self._rule_score,
            schema_adaptation     = self._schema_score,
            efficiency            = self._efficiency,
            policy_drift_handling = self._policy_score,
        )

        return OrgOSObservation(
            done              = done,
            reward            = round(float(reward), 6),
            current_score     = round(float(self._last_score),4),
            workflow_id       = self._workflow_id,
            step_count        = self._step_count,
            app_states        = app_states,
            workflow_goal     = workflow_goal,
            completed_steps   = completed_steps,
            pending_steps     = pending_steps,
            schema_hints      = flat_hints,
            active_rules      = self._rules.get_active_rules_summary(),
            rule_violations   = self._rules.get_violations_this_step(),
            reward_breakdown  = breakdown,
            message           = message,
        )
