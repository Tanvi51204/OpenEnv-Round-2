class OrgOSEnvironment:
    MAX_STEPS = {"A": 15, "B": 20, "C": 18}
    WORKFLOWS = ["A", "B", "C"]

    def __init__(self):
        self._drift    = SchemaDriftEngine(seed=42)
        self._rules    = BusinessRuleEngine()
        self._workflow = WorkflowEngine()
        self._apps: Dict[str, BaseApp] = {
            "jira":        JiraApp(self._drift),
            "zendesk":     ZendeskApp(self._drift),
            "salesforce":  SalesforceApp(self._drift),
            "workday":     WorkdayApp(self._drift),
        }
        self._episode_num   = 0
        self._episode_id    = ""
        self._workflow_id   = "A"
        self._step_count    = 0
        self._last_score    = 0.001
        self._policy_drift_applied = False

        # Reward component trackers
        self._wf_score      = 0.0   # workflow completion
        self._rule_score    = 1.0   # compliance (starts perfect, penalized on violation)
        self._schema_score  = 0.0   # schema adaptation successes
        self._efficiency    = 1.0   # degrades with no-ops
        self._policy_score  = 0.0   # policy drift handling

    def reset(self, workflow_id: Optional[str] = None) -> OrgOSObservation:
        self._episode_num += 1
        self._episode_id = str(uuid.uuid4())
        self._workflow_id = workflow_id or self.WORKFLOWS[(self._episode_num - 1) % 3]
        self._step_count  = 0
        self._last_score  = 0.001
        self._rule_score  = 1.0
        self._wf_score    = 0.0
        self._schema_score = 0.0
        self._efficiency  = 1.0
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

        return self._build_obs(0.001, False, "Episode started. Study the workflow goal and schema hints.")

    def step(self, action: OrgOSAction) -> OrgOSObservation:
        self._step_count += 1
        old_score = self._last_score
        extra_penalty = 0.0

        # 1. Validate app exists
        if action.app not in self._apps:
            return self._build_obs(old_score - 0.05, False, f"Unknown app '{action.app}'")

        # 2. Business rule check (RBAC, approvals)
        ctx = {"agent_role": "support", "manager_approved": False}
        allowed, reason, rule_penalty = self._rules.check_action(action, ctx)
        if not allowed:
            self._rule_score = max(0.0, self._rule_score - 0.08)
            extra_penalty = rule_penalty
            return self._build_obs(
                max(-0.25, old_score + extra_penalty),
                False, f"Rule violation: {reason}"
            )

        # 3. Execute on app
        result = self._apps[action.app].execute(action.operation, action.args)
        if not result["success"]:
            self._efficiency -= 0.02  # penalize failed/no-op actions
            return self._build_obs(old_score - 0.01, False, result["message"])

        # 4. Check schema drift adaptation
        # If agent used canonical field names on a v2/v3 schema → penalize
        if result.get("schema_error"):
            extra_penalty -= 0.20
            return self._build_obs(old_score - 0.20, False,
                f"Stale schema: field '{result['schema_error']}' not found in current schema")
        elif result.get("schema_adapted"):
            # Agent correctly used drifted field name → bonus
            self._schema_score = min(1.0, self._schema_score + 0.1)

        # 5. Re-evaluate workflow completion
        self._wf_score = self._workflow.evaluate(self._apps)

        # 6. Check SLA violations
        sla_ok, sla_pen = self._rules.check_sla(result.get("ticket", {}),
                                                  self._step_count * 2.5)  # 2.5 min per step
        if not sla_ok:
            extra_penalty += sla_pen
            self._rule_score = max(0.0, self._rule_score - 0.05)

        # 7. Compute composite score
        new_score = self._compute_score()
        delta = new_score - old_score + extra_penalty
        self._last_score = max(0.001, min(0.999, new_score))

        # 8. Terminal condition
        done = (self._wf_score >= 0.95 or
                self._step_count >= self.MAX_STEPS[self._workflow_id])
        if done and self._wf_score >= 0.95:
            delta += 0.20  # terminal bonus

        return self._build_obs(delta, done, result["message"])

    def _compute_score(self) -> float:
        raw = (
            0.30 * self._wf_score +
            0.25 * self._rule_score +
            0.20 * self._schema_score +
            0.15 * self._efficiency +
            0.10 * self._policy_score
        )
        return max(0.001, min(0.999, raw))

    def state(self) -> OrgOSState:
        return OrgOSState(
            episode_id=self._episode_id,
            workflow_id=self._workflow_id,
            schema_versions=self._drift._versions,
            step_count=self._step_count,
            max_steps=self.MAX_STEPS.get(self._workflow_id, 15),
            rule_violation_count=len(self._rules._violation_log),
            workflow_completion=self._wf_score,
            rule_compliance_rate=self._rule_score,
            policy_drift_active=self._policy_drift_applied,
        )