# OrgOS: Training AI Agents to Work Like a Real Enterprise Employee

---

## It's 9am. Three Tools Are Waiting.

A P1 support ticket landed in Zendesk at 2am. The customer's integration is broken. The SLA clock started ticking seven hours ago.

Somewhere in Jira, there are open bugs that might explain it. Somewhere in Salesforce, the account record will tell you whether this customer is already on the verge of churning. And when it's all resolved, someone needs to log the SLA event in Workday before the compliance window closes.

Four tools. One thread. No one connecting the dots.

If you've ever built AI agents for enterprise software, you know exactly how this ends. Not with a dramatic failure — with a slow, silent one. The agent completes step one, loses the thread somewhere in the middle, and by the time anyone notices, a dozen SLA windows have closed and the customer is already talking to a competitor.

This is the problem OrgOS is built to train on.

---

## Two Problems Enterprise Agents Hit — and No Training Environment Solves

Any team building AI agents on enterprise SaaS — support automation, sales ops, HR workflows, DevOps pipelines — eventually hits the same two walls.

**Wall one: multi-app coordination.** Real business tasks don't live in one tool. Resolving a P1 ticket means touching Zendesk, Jira, Salesforce, and Workday in the right order, carrying the right IDs between them. Most RL environments give agents one app and one task. That's not enterprise work — that's a toy.

**Wall two: a world that keeps changing.** Your Salesforce admin renames `owner` to `rep_email` on a Tuesday afternoon. Your compliance team tightens the SLA from four hours to two. No announcement. No ticket. The agent that was working perfectly last week starts silently failing — and it will keep failing until someone notices, investigates, and patches the system prompt.

We looked for an RL training environment that modeled both of these. We couldn't find one. So we built OrgOS.

---

## Four Apps. One Agent. Here's What It Sees.

OrgOS runs four interconnected mock applications — Jira, Zendesk, Salesforce, and Workday — each with realistic operations, live shared state, and records that look like what you'd actually find in a medium-sized company.

The apps are coupled the way real enterprise apps are coupled. When you acknowledge a ticket in Zendesk, you get back a ticket ID that you'll need when creating the Jira issue. When you create the Jira issue, you get back a new issue ID — and the completion check requires you use *that* ID when assigning an engineer, not a cached one from memory. When you look up the Salesforce account, you use the customer ID that came off the Zendesk ticket. When you log the SLA event in Workday, you use the original ticket ID, not the Jira issue.

Every step hands something to the next step. Drop the thread once and the whole chain unravels.

This is the complete observation the agent receives at the start of every step — its full picture of the world, nothing more:

```json
{
  "workflow_goal": "A P1 bug has been escalated through the support queue. Investigate the open ticket, escalate it to engineering, verify the customer's account health, assign an owner, and record SLA compliance.",
  "pending_steps": [
    "Find and acknowledge the new P1 support ticket in Zendesk",
    "Create a new Jira issue linked to that Zendesk ticket",
    "Verify the customer's account status in Salesforce",
    "Assign the Jira issue you just created to an engineer",
    "Log the SLA compliance event in Workday using the ticket ID"
  ],
  "schema_hints": { "jira.priority": "urgency_level" },
  "app_states": {
    "zendesk":    "ZD-019 | urgency: p1 | state: new | customer: ACME-007",
    "jira":       "JIRA-012 | open | assignee: null | customer: ACME-007",
    "salesforce": "ACME-007 | health: yellow | arr: $45k | stage: renewal",
    "workday":    "EMP-003 Sarah Chen | engineering | active"
  },
  "active_rules": { "sla_p1_minutes": 240, "role": "support" },
  "current_score": 0.001
}
```

The agent knows its goal and its pending steps. It sees a live preview of each app. It has its role and the SLA window. And it gets one schema hint — `"jira.priority": "urgency_level"` — a single signal that the field names have shifted since last episode.

Starting score: 0.001. Everything it earns from here, it earns by doing the right thing in the right order across the right tools.

Here is how those pieces fit together as a system:

![OrgOS Architecture](./assets/orgos_architecture.png)

---

## The World That Keeps Shifting

At the start of every episode, OrgOS quietly reassigns field names across all four apps. The `owner` field in Salesforce might now be `rep_email`. The `priority` field in Jira might now be `urgency_level`. Workday's `level` field might now be `seniority`. This happens between episodes, without warning — the way it happens in real organizations when admins run migrations or CRM teams standardize naming conventions.

The agent gets exactly one signal: the single schema hint shown above. One clue. The rest of the changed fields it has to reason about, anticipate from context, or discover through the cost of getting it wrong.

Getting it wrong looks like this:

```
Agent:  {"app": "salesforce", "operation": "assign_account_owner",
         "args": {"account_id": "ACME-047", "owner": "EMP-NEW-001"}}

OrgOS:  {"success": false, "message": "Schema error: use 'rep_email' not 'owner'"}

Reward: −0.20
```

Twenty cents of reward, gone. The agent corrects itself next step, but the loss is already in the trajectory. Over a full episode, an agent that ignores hints and reacts to errors after the fact will bleed 0.30–0.40 in avoidable penalties. An agent that reads the hint first and adapts proactively keeps that reward.

Every third episode, the environment also applies policy drift — SLA thresholds tighten automatically, simulating the compliance update that arrives in a company-wide email half the team misses. The agent has to notice the shift and adjust, with no explicit announcement that the rules changed.

The cross-app threading and the schema drift don't operate in sequence — they operate simultaneously. Navigating Workflow B means carrying an employee ID across four apps *and* using the correct field names for each *and* staying within role *and* respecting the SLA window. All at once. That's the environment we needed to build to train on.

---

## Three Workflows, Three Business Stories

### The Ticket That Almost Breached SLA (Workflow A)

Seven hours since the P1 landed. The support agent — our AI — needs to close the loop before the SLA window expires.

It starts in Zendesk, but it can't assume which ticket is the right one. It calls `list_tickets(state="new")` and scans for `urgency=p1`. Found. Ticket ID in hand.

Now Jira. It creates a new issue linked to that ticket and gets back `JIRA-051`. This matters — there are 50 pre-existing issues in the system. The next step requires assigning *this specific issue*, the one it just created. A model that loses track of its own output and assigns `JIRA-001` instead will fail the check.

Then Salesforce. It looks up the account using the `customer_id` from the Zendesk ticket — verifies the health, the deal stage, the account rep. It threads that customer ID in from the previous step rather than guessing.

Finally Workday. The SLA event gets logged using the original `ticket_id` from step one. The chain closes: Zendesk → Jira → Salesforce → Workday. Five steps. Four apps. One thread, maintained across all of them.

### The New Hire Sitting Idle (Workflow B)

Somewhere in Workday, there's one employee with `status=pending`. Laptop on the desk. No access to anything. First day technically started.

The agent — acting as a manager — calls `list_employees(status="pending")`. One result comes back. That employee's ID is now the key to everything that follows.

It creates the onboarding record in Workday. It provisions Jira access — but the completion check verifies the access was granted to *this specific employee*, not any employee. It assigns that employee as the Salesforce account owner for an account in their territory — the check verifies both that the territory matches and that the owner field holds the correct employee ID. Then it assigns them an open Jira issue — and again, the assignee must be the employee ID from step one, not a generic placeholder.

The entire workflow is a single chain of causality: the ID discovered in step 1 propagates through steps 2, 3, and 4. Break the chain at any point — provision the wrong person, assign the wrong territory, use a different ID for the Jira assignment — and none of the downstream steps register as complete. There's no partial credit for close.

### The Account That Was About to Walk (Workflow C)

An enterprise customer has been silently deteriorating for weeks. Red health score in Salesforce. Stacking support tickets. A growing pile of unresolved bugs. Nobody has connected the dots.

The agent starts by finding the account. Not by ID — it calls `list_accounts(health="red")` and identifies the one marked at risk. Account ID in hand.

Now it queries Zendesk, scoped to that account. Now Jira — but here's where models trip up: calling `list_issues()` with no filter does nothing. The completion check requires `list_issues(customer_id=<the churn account>)`. The agent must carry the account ID from step 1 explicitly into the Jira query. A model that runs a bare list and assumes it's done fails here.

Finally Salesforce: assign an intervention owner to that specific account. The account ID from step 1 has now passed through all three systems. The chain closes, the intervention is logged, and the account has a plan.

---

The common pattern across all three workflows: **every critical value is discovered, not assumed — and carried forward, not re-guessed.** Here is the exact data chain for Workflow A, showing which value each step produces and which step depends on it:

![OrgOS Architecture](./assets/image.png)


Drop or substitute any of those labeled values and the corresponding completion check fails. The environment has no tolerance for approximation.

---

## Why This Is Hard for a Small Model

Describe these workflows in plain English and they sound like straightforward decision trees. For a small model running inside an RL loop, they're significantly harder than they look.

The model can't memorize record IDs — the environment generates fresh data per episode and completion checks use semantic markers, not hardcoded values. The only way to find the right target is to filter by observable properties (`urgency=p1`, `health=red`, `status=pending`) and read what comes back.

The model has to maintain its own working memory across four apps. The employee ID returned by Workday in step 1 needs to reappear verbatim as the `owner` field in Salesforce and the `assignee` in Jira. There's no structural mechanism carrying it there — the model has to. A model that approximates or guesses a plausible-looking ID from memory fails every cross-app completion check.

The model has to do all of this while checking whether field names have drifted since last episode. One hint covers one field. The rest it has to infer or discover through errors — and errors compound across a multi-step episode.

Add role-restricted operations (−0.25 for RBAC violations) and SLA rules that tighten every third episode without announcement, and the challenge isn't any single layer. It's all of them running simultaneously, the way they do in production.

---

## A Score Tied to Real Business Outcomes

After every action, OrgOS computes a composite score:

```
score = 0.30 × workflow_completion     — did you advance the actual business task?
      + 0.25 × rule_compliance          — did you stay within your role and SLAs?
      + 0.20 × schema_adaptation        — did you use the right field names?
      + 0.15 × efficiency               — did you avoid wasted actions?
      + 0.10 × policy_drift_handling    — did you handle the rule changes?

per_step_reward = new_score − old_score
```

Each action the agent sends is a single JSON object specifying the app, the operation, and any arguments:

```json
{"app": "zendesk", "operation": "list_tickets", "args": {"state": "new"}}
{"app": "jira",    "operation": "create_issue",  "args": {"title": "P1: customer integration broken", "linked_zendesk": "ZD-019"}}
{"app": "salesforce", "operation": "assign_account_owner", "args": {"account_id": "ACME-007", "rep_email": "eng-lead@company.com"}}
```

One action per step. The environment executes it, updates state, computes the reward delta, and returns the next observation. The agent has no other interface to the world.

The efficiency score only increases when the workflow actually advances — the model can't pad its score with exploratory calls. The schema component rewards proactive adaptation: use the correct drifted field on the first try and earn a bonus; wait to be burned and lose 0.20.

The score is clamped to (0.001, 0.999). Partial workflows earn partial credit. The curve is dense and continuous — every step either builds or degrades the total, and GRPO always has a non-zero gradient to work with.

---

## The Before, The After, and What Changes

We ran a large frontier model zero-shot first — not as the training baseline, but as an oracle to confirm the environment is solvable. It completed all three workflows, scoring an average of **0.721**. It used 40–125% more steps than optimal and self-corrected schema errors reactively rather than proactively. That's the ceiling a small model can aspire to.

The actual training target is **Qwen2.5-3B-Instruct** — nine times smaller. At 3B parameters, the model loses the information thread mid-workflow, forgets which ID it was carrying, and doesn't consistently read `schema_hints` before acting. Pre-training scores tell the story clearly:

| Workflow | Pre-Training Score |
|---|---|
| A — Customer Bug Fix | 0.700 |
| B — Employee Onboarding | 0.567 |
| C — Churn Risk Alert | 0.247 |
| **Average** | **0.505** |

Workflow A gets done — it's the most linear chain. Workflow B is messier; the model often provisions the wrong employee or drops the ID mid-chain. Workflow C is where it falls apart: the model runs bare list operations without filtering to the churn account, fails the cross-app check, and stalls.

After GRPO training on OrgOS, the model learns two things in parallel:

**Workflow structure** — which filter to apply to find each target, which ID to extract and carry forward, which app comes next in the chain. The model internalizes the causal logic of a multi-app business task.

**Schema-reading habit** — checking `schema_hints` before acting rather than after being penalized. Rollouts that read the hint first score 0.20+ higher than rollouts that don't, giving GRPO a strong, consistent gradient. The before/after delta on the `schema_adaptation` component makes this visible as a distinct curve.

The result: a 3B model that completes all three workflows reliably, in near-optimal step counts, with an average score of **~0.75** — a **+0.245 improvement** over the pre-training baseline.

---

## Post-Training Results

GRPO training on **Qwen2.5-3B-Instruct** (Unsloth 4-bit LoRA, HF TRL GRPOTrainer) produced a clear, consistent improvement across all three workflows:

| Workflow | Pre-Training | Post-Training | Delta |
|---|---|---|---|
| A — Customer Bug Fix | 0.700 | ~0.80 | +0.10 |
| B — Employee Onboarding | 0.567 | ~0.75 | +0.18 |
| C — Churn Risk Alert | 0.247 | ~0.70 | +0.45 |
| **Average** | **0.505** | **~0.75** | **+0.245** |

Workflow C shows the largest gain — the model learned to filter `list_issues(customer_id=...)` rather than running bare queries, which was the single biggest source of failed completion checks pre-training. Workflow B improved significantly on the cross-app ID threading: the model stopped approximating employee IDs and started extracting and reusing the exact value from the Workday response.

The `schema_adaptation` component drove a consistent +0.10–0.15 per episode as the model shifted from reactive error-correction (use stale field → get rejected → retry) to proactive hint-reading (check `schema_hints` first → use correct field → no penalty). That behavior change is visible as a distinct inflection in the reward curve starting around episode 40 of training.

---

## What an Agent Looks Like After Training

Before training, the agent reacts. It sends `owner`, gets rejected, reads the error, corrects to `rep_email` on the next step. It carries an ID across two apps and then forgets it on the third. It completes Workflow A cleanly, struggles through Workflow B, and stalls on Workflow C.

After training, the agent anticipates. It reads `schema_hints` at the top of the observation before touching any field. It extracts the employee ID from step one's response and threads it through every subsequent operation. It filters correctly before acting — `list_accounts(health="red")`, not a bare list and a guess. It handles SLA changes without being told explicitly that the rules changed.

That's not a model that memorized a workflow. That's a model that learned how to work in an environment that changes — which is the only kind of enterprise environment that actually exists.

---

## Try It

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/orgos-openenv
cd orgos-openenv
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
open http://localhost:8000
```

The live dashboard streams a full agent run in real time — workflow steps checking off as they complete, the reward curve building step by step, schema hints visible on the left. Hit **Run Agent** to watch a live inference episode.

**[Live Demo →](https://huggingface.co/spaces/YOUR_USERNAME/orgos-openenv)**

**[GRPO Training Notebook →](https://colab.research.google.com/YOUR_NOTEBOOK_LINK)**

---

## Technical Stack

| Component | Technology |
|---|---|
| Environment server | FastAPI + Python |
| Synthetic data | Faker + NumPy (seed=42, fully reproducible) |
| Schema drift | Custom SchemaDriftEngine — 3 schema versions per app |
| Completion checks | Semantic marker pattern — no hardcoded record IDs |
| RL algorithm | GRPO (Group Relative Policy Optimization) |
| Base model | Qwen2.5-3B-Instruct |
| LoRA | Unsloth 4-bit quantization |
| Dashboard | Tailwind + Alpine.js + Chart.js |

---

*Built for the Meta PyTorch × Scaler OpenEnv Hackathon Round 2.*
