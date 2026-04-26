
# OrgOS — Enterprise Workflow RL Environment

**OrgOS** is a multi-app enterprise reinforcement learning environment where an AI agent completes real business workflows across four interconnected SaaS applications. Between episodes the environment injects **schema drift** (renamed API fields) and **policy changes** (tightened SLAs and approval thresholds), forcing agents to generalize rather than memorize.

Built for the [Meta PyTorch × Scaler OpenEnv Hackathon](https://huggingface.co/) — targeting the **Multi-App Enterprise Workflow** sub-theme.

---

## Resources

| | |
|---|---|
| Environment Space | **[huggingface.co/spaces/srishtichugh/orgOS](https://huggingface.co/spaces/srishtichugh/orgOS)** |
| Training Space | **[huggingface.co/spaces/muskansingh1101/orgos-training](https://huggingface.co/spaces/muskansingh1101/orgos-training)** |
| HF Blog Post | **[OrgOS: Teaching Agents to Survive Enterprise API Drift](https://huggingface.co/spaces/srishtichugh/orgOS/blob/main/hf_blog_post.md)** |
| Training Notebook | **[training/grpo_orgos.ipynb](https://colab.research.google.com/drive/1BekDqh64FU0kskSnaNrFU8HEkcdexPC3?usp=sharing)** |
| Training Logs | **[training/grpo_orgos.ipynb](https://github.com/Tanvi51204/OpenEnv-Round-2/tree/main/training/orgos-training)** |
| Youtube Demo Video| **[]()** |


---

## Live Demo

🚀 **[HuggingFace Space →](https://huggingface.co/spaces/srishtichugh/orgOS)**

```bash
# Local quickstart
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000 for the live dashboard
```

---

## The Problem OrgOS Solves

Real enterprise AI agents don't fail because the model is bad — they fail because the environment keeps changing. SaaS APIs rename fields across versions. SLA policies tighten after incidents. Access controls shift when team structures change.

An agent trained on a static dataset will memorize field names like `priority`, `assignee`, `deal_stage`. But in production, those same fields become `severity`, `owner`, and `pipeline_stage`. The agent breaks silently — it still runs, but its actions fail schema validation and real work never gets done.

OrgOS simulates this exactly. Every episode, the agent faces the same four apps with **different field names** and potentially **different SLA rules**. The only path to a high score is reading the `schema_hints` observation and the `active_rules` dict before acting — then using the *current* field names, not the ones it saw in training.

---

## What Makes OrgOS Unique

| Feature | Description |
|---|---|
| **4 Mock SaaS Apps** | Jira, Zendesk, Salesforce, Workday — each with realistic CRUD operations |
| **Schema Drift** | Fields rename between episodes across 3 versioned schemas per app. Agent gets `-0.20` for stale names, `+0.10` for adapted names |
| **Policy Drift** | Every 3rd episode, SLA thresholds tighten automatically (P0: 30 min → 15 min, P1: 4 h → 2 h) |
| **3 Workflows** | Cross-app tasks that require correct sequencing and state carry-over between steps |
| **RBAC** | Support vs. manager roles strictly enforced; `-0.25` penalty for unauthorized actions |
| **Dense Reward** | Per-step composite signal tied to 5 measurable business outcomes |

---

## The Three Workflows

Each workflow tests a different capability: information discovery, state threading between apps, and schema-aware field usage. All three run against the same four apps but require different operation sequences and roles.

---

### Workflow A — Customer Bug Fix
**Role:** `support` | **Steps:** 5 | **Step budget:** 15

A P1 bug has been escalated through the support queue. The agent must move it end-to-end: from acknowledging the ticket in Zendesk, through creating and assigning a Jira issue, to verifying account health in Salesforce and logging SLA compliance in Workday.

| Step | App | Operation | What Must Happen |
|---|---|---|---|
| A1 | Zendesk | `acknowledge_ticket` | Find and acknowledge the new P1 ticket |
| A2 | Jira | `create_issue` | Create a new issue **linked** to that Zendesk ticket |
| A3 | Salesforce | `get_account` | Verify the customer's account status |
| A4 | Jira | `assign_owner` | Assign the **same** issue created in A2 to an engineer |
| A5 | Workday | `log_sla_event` | Log SLA compliance using the ticket ID |

**Why it's hard:** Steps must happen in order — the Jira issue created in A2 must be the one assigned in A4. The agent can't shortcut by assigning an unrelated issue. Schema drift hits Zendesk's `urgency`/`state` fields and Jira's `priority`/`assignee` fields.

**What an untrained agent does:** Calls `list_tickets` in a loop, uses canonical field name `priority` (stale on v2/v3 schemas), gets `-0.20` schema error, never advances past A1.

**What a trained agent does:** Reads `schema_hints` first (e.g. `jira.priority → severity`), calls `acknowledge_ticket` with correct args, threads the ticket ID through to Jira's `create_issue`, then `assign_owner` on the same issue ID.

---

### Workflow B — Employee Onboarding
**Role:** `manager` | **Steps:** 4 | **Step budget:** 20

A new hire is in Workday with `status=pending`. The manager-role agent must complete their onboarding across all four apps — and each step's output feeds the next. The agent must carry `employee_id` and `territory` from step B1 through to B3 and B4.

| Step | App | Operation | What Must Happen |
|---|---|---|---|
| B1 | Workday | `create_onboarding_task` | Find the pending employee and create their onboarding record |
| B2 | Workday | `provision_access` | Provision **Jira** access for that specific `employee_id` |
| B3 | Salesforce | `assign_account_owner` | Assign the new hire as owner of an account **in their own territory** |
| B4 | Jira | `assign_owner` | Assign an open Jira issue to the new hire's `employee_id` |

**Why it's hard:** This is a state-threading problem. There's only one pending employee in Workday, but the agent must discover them via `list_employees` with `status=pending`, extract their `employee_id` and `territory`, then pass those values correctly into B3 (Salesforce territory filter) and B4 (Jira assignee). Hardcoding any ID will fail — the data generator seeds differently each episode.

**RBAC note:** Only `manager` role has full access to all four apps. A `support` agent attempting Workday's `provision_access` or Salesforce's `assign_account_owner` incurs a `-0.25` penalty per violation.

**What an untrained agent does:** Tries to call `create_onboarding_task` directly without listing first, passes wrong `employee_id`, then attempts `assign_account_owner` on a random account (wrong territory → step B3 fails completion check).

**What a trained agent does:** Calls `list_employees` with `status=pending` → extracts `employee_id` + `territory` → threads them correctly into all downstream steps.

---

### Workflow C — Churn Risk Alert
**Role:** `support` | **Steps:** 4 | **Step budget:** 18

An enterprise account is showing churn signals. The agent must identify it in Salesforce, assess the account's support history and open bugs, then assign an intervention owner. The challenge: the at-risk account's ID changes each episode and must be discovered dynamically.

| Step | App | Operation | What Must Happen |
|---|---|---|---|
| C1 | Salesforce | `flag_churn_risk` | Identify and flag the at-risk account |
| C2 | Zendesk | `get_ticket` | Query support tickets **for the churn account's ID** (from C1) |
| C3 | Jira | `list_issues` | List open bugs **with `customer_id=<churn account>`** (from C1) |
| C4 | Salesforce | `assign_account_owner` | Assign an intervention owner to the at-risk account |

**Why it's hard:** Steps C2 and C3 require passing the `account_id` discovered in C1 as a filter argument. A hardcoded ID (or no filter) fails the completion check. Salesforce's `health`/`owner`/`deal_stage` fields drift across schema versions, so the agent must use the current names when calling `flag_churn_risk`.

**Why it scores lowest for untrained agents (0.25):** Schema drift hits Salesforce hardest — three fields rename simultaneously between v1/v2/v3. The untrained model almost never uses the right field names on the first call, burning half its step budget on schema errors before discovering C1's output is needed in C2 and C3.

**What a trained agent does:** Reads `schema_hints` to find current Salesforce field names → calls `flag_churn_risk` correctly → extracts the returned `account_id` → uses it as the filter in both the Zendesk query and Jira `list_issues` call.

---

## Schema Drift — Deep Dive

Each episode, the schema drift engine samples an independent schema version (v1/v2/v3) for each app. v1 uses canonical field names (no drift). v2 and v3 rename fields.

| App | Canonical | v2 | v3 |
|---|---|---|---|
| **Jira** | `priority` | `severity` | `urgency_level` |
| **Jira** | `assignee` | `owner` | `assigned_to` |
| **Jira** | `status` | `state` | `current_state` |
| **Zendesk** | `urgency` | `priority` | `impact_level` |
| **Zendesk** | `agent_email` | `handler` | `assigned_agent` |
| **Salesforce** | `deal_stage` | `pipeline_stage` | `stage` |
| **Salesforce** | `health` | `account_health` | `risk_score` |
| **Salesforce** | `owner` | `account_owner` | `rep_email` |
| **Workday** | `level` | `job_level` | `seniority` |
| **Workday** | `manager_id` | `reports_to` | `direct_manager` |

The observation includes **one schema hint** per episode (e.g. `{"jira.priority": "severity"}`). The agent must use `get_*` and `list_*` operations to discover the rest of the drift by reading what the app returns.

**Reward signals:**
- Using a stale canonical field name: **-0.20**
- Using the correct drifted field name: **+0.10**
- v1 schema (no drift): no penalty, no credit

---

## Policy Drift

Every 3rd episode, SLAs tighten:

| Rule | Default | After Policy Drift |
|---|---|---|
| P0 acknowledgement | 30 min | **15 min** |
| P1 first response | 4 hours | **2 hours** |
| Budget approval threshold | $10,000 | **$5,000** |

Since each environment step simulates ~10 minutes of elapsed time, under policy drift a P0 ticket must be acknowledged within the first **step** — not the first few steps. The agent sees the current thresholds in `active_rules` on every observation.

---

## Reward Function

```
score = 0.30 × workflow_completion
      + 0.25 × rule_compliance
      + 0.20 × schema_adaptation
      + 0.15 × efficiency
      + 0.10 × policy_drift_handling

Per-step delta = new_score − old_score
Schema error penalty     = −0.20  (stale field name used)
RBAC violation penalty   = −0.25  (unauthorized operation)
SLA breach penalty       = −0.10 to −0.15
Terminal completion bonus = +0.20  (all workflow steps done)
```

`efficiency` only increases when a **new workflow step is completed** — padding with repeated `list_*` calls doesn't help. This is what makes single-step reward exploitation hard and why the multi-step reward function in training is critical.

---

## Applications & Operations

| App | Key Operations |
|---|---|
| **Jira** | `get_issue`, `create_issue`, `update_status`, `set_priority`, `assign_owner`, `link_zendesk_ticket`, `close_issue`, `list_issues` |
| **Zendesk** | `get_ticket`, `acknowledge_ticket`, `set_urgency`, `assign_agent`, `escalate_to_jira`, `resolve_ticket`, `add_note`, `list_tickets` |
| **Salesforce** | `get_account`, `list_accounts`, `update_deal_stage`, `flag_churn_risk`, `assign_account_owner`, `log_interaction`, `get_opportunity` |
| **Workday** | `get_employee`, `list_employees`, `provision_access`, `log_sla_event`, `request_budget_approval`, `create_onboarding_task`, `complete_task` |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start new episode (`{"workflow_id": "A"\|"B"\|"C"}`) |
| `POST` | `/step` | Take action (`{"app": ..., "operation": ..., "args": {...}}`) |
| `GET` | `/state` | Current episode metadata |
| `GET` | `/schema/apps` | All app operations catalogue |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/` | Live dashboard (UI) |

---

## Training

The [`training/grpo_orgos.ipynb`](training/grpo_orgos.ipynb) notebook trains **Qwen2.5-3B-Instruct** with **Unsloth 4-bit LoRA** using **HF TRL GRPOTrainer** (150 GRPO steps, multi-step reward, checkpoints every 30 steps).

Also runnable as a live HF Space: **[muskansingh1101/orgos-training](https://huggingface.co/spaces/muskansingh1101/orgos-training)**

### What Training Teaches the Agent

The key behavioral changes GRPO induces:

1. **Schema awareness** — the agent learns to read `schema_hints` in the observation before constructing action args. Before training it ignores hints and uses canonical names; after training it uses the drifted names.

2. **Step sequencing** — the agent learns to follow `pending_steps` in order. Before training it calls operations randomly; after training it completes A1 before attempting A2, and carries state (IDs) from earlier steps forward.

3. **Discovery before action** — the agent learns to call `list_*` or `get_*` first to discover record IDs, rather than guessing or hardcoding them.

### Results

| Workflow | Before GRPO | After GRPO | Δ |
|---|---|---|---|
| A — Customer Bug Fix | 0.70 | ~0.82 | +0.12 |
| B — Employee Onboarding | 0.57 | ~0.74 | +0.17 |
| C — Churn Risk Alert | 0.25 | ~0.48 | +0.23 |
| **Average** | **0.50** | **~0.68** | **+0.18** |

The largest gain is on Workflow C — the most schema-sensitive workflow (Salesforce has 3 drifting fields simultaneously). The untrained model almost never makes it past C1; the trained model completes C1→C2→C3 reliably.

![Training Curve](training/plots/training_curve.png)
*Reward per training step — 150 GRPO steps on Qwen2.5-3B-Instruct*

![Baseline vs Trained](training/plots/baseline_vs_trained.png)
*Per-workflow score: untrained baseline vs. GRPO-trained agent*

![Score Distribution](training/plots/score_distribution.png)
*Distribution of episode scores before and after training*

---

## Action / Observation Format

**Action:**
```json
{"app": "zendesk", "operation": "acknowledge_ticket", "args": {"ticket_number": "ZD-001"}}
```

**Observation (key fields):**
```json
{
  "workflow_goal": "Workflow A — Customer Bug Fix: A P1 bug has been escalated...",
  "pending_steps": ["Assign the Jira issue you just created to an engineer", "Log the SLA compliance event in Workday"],
  "completed_steps": ["A1", "A2", "A3"],
  "schema_hints": {"jira.priority": "severity"},
  "active_rules": {"sla_p0_minutes": 15, "sla_p1_hours": 2, "approval_threshold": 5000},
  "current_score": 0.42,
  "message": "Jira issue JI-001 created and linked to ZD-001"
}
```

---

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Run baseline inference (requires LLM API)
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
python inference.py

# 4. Or use the Python client
from client import OrgOSEnvClient
client = OrgOSEnvClient("http://localhost:8000")
result = client.reset(workflow_id="A")
print(result.observation.workflow_goal)
```

## Docker

```bash
docker build -t orgos-env .
docker run -p 8000:8000 orgos-env
```

---

## Project Structure

```
openEnv/
├── server/
│   ├── app.py              # FastAPI routes (15 endpoints)
│   ├── environment.py      # OrgOSEnvironment — reset/step/state
│   ├── schema_drift.py     # Per-episode field renames (3 versions per app)
│   ├── business_rules.py   # RBAC + SLA enforcement + policy drift
│   ├── workflow_engine.py  # 3 cross-app workflow definitions + completion checks
│   ├── data_generator.py   # Synthetic data (seed=42 + episode_num)
│   └── apps/
│       ├── jira.py
│       ├── zendesk.py
│       ├── salesforce.py
│       └── workday.py
├── models.py               # Pydantic models
├── client.py               # OrgOSEnvClient
├── inference.py            # Baseline inference loop
├── ui/index.html           # Live dashboard (Tailwind + Alpine.js + Chart.js)
├── training/
│   ├── grpo_orgos.ipynb   # GRPO training notebook (Colab-ready)
│   └── plots/             # Training result plots
├── openenv.yaml            # OpenEnv manifest
└── Dockerfile
```
---
title: Orgos Training
emoji: 🏆
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---

# OrgOS — GRPO Training Space (can be found deployed at HF : )

This Space trains **Qwen2.5-3B-Instruct** with **Unsloth 4-bit LoRA** using **HF TRL GRPOTrainer** on the OrgOS enterprise workflow RL environment.

---

## What Is OrgOS?

OrgOS is a multi-app enterprise RL environment where an AI agent completes real business workflows across four interconnected SaaS apps (Jira, Zendesk, Salesforce, Workday). Between episodes the environment injects **schema drift** (renamed fields) and **policy changes** (tightened SLAs), forcing agents to generalize rather than memorize.

🌐 **[Environment Space →](https://huggingface.co/spaces/muskansingh1101/orgos-training)**

---

## Training Setup

| Config | Value |
|---|---|
| **Model** | Qwen2.5-3B-Instruct (4-bit via Unsloth) |
| **Algorithm** | GRPO (Generalized Reward Policy Optimization) |
| **LoRA rank** | r=16 |
| **Steps** | 150 |
| **Batch size** | 1 (grad accum 2) |
| **Learning rate** | 8e-6 |
| **Reward** | Multi-step (2 steps per sample) |
| **Checkpoints** | Every 30 steps to Drive |

The reward function combines 5 components: workflow completion (0.30), rule compliance (0.25), schema adaptation (0.20), efficiency (0.15), policy drift handling (0.10).

---

## Results

| Workflow | Before GRPO | After GRPO | Δ |
|---|---|---|---|
| A — Customer Bug Fix | 0.70 | ~0.82 | +0.12 |
| B — Employee Onboarding | 0.57 | ~0.74 | +0.17 |
| C — Churn Risk Alert | 0.25 | ~0.48 | +0.23 |
| **Average** | **0.50** | **~0.68** | **+0.18** |

![Training Curve](plots/training_curve.png)
*Reward per training step*

![Baseline vs Trained](plots/baseline_vs_trained.png)
*Per-workflow score comparison*

![Score Distribution](plots/score_distribution.png)
*Episode score distribution before and after GRPO*

---

MIT License · Built for Meta PyTorch × Scaler OpenEnv Hackathon Round 2
