---
title: OrgOS Enterprise Workflow RL Environment
emoji: 🏢
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - rl
  - enterprise
  - multi-app
---

# OrgOS — Enterprise Workflow RL Environment

**OrgOS** is a multi-app enterprise reinforcement learning environment where an AI agent completes real business workflows across four interconnected SaaS applications. Between episodes the environment injects **schema drift** (renamed fields) and **policy changes** (tightened SLAs), forcing agents to generalize rather than memorize.

Built for the [Meta PyTorch × Scaler OpenEnv Hackathon](https://huggingface.co/) — targeting the **Multi-App Enterprise Workflow** sub-theme.

---

## Resources

| | |
|---|---|
| 🤗 Environment Space | **[huggingface.co/spaces/tanvibisht/orgos-openenv](https://huggingface.co/spaces/tanvibisht/orgos-openenv)** |
| 🏋️ Training Space | **[huggingface.co/spaces/muskansingh1101/orgos-training](https://huggingface.co/spaces/muskansingh1101/orgos-training)** |
| 📝 HF Blog Post | **[OrgOS: Teaching Agents to Survive Enterprise API Drift](https://huggingface.co/blog/muskansingh1101/orgos-openenv)** |
| 📓 Training Notebook | **[training/grpo_orgos.ipynb](training/grpo_orgos.ipynb)** |

---

## Live Demo

🚀 **[HuggingFace Space →](https://huggingface.co/spaces/tanvibisht/orgos-openenv)**

```bash
# Local quickstart
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000 for the live dashboard
```

---

## What Makes OrgOS Unique

| Feature | Description |
|---|---|
| **4 Mock SaaS Apps** | Jira, Zendesk, Salesforce, Workday — each with realistic operations |
| **Schema Drift** | Fields rename between episodes (e.g. `priority → severity → urgency_level`). Agent gets `-0.20` for stale names, `+0.10` for adapted names |
| **Policy Drift** | Every 3rd episode, SLA thresholds tighten automatically |
| **3 Workflows** | Cross-app tasks of increasing complexity: Bug Fix → Onboarding → Churn Alert |
| **RBAC** | Support vs. manager roles enforced; `-0.25` penalty for unauthorized actions |
| **Dense Reward** | Per-step composite signal tied to 5 measurable business outcomes |

---

## Applications & Operations

| App | Key Operations |
|---|---|
| **Jira** | `get_issue`, `create_issue`, `update_status`, `set_priority`, `assign_owner`, `link_zendesk_ticket`, `close_issue`, `list_issues` |
| **Zendesk** | `get_ticket`, `acknowledge_ticket`, `set_urgency`, `assign_agent`, `escalate_to_jira`, `resolve_ticket`, `add_note`, `list_tickets` |
| **Salesforce** | `get_account`, `list_accounts`, `update_deal_stage`, `flag_churn_risk`, `assign_account_owner`, `log_interaction`, `get_opportunity` |
| **Workday** | `get_employee`, `list_employees`, `provision_access`, `log_sla_event`, `request_budget_approval`, `create_onboarding_task`, `complete_task` |

---

## Workflows

### Workflow A — Customer Bug Fix (support role, 5 steps, max 15)
1. Acknowledge Zendesk ticket
2. Create linked Jira issue
3. Assign Jira issue to engineer
4. Log SLA event in Workday
5. Query Salesforce for account health

### Workflow B — Employee Onboarding (manager role, 4 steps, max 20)
1. Create employee record in Workday
2. Provision Jira access
3. Add employee to Salesforce team
4. Create Zendesk support profile

### Workflow C — Churn Risk Alert (support role, 4 steps, max 18)
1. Flag churn risk in Salesforce
2. Escalate to Zendesk ticket
3. Create Jira tracking issue
4. Log SLA event in Workday

---

## Action / Observation Format

**Action:**
```json
{"app": "zendesk", "operation": "acknowledge_ticket", "args": {"ticket_number": "ZD-001"}}
```

**Observation (key fields):**
```json
{
  "workflow_goal": "Resolve customer bug report end-to-end",
  "pending_steps": ["Assign Jira issue to engineer", "Log SLA event in Workday"],
  "schema_hints": {"jira.priority": "severity"},
  "active_rules": {"sla_p0_minutes": 30},
  "current_score": 0.42,
  "message": "Jira issue JI-001 created and linked to ZD-001"
}
```

---

## Reward Function

```
score = 0.30 × workflow_completion
      + 0.25 × rule_compliance
      + 0.20 × schema_adaptation
      + 0.15 × efficiency
      + 0.10 × policy_drift_handling

Per-step delta = new_score − old_score
Schema error penalty   = −0.20
RBAC violation penalty = −0.25
Terminal completion bonus = +0.20
```

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
| `GET` | `/ui/run-agent` | SSE stream: live agent inference |

---

## Training

The [`training/grpo_orgos.ipynb`](training/grpo_orgos.ipynb) notebook trains **Qwen2.5-3B-Instruct** with **Unsloth 4-bit LoRA** using **HF TRL GRPOTrainer** (150 GRPO steps, multi-step reward, Drive checkpoints every 30 steps).

Also runnable as a live HF Space: **[muskansingh1101/orgos-training](https://huggingface.co/spaces/muskansingh1101/orgos-training)**

### Results

| Workflow | Before GRPO | After GRPO | Δ |
|---|---|---|---|
| A — Customer Bug Fix | 0.70 | ~0.82 | +0.12 |
| B — Employee Onboarding | 0.57 | ~0.74 | +0.17 |
| C — Churn Risk Alert | 0.25 | ~0.48 | +0.23 |
| **Average** | **0.50** | **~0.68** | **+0.18** |

![Training Curve](training/plots/training_curve.png)
*Reward per training step — 150 GRPO steps on Qwen2.5-3B-Instruct*

![Baseline vs Trained](training/plots/baseline_vs_trained.png)
*Per-workflow score: untrained baseline vs. GRPO-trained agent*

![Score Distribution](training/plots/score_distribution.png)
*Distribution of episode scores before and after training*

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
export HF_TOKEN=your_token
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
│   ├── schema_drift.py     # Per-episode field renames
│   ├── business_rules.py   # RBAC + SLA enforcement
│   ├── workflow_engine.py  # 3 cross-app workflow definitions
│   ├── data_generator.py   # Synthetic data (seed=42)
│   └── apps/
│       ├── jira.py
│       ├── zendesk.py
│       ├── salesforce.py
│       └── workday.py
├── models.py               # Pydantic models
├── client.py               # OrgOSEnvClient
├── inference.py            # Baseline inference loop + SSE generator
├── ui/index.html           # Live dashboard (Tailwind + Alpine.js + Chart.js)
├── training/
│   └── grpo_orgos.ipynb   # GRPO training notebook (Colab)
├── openenv.yaml            # OpenEnv manifest
└── Dockerfile
```

---

MIT License · Built for Meta PyTorch × Scaler OpenEnv Hackathon Round 2
