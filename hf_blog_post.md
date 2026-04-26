# OrgOS: Teaching Agents to Survive Enterprise API Drift

*Submitted to the Meta PyTorch × Scaler OpenEnv Hackathon Round 2*

---

## The Problem

Enterprise AI agents break in production — not because the model is bad, but because the environment keeps changing. SaaS APIs rename fields. SLAs tighten. Access policies shift. An agent trained on yesterday's Jira schema fails when `priority` becomes `severity`.

Static datasets can't capture this. You need an environment that drifts.

---

## What We Built: OrgOS

**OrgOS** is a multi-app enterprise RL environment where an AI agent completes real business workflows across four interconnected mock SaaS applications: **Jira, Zendesk, Salesforce, and Workday**.

### Three Cross-App Workflows

| Workflow | Role | Steps |
|---|---|---|
| A — Customer Bug Fix | Support | Acknowledge ticket → Create Jira issue → Assign engineer → Log SLA → Check account health |
| B — Employee Onboarding | Manager | Create Workday record → Provision Jira access → Add to Salesforce → Create Zendesk profile |
| C — Churn Risk Alert | Support | Flag churn in Salesforce → Escalate to Zendesk → Create Jira tracker → Log SLA event |

### What Makes It Hard

**Schema Drift**: Every episode, field names can change across versions. `priority` → `severity` → `urgency_level`. The agent sees a `schema_hints` dict telling it the current mapping — but only if it reads it. Using stale field names incurs a `-0.20` penalty. Using adapted names earns `+0.10`.

**Policy Drift**: Every 3rd episode, SLA thresholds tighten automatically (P0 response: 30 min → 15 min). Agents that ignore `active_rules` get caught.

**RBAC**: Support vs. manager roles are strictly enforced. Unauthorized actions cost `-0.25`.

### Reward Function

```
score = 0.30 × workflow_completion
      + 0.25 × rule_compliance
      + 0.20 × schema_adaptation
      + 0.15 × efficiency
      + 0.10 × policy_drift_handling
```

The agent receives dense per-step signals, not just terminal rewards.

---

## Training: GRPO on Qwen2.5-3B

We trained **Qwen2.5-3B-Instruct** with **Unsloth 4-bit LoRA** using **HF TRL GRPOTrainer** for 150 steps.

### Key Design Choices

**Multi-step reward**: Instead of rewarding just the GRPO-generated action, we continue 1 more greedy step with the model and return the cumulative 2-step score. This prevents the model from collapsing to safe list_* operations that look good on single-step rewards but don't advance workflows.

**System prompt engineering**: The prompt explicitly instructs the agent to read `schema_hints` before choosing field names and to check `pending_steps` to know what the workflow needs next.

**Pinned TRL**: We pin `trl<=0.24` for API stability — newer versions changed the GRPOTrainer interface.

### Training Config

| Config | Value |
|---|---|
| Model | Qwen2.5-3B-Instruct (4-bit) |
| LoRA rank | r=16 |
| Steps | 150 |
| LR | 8e-6 |
| Batch | 1 (grad accum 2) |
| Reward | 2-step cumulative |

---

## Results

| Workflow | Before GRPO | After GRPO | Δ |
|---|---|---|---|
| A — Customer Bug Fix | 0.70 | ~0.82 | +0.12 |
| B — Employee Onboarding | 0.57 | ~0.74 | +0.17 |
| C — Churn Risk Alert | 0.25 | ~0.48 | +0.23 |
| **Average** | **0.50** | **~0.68** | **+0.18** |

The biggest gain is on Workflow C (Churn Risk Alert) — the hardest workflow, which requires the most cross-app coordination. The untrained model barely scores 0.25 on it; after GRPO it reaches 0.48.

The trained agent learns to:
1. Read `schema_hints` and use the current field names instead of stale canonical ones
2. Follow `pending_steps` in order instead of randomly calling available operations
3. Respect `active_rules` (SLA thresholds, RBAC permissions)

---

## Try It

- 🌐 **Environment**: [huggingface.co/spaces/tanvibisht/orgos-openenv](https://huggingface.co/spaces/tanvibisht/orgos-openenv)
- 🏋️ **Training Space**: [huggingface.co/spaces/muskansingh1101/orgos-training](https://huggingface.co/spaces/muskansingh1101/orgos-training)
- 📓 **Notebook**: [training/grpo_orgos.ipynb](https://github.com/muskansingh1101/OpenEnv-Round-2/blob/main/training/grpo_orgos.ipynb)

---

## Why It Matters

Any agent that automates enterprise workflows will face API drift. The tools it was trained on today will be renamed, versioned, or deprecated tomorrow. OrgOS is a controlled environment for studying exactly this failure mode — and for training agents that adapt instead of break.

---

*Built for Meta PyTorch × Scaler OpenEnv Hackathon Round 2. MIT License.*
