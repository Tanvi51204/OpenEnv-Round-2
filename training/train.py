"""
OrgOS GRPO Training Script
Equivalent to training/grpo_orgos.ipynb but runs headlessly.

Outputs:
  training_log.txt       — structured training log for submission
  before_after_curves.png — score improvement chart
  orgos_lora_adapter/    — trained LoRA weights
"""

import datetime
import json
import os
import re
import subprocess
import sys
import time
from typing import List

import httpx
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

MODEL_NAME             = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
ENV_URL                = "http://localhost:8000"
LOG_FILE               = "training_log.txt"
N_PROMPTS_PER_WORKFLOW = 20
N_EVAL                 = 10
NUM_EPOCHS             = 3
BATCH_SIZE             = 4
GRAD_ACCUM             = 2
LR                     = 5e-5
NUM_GEN                = 4
TEMPERATURE            = 0.8
BETA                   = 0.04
LORA_R                 = 16
MAX_SEQ_LEN            = 2048

# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------

with open(LOG_FILE, "w") as f:
    f.write(f"# OrgOS GRPO Training Log\n")
    f.write(f"# Generated: {datetime.datetime.utcnow().isoformat()}Z\n\n")


def tlog(line: str) -> None:
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------
# Start OrgOS environment server
# ------------------------------------------------------------------

def start_env_server():
    print("Starting OrgOS environment server...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until healthy
    for _ in range(20):
        time.sleep(2)
        try:
            health = httpx.get(f"{ENV_URL}/health", timeout=5).json()
            if health.get("status") == "healthy":
                tlog(f"[ENV] status=healthy version={health.get('version', '?')}")
                return proc
        except Exception:
            pass
    raise RuntimeError("OrgOS server failed to start after 40 seconds")


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,
        load_in_4bit   = True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r              = LORA_R,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha     = LORA_R,
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tlog(f"[TRAIN_CONFIG] model={MODEL_NAME} lora_r={LORA_R} "
         f"max_seq_len={MAX_SEQ_LEN} trainable_params={trainable:,} quantization=4bit")
    return model, tokenizer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are OrgOS Agent — an enterprise workflow automation agent.
You operate across four SaaS applications: Jira, Zendesk, Salesforce, and Workday.

Each turn you receive a JSON observation with:
  - workflow_goal    : the task you must complete
  - pending_steps    : remaining steps in the workflow
  - app_states       : current state of each app
  - schema_hints     : field renames in effect this episode (e.g. {"jira.priority": "severity"})
  - active_rules     : current SLA / approval thresholds
  - message          : feedback from the last action
  - current_score    : your cumulative score (0.001-0.999)

Respond ONLY with a valid JSON object — no markdown, no explanation.

Action format:
  {"app": "<app>", "operation": "<op>", "args": {...}}

Available apps and key operations:
  jira:       get_issue, create_issue, update_status, set_priority, assign_owner,
              add_label, link_zendesk_ticket, close_issue, list_issues
  zendesk:    get_ticket, acknowledge_ticket, set_urgency, assign_agent,
              escalate_to_jira, resolve_ticket, add_note, list_tickets,
              create_agent_profile
  salesforce: get_account, list_accounts, update_deal_stage, flag_churn_risk,
              assign_account_owner, log_interaction, get_opportunity
  workday:    get_employee, list_employees, provision_access, log_sla_event,
              request_budget_approval, create_onboarding_task, complete_task

CRITICAL RULES:
1. Read schema_hints FIRST — if "jira.priority" -> "severity", use "severity" not "priority" in args.
2. Complete ALL pending_steps in order.
3. Do not repeat a successful action.
4. If an operation fails, read the message carefully and adapt.
5. Use list_* operations to discover record IDs when needed.
6. Stop when pending_steps is empty or done=true.
"""


def obs_to_text(obs: dict) -> str:
    hints   = obs.get("schema_hints", {})
    pending = obs.get("pending_steps", [])
    lines = [
        f"current_score: {obs['current_score']}",
        f"step_count:    {obs['step_count']}",
        f"workflow_id:   {obs['workflow_id']}",
        "",
        "=== WORKFLOW GOAL ===",
        obs["workflow_goal"],
        "",
        "=== PENDING STEPS ===",
        "\n".join(f"  - {s}" for s in pending) or "  (all steps complete!)",
        "",
        "=== SCHEMA HINTS (use these field names) ===",
        json.dumps(hints, indent=2) if hints else "  (no drift — use canonical names)",
        "",
        "=== ACTIVE RULES ===",
        json.dumps(obs.get("active_rules", {}), indent=2),
        "",
        "=== LAST MESSAGE ===",
        obs["message"],
        "",
        "=== APP STATES ===",
    ]
    # workflow-relevant apps only — skip apps the workflow doesn't touch
    WORKFLOW_APPS = {
        "A": {"jira", "zendesk", "salesforce", "workday"},
        "B": {"zendesk", "salesforce", "workday"},
        "C": {"jira", "zendesk", "salesforce"},
    }
    relevant = WORKFLOW_APPS.get(
        obs.get("workflow_id", "A"),
        {"jira", "zendesk", "salesforce", "workday"},
    )
    for app_name, view in obs.get("app_states", {}).items():
        if app_name not in relevant:
            continue
        lines.append(f"  [{app_name.upper()}]")
        view_str = str(view)
        if len(view_str) > 600:
            view_str = view_str[:600] + "...[truncated]"
        lines.append(f"  {view_str}")
        lines.append("")
    return "\n".join(lines)


def parse_action(text: str):
    text = re.sub(r"```(?:json)?\s*", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def build_prompt(obs_text: str, tokenizer) -> str:
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n---\n\n" + obs_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ------------------------------------------------------------------
# Prompt dataset
# ------------------------------------------------------------------

def build_prompt_dataset(tokenizer) -> Dataset:
    rows = []
    print("Collecting prompts from env resets...", flush=True)
    for wf in ["A", "B", "C"]:
        for _ in range(N_PROMPTS_PER_WORKFLOW):
            result   = httpx.post(f"{ENV_URL}/reset", json={"workflow_id": wf}).json()
            obs      = result["observation"]
            obs_text = obs_to_text(obs)
            rows.append({
                "prompt":      build_prompt(obs_text, tokenizer),
                "workflow_id": wf,
                "obs_text":    obs_text,
            })
    tlog(f"[TRAIN_CONFIG] algorithm=GRPO prompts={len(rows)} "
         f"workflows=A,B,C prompts_per_workflow={N_PROMPTS_PER_WORKFLOW}")
    return Dataset.from_list(rows)


# ------------------------------------------------------------------
# Reward function
# ------------------------------------------------------------------

def orgos_reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]:
    workflow_ids = kwargs.get("workflow_id", ["A"] * len(completions))
    rewards = []
    for completion, wf_id in zip(completions, workflow_ids):
        action = parse_action(completion)
        if action is None:
            rewards.append(-0.1)
            continue
        try:
            httpx.post(f"{ENV_URL}/reset", json={"workflow_id": wf_id}, timeout=10)
            result = httpx.post(f"{ENV_URL}/step", json=action, timeout=10).json()
            rewards.append(float(result["reward"]))
        except Exception:
            rewards.append(-0.1)
    return rewards


# ------------------------------------------------------------------
# Episode evaluation
# ------------------------------------------------------------------

def run_episode_with_model(model, tokenizer, workflow_id: str, max_steps: int = 15) -> float:
    result = httpx.post(f"{ENV_URL}/reset", json={"workflow_id": workflow_id}).json()
    obs    = result["observation"]

    for _ in range(max_steps):
        if obs["done"]:
            break

        # Stateless single-turn prompt — matches the GRPO training format.
        # obs["message"] already carries last-action feedback, so no history needed.
        obs_text = obs_to_text(obs)
        messages = [{"role": "user",
                     "content": SYSTEM_PROMPT + "\n\n---\n\n" + obs_text}]

        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = 256,
                temperature    = 0.0,
                do_sample      = False,
                pad_token_id   = tokenizer.eos_token_id,
            )
        action_str = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        action = parse_action(action_str)
        if action is None:
            break

        result = httpx.post(f"{ENV_URL}/step", json=action).json()
        obs    = result["observation"]
        if obs["done"]:
            break

    return obs.get("current_score", 0.001)


def evaluate(model, tokenizer, phase: str) -> dict:
    scores = {wf: [] for wf in ["A", "B", "C"]}
    tlog(f"[EVAL_START] phase={phase}")
    for wf in ["A", "B", "C"]:
        for ep in range(N_EVAL):
            score = run_episode_with_model(model, tokenizer, wf)
            scores[wf].append(score)
            tlog(f"[EVAL] phase={phase} workflow={wf} episode={ep+1} score={score:.4f}")
        wf_mean = np.mean(scores[wf])
        tlog(f"[EVAL_WORKFLOW] phase={phase} workflow={wf} "
             f"mean={wf_mean:.4f} min={min(scores[wf]):.4f} max={max(scores[wf]):.4f}")
    overall = np.mean([s for v in scores.values() for s in v])
    tlog(f"[EVAL_END] phase={phase} overall_mean={overall:.4f}")
    return scores


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

def plot_results(baseline_scores: dict, post_scores: dict) -> None:
    fig = plt.figure(figsize=(14, 8), facecolor="#0f172a")
    fig.suptitle("OrgOS: Before vs After GRPO Training", fontsize=15,
                 color="white", fontweight="bold", y=0.98)

    gs     = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    COLORS = {"before": "#f87171", "after": "#34d399", "bg": "#1e293b", "grid": "#334155"}
    LABELS = {
        "A": "Workflow A\nCustomer Bug Fix",
        "B": "Workflow B\nEmployee Onboarding",
        "C": "Workflow C\nChurn Risk Alert",
    }

    for col, wf in enumerate(["A", "B", "C"]):
        ax    = fig.add_subplot(gs[0, col])
        ax.set_facecolor(COLORS["bg"])
        ax.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.7)
        before = baseline_scores[wf]
        after  = post_scores[wf]
        delta  = np.mean(after) - np.mean(before)
        ax.plot(before, color=COLORS["before"], linewidth=1.5, alpha=0.8, label="Before GRPO")
        ax.plot(after,  color=COLORS["after"],  linewidth=1.5, alpha=0.8, label="After GRPO")
        ax.axhline(np.mean(before), color=COLORS["before"], linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(np.mean(after),  color=COLORS["after"],  linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(LABELS[wf] + f"\n(Δ = {delta:+.4f})", color="white", fontsize=9)
        ax.set_xlabel("Episode", color="#94a3b8", fontsize=8)
        ax.set_ylabel("Final Score", color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#64748b", labelsize=7)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, facecolor="#1e293b", labelcolor="white",
                  edgecolor="#475569", framealpha=0.8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    ax_hist = fig.add_subplot(gs[1, :])
    ax_hist.set_facecolor(COLORS["bg"])
    ax_hist.grid(color=COLORS["grid"], linewidth=0.5, alpha=0.5, axis="x")
    all_before = [s for v in baseline_scores.values() for s in v]
    all_after  = [s for v in post_scores.values() for s in v]
    bins = np.linspace(0, 1, 25)
    ax_hist.hist(all_before, bins=bins, color=COLORS["before"], alpha=0.6,
                 label=f"Before GRPO  (mean={np.mean(all_before):.4f})", edgecolor="none")
    ax_hist.hist(all_after,  bins=bins, color=COLORS["after"],  alpha=0.6,
                 label=f"After GRPO   (mean={np.mean(all_after):.4f})", edgecolor="none")
    ax_hist.axvline(np.mean(all_before), color=COLORS["before"], linestyle="--", linewidth=1.5)
    ax_hist.axvline(np.mean(all_after),  color=COLORS["after"],  linestyle="--", linewidth=1.5)
    ax_hist.set_title("Score Distribution Across All Workflows", color="white", fontsize=10)
    ax_hist.set_xlabel("Final Score", color="#94a3b8", fontsize=9)
    ax_hist.set_ylabel("Count", color="#94a3b8", fontsize=9)
    ax_hist.tick_params(colors="#64748b", labelsize=8)
    ax_hist.legend(fontsize=9, facecolor="#1e293b", labelcolor="white",
                   edgecolor="#475569", framealpha=0.9)
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#334155")

    plt.savefig("before_after_curves.png", dpi=150, bbox_inches="tight",
                facecolor="#0f172a", edgecolor="none")
    plt.close()
    tlog("[ARTIFACT] file=before_after_curves.png")


# ------------------------------------------------------------------
# Training callback
# ------------------------------------------------------------------

class OrgOSLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step        = state.global_step
        loss        = logs.get("loss", logs.get("train_loss", "?"))
        mean_reward = logs.get("reward", logs.get("mean_reward", "?"))
        kl          = logs.get("kl", logs.get("approx_kl", "?"))
        lr_now      = logs.get("learning_rate", "?")

        loss_str   = f"{loss:.6f}"        if isinstance(loss, float)        else str(loss)
        reward_str = f"{mean_reward:.4f}" if isinstance(mean_reward, float) else str(mean_reward)
        kl_str     = f"{kl:.6f}"          if isinstance(kl, float)          else str(kl)
        lr_str     = f"{lr_now:.2e}"      if isinstance(lr_now, float)      else str(lr_now)

        tlog(f"[TRAIN_STEP] step={step} loss={loss_str} "
             f"mean_reward={reward_str} kl={kl_str} lr={lr_str}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    server_proc = start_env_server()

    try:
        model, tokenizer = load_model()

        prompt_dataset = build_prompt_dataset(tokenizer)

        # Sanity-check reward function
        test_r = orgos_reward_fn(
            completions = ['{"app": "zendesk", "operation": "list_tickets", "args": {"state": "new"}}',
                           "not json"],
            prompts     = ["", ""],
            workflow_id = ["A", "A"],
        )
        tlog(f"[REWARD_FN_CHECK] valid_action={test_r[0]:.4f} invalid_action={test_r[1]:.4f}")

        # Baseline evaluation
        FastLanguageModel.for_inference(model)
        baseline_scores = evaluate(model, tokenizer, phase="baseline")
        baseline_mean   = np.mean([s for v in baseline_scores.values() for s in v])

        # GRPO training
        model.train()
        tlog(f"[TRAIN_CONFIG] epochs={NUM_EPOCHS} batch_size={BATCH_SIZE} "
             f"grad_accum={GRAD_ACCUM} lr={LR} num_generations={NUM_GEN} "
             f"temperature={TEMPERATURE} beta_kl={BETA}")

        grpo_config = GRPOConfig(
            output_dir                  = "./orgos_grpo_ckpt",
            num_train_epochs            = NUM_EPOCHS,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRAD_ACCUM,
            learning_rate               = LR,
            warmup_steps                = 10,
            logging_steps               = 5,
            save_steps                  = 100,
            bf16                        = torch.cuda.is_bf16_supported(),
            fp16                        = not torch.cuda.is_bf16_supported(),
            max_grad_norm               = 1.0,
            num_generations             = NUM_GEN,
            max_new_tokens              = 256,
            temperature                 = TEMPERATURE,
            beta                        = BETA,
            report_to                   = "none",
            seed                        = 42,
        )

        trainer = GRPOTrainer(
            model            = model,
            args             = grpo_config,
            reward_funcs     = orgos_reward_fn,
            train_dataset    = prompt_dataset,
            processing_class = tokenizer,
            callbacks        = [OrgOSLogCallback()],
        )

        tlog("[TRAIN_START]")
        train_result = trainer.train()
        tlog(f"[TRAIN_END] total_steps={train_result.global_step} "
             f"train_loss={train_result.training_loss:.6f} "
             f"train_runtime_s={train_result.metrics.get('train_runtime', 0):.1f}")

        # Post-training evaluation
        FastLanguageModel.for_inference(model)
        post_scores = evaluate(model, tokenizer, phase="post_training")
        post_mean   = np.mean([s for v in post_scores.values() for s in v])
        improvement = post_mean - baseline_mean

        tlog(
            f"[TRAIN_SUMMARY] "
            f"model={MODEL_NAME} algorithm=GRPO "
            f"baseline_mean={baseline_mean:.4f} "
            f"post_training_mean={post_mean:.4f} "
            f"improvement={improvement:+.4f} "
            f"workflow_A_before={np.mean(baseline_scores['A']):.4f} "
            f"workflow_A_after={np.mean(post_scores['A']):.4f} "
            f"workflow_B_before={np.mean(baseline_scores['B']):.4f} "
            f"workflow_B_after={np.mean(post_scores['B']):.4f} "
            f"workflow_C_before={np.mean(baseline_scores['C']):.4f} "
            f"workflow_C_after={np.mean(post_scores['C']):.4f}"
        )

        # Save artifacts
        plot_results(baseline_scores, post_scores)
        model.save_pretrained("orgos_lora_adapter")
        tokenizer.save_pretrained("orgos_lora_adapter")
        tlog("[ARTIFACT] file=orgos_lora_adapter/")
        tlog("[ARTIFACT] file=training_log.txt")

        print(f"\nDone. Improvement: {baseline_mean:.4f} → {post_mean:.4f} ({improvement:+.4f})")

    finally:
        server_proc.terminate()


if __name__ == "__main__":
    main()
