"""
Baseline inference script for the OrgOS OpenEnv environment.
Runs all three workflows (A / B / C) and reports scores.

Required environment variables:
    API_BASE_URL   — LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     — model identifier (default: gpt-4o-mini)
    HF_TOKEN       — API key for the LLM endpoint
    ENV_URL        — environment server URL (default: http://localhost:8000)

STDOUT FORMAT (OpenEnv spec):
    [START] task=<workflow_name> env=orgos-openenv model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   task=<workflow_name> score=<0.00> steps=<n>
"""

import json
import os
import re
import sys
import time
from typing import AsyncGenerator, Dict, List, Optional

import httpx
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",       "http://localhost:8000")

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN is not set — LLM calls may fail.", file=sys.stderr)

llm_client = OpenAI(api_key=HF_TOKEN or "sk-placeholder", base_url=API_BASE_URL)

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are OrgOS Agent — an enterprise workflow automation agent.
You operate across four SaaS applications: Jira, Zendesk, Salesforce, and Workday.

Each turn you receive an observation with:
  - workflow_goal    : the business objective you must achieve
  - completed_steps  : step IDs already finished (e.g. ["A1", "A2"])
  - app_states       : current state of each app — your primary source of truth
  - schema_hints     : PARTIAL field renames in effect (e.g. {"jira.priority": "severity"})
                       Not all drift is revealed — probe with get_* if a field is rejected.
  - active_rules     : current SLA / approval thresholds
  - message          : feedback from the last action
  - current_score    : your cumulative score (0.001–0.999)

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
1. Read schema_hints FIRST — if "jira.priority" → "severity", use "severity" not "priority" in args.
   If a field is rejected, use get_* or list_* to probe the current schema before retrying.
2. Inspect app_states to determine what has been done and what still needs action.
3. Use list_* and get_* operations to discover record IDs — never assume them.
4. Do not repeat a successful action.
5. If an operation fails, read the message and adapt your field names or args.
6. Stop when completed_steps covers all workflow steps or done=true.
"""

WORKFLOW_NAMES = {
    "A": "workflow-a-bug-fix",
    "B": "workflow-b-onboarding",
    "C": "workflow-c-churn-alert",
}

# ------------------------------------------------------------------
# OpenEnv stdout logging helpers
# ------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(task_name: str, score: float, steps: int) -> None:
    safe_score = max(0.001, min(0.999, float(score)))
    print(f"[END] task={task_name} score={safe_score:.4f} steps={steps}", flush=True)


# ------------------------------------------------------------------
# HTTP helpers
# ------------------------------------------------------------------

def api_post(path: str, payload: dict = None) -> dict:
    url  = ENV_URL.rstrip("/") + path
    resp = httpx.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_get(path: str) -> dict:
    url  = ENV_URL.rstrip("/") + path
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ------------------------------------------------------------------
# Observation formatter
# ------------------------------------------------------------------

def obs_to_text(obs: dict) -> str:
    completed = obs.get("completed_steps", [])
    completed_str = ", ".join(completed) if completed else "none"

    lines = [
        f"current_score:  {obs['current_score']}",
        f"step_count:     {obs['step_count']}",
        f"workflow_id:    {obs['workflow_id']}",
        f"completed_steps: [{completed_str}]",
        "",
        "=== WORKFLOW GOAL ===",
        obs["workflow_goal"],
        "",
        "=== SCHEMA HINTS (partial — probe with get_* for unknown fields) ===",
        json.dumps(obs["schema_hints"], indent=2) if obs["schema_hints"] else "  (no drift — use canonical names)",
        "",
        "=== ACTIVE RULES ===",
        json.dumps(obs["active_rules"], indent=2),
        "",
        "=== LAST MESSAGE ===",
        obs["message"],
        "",
        "=== APP STATES (use these to determine what still needs to be done) ===",
    ]
    for app_name, view in obs.get("app_states", {}).items():
        lines.append(f"  [{app_name.upper()}]")
        lines.append(f"  {view}")
        lines.append("")
    if obs.get("rule_violations"):
        lines.append("=== RULE VIOLATIONS (fix these!) ===")
        for v in obs["rule_violations"]:
            lines.append(f"  ⚠  {v}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Single-workflow inference loop
# ------------------------------------------------------------------

def run_workflow(workflow_id: str) -> float:
    task_name = WORKFLOW_NAMES.get(workflow_id, f"workflow-{workflow_id.lower()}")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Running Workflow {workflow_id}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    result  = api_post("/reset", {"workflow_id": workflow_id})
    obs     = result["observation"]
    history: List[dict] = []
    steps_taken = 0

    log_start(task=task_name, env_name="orgos-openenv", model=MODEL_NAME)

    try:
        for step_num in range(1, 60):
            if obs["done"]:
                break

            obs_text = obs_to_text(obs)
            history.append({"role": "user", "content": obs_text})

            # Trim history to avoid context overflow
            # if len(history) > 20:
            #     history = history[-20:]
            # Trim history — always keep an even number so roles alternate correctly
            if len(history) > 20:
                history = history[-20:]
            # Ensure history starts with a user message (Gemma requires strict alternation)
            if history and history[0]["role"] != "user":
                history = history[1:]            

            try:
                response = llm_client.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = [{"role": "system", "content": SYSTEM_PROMPT}] + history,
                    temperature = 0.0,
                    max_tokens  = 300,
                )
                action_str = response.choices[0].message.content.strip()
            except Exception as exc:
                print(f"  Step {step_num}: LLM call failed: {exc}", file=sys.stderr)
                log_step(step_num, "null", 0.0, True, str(exc))
                break

            history.append({"role": "assistant", "content": action_str})

            # Parse action JSON
            action = None
            try:
                action = json.loads(action_str)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", action_str, re.DOTALL)
                if m:
                    try:
                        action = json.loads(m.group())
                    except Exception:
                        pass

            if action is None:
                print(f"  Step {step_num}: Could not parse action JSON.", file=sys.stderr)
                log_step(step_num, action_str, -0.05, False, "json_parse_error")
                break

            action_label = json.dumps(action, separators=(",", ":"))
            print(
                f"  Step {step_num:2d} | score={obs['current_score']:.4f} | {action_label}",
                file=sys.stderr,
            )

            result      = api_post("/step", action)
            obs         = result["observation"]
            step_reward = result["reward"]
            done        = result["done"]
            error_msg   = (
                obs["message"]
                if obs.get("rule_violations") or step_reward < 0
                else None
            )

            print(f"           → {obs['message']}", file=sys.stderr)

            steps_taken = step_num
            log_step(
                step   = step_num,
                action = action_label,
                reward = step_reward,
                done   = done,
                error  = error_msg,
            )

            if done:
                break

            time.sleep(0.2)

    finally:
        final = obs.get("current_score", 0.001) if isinstance(obs, dict) else 0.001
        log_end(task_name=task_name, score=final, steps=steps_taken)

    final_score = obs["current_score"]
    wf_done     = not obs.get("pending_steps")
    print(
        f"\n  Workflow {workflow_id} final score: {final_score:.4f}  "
        f"steps: {obs['step_count']}  completed: {wf_done}",
        file=sys.stderr,
    )
    return final_score


# ------------------------------------------------------------------
# Async generator for SSE streaming from the UI
# ------------------------------------------------------------------

async def run_workflow_generator(
    workflow_id: str = "A",
    env_ref=None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator that runs one inference episode and yields
    SSE-friendly event dicts for the dashboard UI.

    Each yielded dict has a "type" key:
      "reset"  — episode started
      "step"   — one action taken
      "done"   — episode ended
      "error"  — something went wrong
    """
    import asyncio

    if env_ref is None:
        # Fall back to HTTP if no direct env reference
        result = api_post("/reset", {"workflow_id": workflow_id})
    else:
        from models import OrgOSAction as _Action
        obs_obj = env_ref.reset(workflow_id=workflow_id)
        result  = {"observation": obs_obj.model_dump(), "reward": obs_obj.reward, "done": False}

    obs     = result["observation"]
    history: List[dict] = []

    yield {"type": "reset", "observation": obs, "workflow_id": workflow_id}
    await asyncio.sleep(0)

    for step_num in range(1, 60):
        if obs["done"]:
            break

        obs_text = obs_to_text(obs)
        history.append({"role": "user", "content": obs_text})
        if len(history) > 20:
            history = history[-20:]

        try:
            response = llm_client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = [{"role": "system", "content": SYSTEM_PROMPT}] + history,
                temperature = 0.0,
                max_tokens  = 300,
            )
            action_str = response.choices[0].message.content.strip()
        except Exception as exc:
            yield {"type": "error", "step": step_num, "message": str(exc)}
            break

        history.append({"role": "assistant", "content": action_str})

        action = None
        try:
            action = json.loads(action_str)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", action_str, re.DOTALL)
            if m:
                try:
                    action = json.loads(m.group())
                except Exception:
                    pass

        if action is None:
            yield {"type": "error", "step": step_num, "message": "JSON parse error"}
            break

        if env_ref is None:
            result = api_post("/step", action)
        else:
            from models import OrgOSAction as _Action
            try:
                act     = _Action(**action)
                obs_obj = env_ref.step(act)
                result  = {
                    "observation": obs_obj.model_dump(),
                    "reward":      obs_obj.reward,
                    "done":        obs_obj.done,
                }
            except Exception as exc:
                yield {"type": "error", "step": step_num, "message": str(exc)}
                break

        obs         = result["observation"]
        step_reward = result["reward"]
        done        = result["done"]

        yield {
            "type":        "step",
            "step":        step_num,
            "action":      action,
            "observation": obs,
            "reward":      step_reward,
            "done":        done,
        }
        await asyncio.sleep(0)

        if done:
            break

    yield {
        "type":        "done",
        "final_score": obs.get("current_score", 0.001),
        "steps":       obs.get("step_count", step_num),
        "completed":   not obs.get("pending_steps"),
    }


# ------------------------------------------------------------------
# Main — run all three workflows sequentially
# ------------------------------------------------------------------

def main():
    print("OrgOS OpenEnv — Baseline Inference", file=sys.stderr)
    print(f"Model : {MODEL_NAME}", file=sys.stderr)
    print(f"Env   : {ENV_URL}", file=sys.stderr)

    try:
        health = api_get("/health")
        assert health.get("status") in ("ok", "healthy"), f"Unexpected status: {health}"
        print("Health check: OK\n", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] Environment not reachable at {ENV_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    scores: Dict[str, float] = {}
    for wf_id in ["A", "B", "C"]:
        try:
            scores[f"workflow_{wf_id}"] = run_workflow(wf_id)
        except Exception as exc:
            print(f"[ERROR] Workflow {wf_id} failed: {exc}", file=sys.stderr)
            scores[f"workflow_{wf_id}"] = 0.001

    print("\n" + "="*60, file=sys.stderr)
    print("  BASELINE RESULTS", file=sys.stderr)
    print("="*60, file=sys.stderr)
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}", file=sys.stderr)
    avg = round(sum(scores.values()) / len(scores), 4)
    print(f"  average: {avg:.4f}", file=sys.stderr)
    print("="*60, file=sys.stderr)

    with open("baseline_scores.json", "w") as f:
        json.dump({"scores": scores, "average": avg}, f, indent=2)
    print("\nScores written to baseline_scores.json", file=sys.stderr)


if __name__ == "__main__":
    main()
