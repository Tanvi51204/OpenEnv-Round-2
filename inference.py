"""
Baseline inference script for the Data Cleaning OpenEnv environment.
Uses the OpenAI client against all 3 tasks and reports scores.

Required environment variables:
    API_BASE_URL   — LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     — model identifier
    HF_TOKEN       — API key
    ENV_URL        — environment server URL (default: http://localhost:8000)

STDOUT FORMAT (OpenEnv spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   task=<task_name> score=<0.00> steps=<n>
"""

import json
import os
import re
import sys
import time
from typing import List, Optional
import httpx
from openai import OpenAI

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:8000")

if not HF_TOKEN:
    print("[WARNING] HF_TOKEN is not set — LLM calls may fail.", file=sys.stderr)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are a data cleaning agent. You control a data cleaning environment
through JSON actions. Each turn you receive an observation JSON describing the current state
of a dataset (preview, missing counts, duplicate count, dtype issues, current score, etc.)
and a task description.

Your job is to pick the single best action to improve the dataset quality.

Respond ONLY with a valid JSON object — no markdown, no explanation, just the JSON.

Available operations and their required parameters:

1. fill_missing
   {"operation": "fill_missing", "column": "<col>", "params": {"strategy": "median|mean|mode|constant", "value": <only if constant>}}

2. drop_duplicates
   {"operation": "drop_duplicates"}

3. fix_format
   {"operation": "fix_format", "column": "phone|listed_date|signup_date|country"}

4. replace_value
   {"operation": "replace_value", "column": "<col>", "params": {"old": "<val>", "new": "<val>"}}

5. drop_outliers
   {"operation": "drop_outliers", "column": "<numeric_col>"}

6. fix_dtype
   {"operation": "fix_dtype", "column": "<col>", "params": {"dtype": "float|int|str"}}

Rules:
- Address the highest-impact issues first (missing values > duplicates > outliers > format).
- Do not repeat an operation that returned no effect (watch the 'message' field).
- Stop when current_score >= 0.95.
"""


# ------------------------------------------------------------------
# OpenEnv stdout logging helpers
# ------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task_name: str, score: float, steps: int) -> None:
    safe_score = max(0.01, min(0.99, float(score)))
    print(
        f"[END] task={task_name} score={safe_score:.4f} steps={steps}",
        flush=True
    )


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
# Agent loop
# ------------------------------------------------------------------

def obs_to_text(obs: dict) -> str:
    lines = [
        f"current_score: {obs['current_score']}",
        f"step_count:    {obs['step_count']}",
        f"data_shape:    {obs['data_shape']}",
        f"duplicate_count: {obs['duplicate_count']}",
        f"missing_counts: {json.dumps(obs['missing_counts'])}",
        f"dtype_issues:   {json.dumps(obs['dtype_issues'])}",
        f"message:        {obs['message']}",
        "",
        "=== DATA PREVIEW (first 10 rows) ===",
        obs["data_preview"],
        "",
        "=== TASK DESCRIPTION ===",
        obs["task_description"],
    ]
    return "\n".join(lines)


def run_task(task_id: int) -> float:
    task_name = f"data-cleaning-task{task_id}"

    # Human-readable header (stderr so it doesn't interfere with stdout format)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  Running Task {task_id}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    result  = api_post("/reset", {"task_id": task_id})
    obs     = result["observation"]
    history = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_name, env="data-cleaning-openenv", model=MODEL_NAME)

    try:
        for step_num in range(1, 50):
            if obs["done"]:
                success = obs["current_score"] >= 0.95
                break

            obs_text = obs_to_text(obs)
            history.append({"role": "user", "content": obs_text})

            try:
                response = client.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = [{"role": "system", "content": SYSTEM_PROMPT}] + history,
                    temperature = 0.0,
                    max_tokens  = 256,
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
                print(f"  Step {step_num}: Could not parse action JSON, skipping.", file=sys.stderr)
                log_step(step_num, action_str, -0.05, False, "json_parse_error")
                break

            action_label = json.dumps(action, separators=(",", ":"))
            print(
                f"  Step {step_num:2d} | score={obs['current_score']:.4f} | action={action_label}",
                file=sys.stderr,
            )

            result      = api_post("/step", action)
            obs         = result["observation"]
            step_reward = result["reward"]
            done        = result["done"]
            error_msg   = None if obs["message"].startswith("Fill") or step_reward >= 0 else obs["message"]

            print(f"           -> {obs['message']}", file=sys.stderr)

            rewards.append(step_reward)
            steps_taken = step_num

            log_step(
                step   = step_num,
                action = action_label,
                reward = step_reward,
                done   = done,
                error  = error_msg,
            )

            if done:
                success = obs["current_score"] >= 0.95
                break

            time.sleep(0.3)

    finally:
        final = obs.get("current_score", 0.01) if isinstance(obs, dict) else 0.01
        log_end(task_name=task_name, score=final, steps=steps_taken)

    final_score = obs["current_score"]
    print(
        f"\n  Task {task_id} final score: {final_score:.4f}  (steps used: {obs['step_count']})",
        file=sys.stderr,
    )
    return final_score


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("Data Cleaning OpenEnv -- Baseline Inference", file=sys.stderr)
    print(f"Model : {MODEL_NAME}", file=sys.stderr)
    print(f"Env   : {ENV_URL}", file=sys.stderr)

    # Smoke-test health endpoint
    try:
        health = api_get("/health")
        assert health.get("status") in ("ok", "healthy"), f"Unexpected status: {health}"
        print("Health check: OK\n", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] Environment not reachable at {ENV_URL}: {exc}", file=sys.stderr)
        print("[ERROR] Make sure the server is running and ENV_URL is correct.", file=sys.stderr)
        sys.exit(1)

    scores = {}
    for task_id in [1, 2, 3]:
        try:
            scores[f"task{task_id}"] = run_task(task_id)
        except Exception as exc:
            print(f"[ERROR] Task {task_id} failed: {exc}", file=sys.stderr)
            scores[f"task{task_id}"] = 0.01

    print("\n" + "="*60, file=sys.stderr)
    print("  BASELINE RESULTS", file=sys.stderr)
    print("="*60, file=sys.stderr)
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}", file=sys.stderr)
    avg = round(sum(scores.values()) / len(scores), 4)
    print(f"  average: {avg:.4f}", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # Write scores to file for automated validators
    with open("baseline_scores.json", "w") as f:
        json.dump({"scores": scores, "average": avg}, f, indent=2)
    print("\nScores written to baseline_scores.json", file=sys.stderr)


if __name__ == "__main__":
    main()