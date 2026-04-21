---
title: Data Cleaning Environment
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - rl
  - data-cleaning
---

# Data Cleaning OpenEnv

A **real-world data cleaning environment** for training and evaluating AI agents.

An agent interacts with a dirty pandas DataFrame through a standard `reset() / step() / state()` HTTP API, learning to fix common data quality problems — missing values, duplicate rows, inconsistent formats, statistical outliers, and dtype errors — across three progressively harder tasks.

🤗 **Live HuggingFace Space:** https://srishtichugh-openenv-hack.hf.space
📖 **Interactive API docs:** https://srishtichugh-openenv-hack.hf.space/docs
✅ **Health check:** https://srishtichugh-openenv-hack.hf.space/health

---

## Environment Description & Motivation

Real-world datasets are almost never clean. Data engineers routinely spend 60–80 % of their time on data cleaning tasks: filling missing values with statistically appropriate strategies, removing duplicates, standardising inconsistent formats (phone numbers, dates, country names), and detecting extreme outliers.

This environment turns those tasks into a reinforcement learning challenge with:

- **Deterministic, programmatic graders** — ground-truth clean DataFrames are generated with a fixed seed; every reward signal is reproducible.
- **Meaningful partial rewards** — every step emits a delta reward proportional to how much of the dataset it cleaned, so the agent receives useful signal throughout the episode rather than only at the end.
- **Three difficulty levels** — easy, medium, hard — letting agents learn a curriculum from simple null-filling up to full multi-issue pipelines.
- **No external data downloads** — all datasets are generated synthetically via `numpy` + `Faker` with `seed=42`.

---

## Action Space

Actions are JSON objects sent to `POST /step`.

| `operation` | Required `column` | `params` | Description |
|---|---|---|---|
| `fill_missing` | ✅ | `{"strategy": "median\|mean\|mode\|constant", "value": ...}` | Fill NaN values in a column |
| `drop_duplicates` | ❌ | — | Remove all duplicate rows |
| `fix_format` | ✅ | — | Standardise phone/date/country format |
| `replace_value` | ✅ | `{"old": ..., "new": ...}` | Replace a specific value |
| `drop_outliers` | ✅ | — | Remove IQR outliers from a numeric column |
| `fix_dtype` | ✅ | `{"dtype": "float\|int\|str"}` | Cast column to correct dtype |

**Format rules enforced by `fix_format`:**

| Column | Target format |
|---|---|
| `phone` | `NNN-NNN-NNNN` |
| `listed_date` / `signup_date` | `YYYY-MM-DD` |
| `country` | Title-cased canonical name (`USA`, `UK`, `Canada`, `Australia`, `Germany`) |

**Example actions:**
```json
{"operation": "fill_missing",    "column": "salary",          "params": {"strategy": "median"}}
{"operation": "fill_missing",    "column": "department",      "params": {"strategy": "mode"}}
{"operation": "drop_duplicates"}
{"operation": "fix_format",      "column": "phone"}
{"operation": "fix_format",      "column": "signup_date"}
{"operation": "drop_outliers",   "column": "purchase_amount"}
```

---

## Observation Space

Every `POST /reset` and `POST /step` returns:
```json
{
  "observation": {
    "done":             false,
    "reward":           0.40,
    "data_preview":     "name,age,salary,...\n...",
    "data_shape":       [100, 5],
    "missing_counts":   {"age": 20, "salary": 20, "department": 10},
    "duplicate_count":  0,
    "dtype_issues":     {},
    "task_description": "Task 1 (Easy) — Fill Missing Values\n...",
    "message":          "Filled 20 missing values in 'age' using median.",
    "step_count":       1,
    "current_score":    0.4000
  },
  "reward": 0.40,
  "done":   false,
  "info":   {}
}
```

| Field | Type | Description |
|---|---|---|
| `done` | bool | Episode finished (score ≥ 0.95 or max steps reached) |
| `reward` | float | Per-step delta reward (see Reward Function) |
| `data_preview` | string | First 10 rows of current DataFrame as CSV |
| `data_shape` | [int, int] | Current `[rows, cols]` |
| `missing_counts` | object | `{column: null_count}` for columns with NaN |
| `duplicate_count` | int | Number of duplicate rows |
| `dtype_issues` | object | `{column: issue_description}` for suspected dtype mismatches |
| `task_description` | string | Full task instructions with available operations |
| `message` | string | Human-readable result of the last action |
| `step_count` | int | Steps taken in this episode |
| `current_score` | float | Running grader score 0.0 – 1.0 |

---

## State Space

`GET /state` returns episode metadata (does not modify state):
```json
{
  "episode_id":      "a8f026a9-...",
  "task_id":         1,
  "step_count":      2,
  "max_steps":       20,
  "total_errors":    50,
  "errors_remaining": 30
}
```

---

## Tasks

### Task 1 — Fill Missing Values *(Easy)*

| Property | Value |
|---|---|
| Dataset | 100-row employee records (name, age, salary, department, experience) |
| Issues | ~20 % NaN in `age`, `salary`; ~10 % NaN in `department` |
| Goal | Fill all missing values |
| Valid operations | `fill_missing` |
| Grader | `1.0 − remaining_nulls / original_nulls` |
| Max steps | 20 |
| Optimal steps | 3 (one per affected column) |

### Task 2 — Fix Formats + Remove Duplicates *(Medium)*

| Property | Value |
|---|---|
| Dataset | 215-row product catalog (product_id, price, category, phone, listed_date) |
| Issues | ~60 % phone numbers in mixed formats, ~60 % dates in mixed formats, 15 duplicate rows |
| Goal | Standardise all phone/date formats and remove duplicates |
| Valid operations | `fix_format`, `drop_duplicates` |
| Grader | `0.35 × phone_score + 0.35 × date_score + 0.30 × dupe_score` |
| Max steps | 30 |
| Optimal steps | 3 |

### Task 3 — Full Cleaning Pipeline *(Hard)*

| Property | Value |
|---|---|
| Dataset | 320-row customer database (name, age, purchase_amount, country, email, signup_date) |
| Issues | Missing values (4 cols), 20 duplicate rows, outliers in `purchase_amount` (~3× normal), mixed country capitalisation, mixed date formats |
| Goal | Fix all issues end-to-end |
| Valid operations | All 6 operations |
| Grader | `0.25×null + 0.20×dupe + 0.20×outlier + 0.175×country + 0.175×date` |
| Max steps | 40 |
| Optimal steps | 8 |

---

## Reward Function

| Scenario | Reward |
|---|---|
| Score improves (delta > 0) | `new_score − old_score` (positive) |
| Operation had no effect | `−0.01` |
| Invalid operation / bad column | `−0.05` |
| Episode completed (score ≥ 0.95) | `delta + 0.20` terminal bonus |

Rewards are bounded to **[−0.05, 1.2]**. A partial reward is emitted on every step, giving the agent dense signal throughout the episode.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check → `{"status": "healthy"}` |
| `POST` | `/reset` | Start episode. Body: `{"task_id": 1\|2\|3}` (optional; default: round-robin) |
| `POST` | `/step` | Execute action. Body: action JSON |
| `POST` | `/state` | Get episode metadata |
| `GET` | `/metadata` | Environment name, version, task list |
| `GET` | `/schema` | Full action / observation / state JSON schemas |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Baseline Scores

| Task | Difficulty | Score |
|---|---|---|
| 1 — Fill Missing Values | Easy | 0.999 |
| 2 — Fix Formats + Duplicates | Medium | 0.999 |
| 3 — Full Cleaning Pipeline | Hard | 0.999 |
| **Average** | — | **0.999** |

*Produced by `google/gemma-3-27b-it` via NVIDIA NIM, `temperature=0`. Full step-by-step agent logs: `inference_log.txt`.*

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)

### Local — Python
```bash
# 1. Clone and install dependencies
git clone https://github.com/Tanvi51204/openEnv.git
cd openEnv
pip install -r requirements.txt

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Open Swagger UI
open http://localhost:8000/docs
```

### Local — Docker
```bash
docker build -t data-cleaning-env .
docker run -p 8000:8000 data-cleaning-env
```

### Quick API test
```bash
# Health
curl http://localhost:8000/health

# Start Task 1
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'

# Fill missing values
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"operation": "fill_missing", "column": "salary", "params": {"strategy": "median"}}'
```

### Python client
```python
from client import DataCleaningEnvClient
from models import DataCleaningAction

with DataCleaningEnvClient("http://localhost:8000") as env:
    result = env.reset(task_id=1)
    print(result.observation.missing_counts)   # {'age': 20, 'salary': 20, 'department': 10}

    action = DataCleaningAction(
        operation="fill_missing",
        column="salary",
        params={"strategy": "median"},
    )
    result = env.step(action)
    print(result.observation.current_score)    # 0.4
    print(result.reward)                       # 0.4
```

### Run baseline inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."          # your API key
export ENV_URL="http://localhost:8000"

python inference.py
```

Produces `[START]` / `[STEP]` / `[END]` lines to stdout and `baseline_scores.json`.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | — | API key for LLM calls |
| `ENV_URL` | `http://localhost:8000` | Environment server URL |

---

## Project Structure
```
openenv-data-cleaning/
├── models.py              Pydantic contracts — Action / Observation / State
├── client.py              Sync HTTP client (reset / step / state / health)
├── inference.py           Baseline LLM agent with [START]/[STEP]/[END] logging
├── openenv.yaml           OpenEnv manifest
├── Dockerfile             python:3.11-slim, non-root user, HEALTHCHECK
├── requirements.txt       pip dependencies
├── pyproject.toml         Python package metadata + openenv-core dependency
└── server/
    ├── app.py             FastAPI routes + /metadata + /schema
    ├── environment.py     reset / step / state logic + 6 operations + rewards
    ├── data_generator.py  Synthetic dataset generation (seed=42, reproducible)
    └── tasks/
        ├── task1_missing.py    Easy  — fill NaN grader
        ├── task2_format.py     Medium — format + duplicates grader
        └── task3_pipeline.py   Hard  — full pipeline grader
```

---

## Live Demo

🤗 **HuggingFace Space:** https://srishtichugh-openenv-hack.hf.space

- Health: https://srishtichugh-openenv-hack.hf.space/health
- Docs:   https://srishtichugh-openenv-hack.hf.space/docs