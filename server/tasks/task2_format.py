"""
Task 2 — Medium: Fix Formats + Remove Duplicates
Objective: Standardise phone & date formats and drop duplicate rows.
Score: weighted average of format_score (0.7) + dupe_score (0.3)
"""

import re
import pandas as pd
from server.data_generator import generate_task2_datasets

TASK_ID = 2
MAX_STEPS = 30
DESCRIPTION = (
    "Task 2 (Medium) — Fix Formats and Remove Duplicates\n"
    "You have a product catalog with:\n"
    "  • Phone numbers in mixed formats (need: NNN-NNN-NNNN)\n"
    "  • Dates in mixed formats (need: YYYY-MM-DD)\n"
    "  • Duplicate rows (~15)\n\n"
    "Available operations:\n"
    "  fix_format  — column: 'phone' | 'listed_date'\n"
    "  drop_duplicates — no column needed\n\n"
    "Example actions:\n"
    '  {"operation": "fix_format", "column": "phone"}\n'
    '  {"operation": "fix_format", "column": "listed_date"}\n'
    '  {"operation": "drop_duplicates"}'
)

PHONE_RE = re.compile(r"^\d{3}-\d{3}-\d{4}$")
DATE_RE  = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load():
    dirty, clean = generate_task2_datasets()
    original_phone_issues = int((~dirty["phone"].str.match(PHONE_RE)).sum())
    original_date_issues  = int((~dirty["listed_date"].apply(
        lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
    )).sum())
    original_dupes = len(dirty) - len(dirty.drop_duplicates())
    meta = {
        "orig_phone": original_phone_issues,
        "orig_date":  original_date_issues,
        "orig_dupes": original_dupes,
    }
    return dirty.copy(), clean, meta


def score(current_df, meta: dict) -> float:
    phone_issues = int((~current_df["phone"].str.match(PHONE_RE)).sum())
    date_issues  = int((~current_df["listed_date"].apply(
        lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
    )).sum())
    dupes        = len(current_df) - len(current_df.drop_duplicates())

    phone_score = 1.0 - phone_issues / max(meta["orig_phone"], 1)
    date_score  = 1.0 - date_issues  / max(meta["orig_date"],  1)
    dupe_score  = 1.0 - dupes        / max(meta["orig_dupes"], 1)

    combined = 0.35 * phone_score + 0.35 * date_score + 0.30 * dupe_score
    return round(max(0.01, min(0.99, combined)), 4)


def count_errors(current_df, meta: dict) -> int:
    phone_issues = int((~current_df["phone"].str.match(PHONE_RE)).sum())
    date_issues  = int((~current_df["listed_date"].apply(
        lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
    )).sum())
    dupes = len(current_df) - len(current_df.drop_duplicates())
    return phone_issues + date_issues + dupes
