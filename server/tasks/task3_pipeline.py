"""
Task 3 — Hard: Full Cleaning Pipeline
Objective: Fix missing values, remove duplicates, handle outliers, standardise
           country capitalisation and date formats.
Score: equal-weight average of 4 sub-scores.
"""

import re
import numpy as np
import pandas as pd
from server.data_generator import generate_task3_datasets

TASK_ID = 3
MAX_STEPS = 40
DESCRIPTION = (
    "Task 3 (Hard) — Full Cleaning Pipeline\n"
    "You have a customer database with multiple issues:\n"
    "  1. Missing values in 'age', 'purchase_amount', 'country', 'signup_date'\n"
    "  2. ~20 duplicate rows\n"
    "  3. Outliers in 'purchase_amount' (injected values ~10x normal)\n"
    "  4. Mixed case in 'country' (need: title case, e.g. 'Usa' → 'USA')\n"
    "  5. Mixed date formats in 'signup_date' (need: YYYY-MM-DD)\n\n"
    "Available operations:\n"
    "  fill_missing    — column + params.strategy ('median'|'mean'|'mode'|'constant')\n"
    "  drop_duplicates — no column needed\n"
    "  drop_outliers   — column (numeric); uses IQR method\n"
    "  fix_format      — column: 'country' | 'signup_date'\n"
    "  fix_dtype       — column + params.dtype ('float'|'int'|'str')\n\n"
    "Example actions:\n"
    '  {"operation": "fill_missing",    "column": "age",             "params": {"strategy": "median"}}\n'
    '  {"operation": "drop_duplicates"}\n'
    '  {"operation": "drop_outliers",   "column": "purchase_amount"}\n'
    '  {"operation": "fix_format",      "column": "signup_date"}\n'
    '  {"operation": "fix_format",      "column": "country"}'
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_COUNTRIES = {"USA", "UK", "Canada", "Australia", "Germany"}


def load():
    dirty, clean = generate_task3_datasets()
    orig_nulls = int(dirty.isnull().sum().sum())
    orig_dupes = len(dirty) - len(dirty.drop_duplicates())

    # Outlier baseline: count rows where purchase_amount > Q3 + 3*IQR
    pa = dirty["purchase_amount"].dropna()
    q1, q3 = pa.quantile(0.25), pa.quantile(0.75)
    iqr = q3 - q1
    orig_outliers = int((pa > q3 + 3 * iqr).sum())

    orig_country_issues = int((~dirty["country"].isin(VALID_COUNTRIES) &
                               dirty["country"].notna()).sum())
    orig_date_issues    = int((~dirty["signup_date"].apply(
        lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
    )).sum())

    meta = {
        "orig_nulls":           orig_nulls,
        "orig_dupes":           orig_dupes,
        "orig_outliers":        max(orig_outliers, 1),
        "orig_country_issues":  max(orig_country_issues, 1),
        "orig_date_issues":     max(orig_date_issues, 1),
        "q1": q1, "q3": q3, "iqr": iqr,
    }
    return dirty.copy(), clean, meta


def score(current_df, meta: dict) -> float:
    remaining_nulls = int(current_df.isnull().sum().sum())
    remaining_dupes = len(current_df) - len(current_df.drop_duplicates())

    pa = current_df["purchase_amount"].dropna()
    remaining_outliers = int((pa > meta["q3"] + 3 * meta["iqr"]).sum())

    remaining_country = int((~current_df["country"].isin(VALID_COUNTRIES) &
                              current_df["country"].notna()).sum())
    remaining_dates   = int((~current_df["signup_date"].apply(
        lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
    )).sum())

    null_score     = 1.0 - remaining_nulls    / max(meta["orig_nulls"],    1)
    dupe_score     = 1.0 - remaining_dupes    / max(meta["orig_dupes"],    1)
    outlier_score  = 1.0 - remaining_outliers / meta["orig_outliers"]
    country_score  = 1.0 - remaining_country  / meta["orig_country_issues"]
    date_score     = 1.0 - remaining_dates    / meta["orig_date_issues"]

    combined = 0.25 * null_score + 0.20 * dupe_score + 0.20 * outlier_score \
             + 0.175 * country_score + 0.175 * date_score
    return round(max(0.01, min(0.99, combined)), 4)


def count_errors(current_df, meta: dict) -> int:
    remaining_nulls = int(current_df.isnull().sum().sum())
    remaining_dupes = len(current_df) - len(current_df.drop_duplicates())
    pa = current_df["purchase_amount"].dropna()
    remaining_outliers = int((pa > meta["q3"] + 3 * meta["iqr"]).sum())
    remaining_country = int((~current_df["country"].isin(VALID_COUNTRIES) &
                              current_df["country"].notna()).sum())
    remaining_dates   = int((~current_df["signup_date"].apply(
        lambda x: bool(DATE_RE.match(str(x))) if pd.notna(x) else False
    )).sum())
    return remaining_nulls + remaining_dupes + remaining_outliers + \
           remaining_country + remaining_dates
