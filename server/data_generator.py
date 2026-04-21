"""
Synthetic dataset generation with a fixed seed for full reproducibility.
All datasets are generated purely from numpy/random — no external downloads.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List

SEED = 42

# ---------------------------------------------------------------------------
# Shared name pools (cross-referenced across apps)
# ---------------------------------------------------------------------------

FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace",
               "Heidi", "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj",
               "Oscar", "Peggy", "Quinn", "Romeo", "Sybil", "Trent"]
LAST_NAMES  = ["Smith", "Jones", "Brown", "Taylor", "Wilson", "Davis",
               "Miller", "Anderson", "Thomas", "Jackson"]


# ---------------------------------------------------------------------------
# Task 1 — Employee records with missing values
# ---------------------------------------------------------------------------

def generate_task1_datasets():
    """Returns (dirty_df, clean_df) for Task 1."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n = 100
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]

    names      = [f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(n)]
    ages       = rng.integers(22, 60, size=n).astype(float)
    salaries   = rng.integers(40_000, 120_000, size=n).astype(float)
    depts      = rng.choice(departments, size=n)
    experience = rng.integers(0, 30, size=n).astype(float)

    clean_df = pd.DataFrame({
        "name":       names,
        "age":        ages,
        "salary":     salaries,
        "department": depts,
        "experience": experience,
    })

    dirty_df = clean_df.copy()
    for col, frac in [("age", 0.20), ("salary", 0.20), ("department", 0.10)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        dirty_df.loc[idx, col] = np.nan

    return dirty_df.reset_index(drop=True), clean_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 2 — Product catalog with format & duplicate issues
# ---------------------------------------------------------------------------

def _scramble_phone(phone: str, rng) -> str:
    digits = phone.replace("-", "")
    fmt = rng.integers(0, 3)
    if fmt == 0:
        return digits
    elif fmt == 1:
        return f"({digits[:3]}){digits[3:]}"
    else:
        return phone


def _scramble_date(date_str: str, rng) -> str:
    dt = pd.to_datetime(date_str)
    fmt = rng.integers(0, 3)
    if fmt == 0:
        return dt.strftime("%Y-%m-%d")
    elif fmt == 1:
        return dt.strftime("%b %d %Y")
    else:
        return dt.strftime("%d/%m/%Y")


def generate_task2_datasets():
    """Returns (dirty_df, clean_df) for Task 2."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n = 200
    categories = ["Electronics", "Clothing", "Food", "Books", "Toys"]

    product_ids    = [f"P{str(i).zfill(4)}" for i in range(1, n + 1)]
    product_names  = [f"Product_{i}" for i in range(1, n + 1)]
    prices         = np.round(rng.uniform(5.0, 500.0, size=n), 2)
    categories_col = rng.choice(categories, size=n)
    phones         = [
        f"{rng.integers(100,999)}-{rng.integers(100,999)}-{rng.integers(1000,9999)}"
        for _ in range(n)
    ]
    days_offset = rng.integers(0, 1000, size=n)
    dates = [
        (pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in days_offset
    ]

    clean_df = pd.DataFrame({
        "product_id":   product_ids,
        "product_name": product_names,
        "price":        prices,
        "category":     categories_col,
        "phone":        phones,
        "listed_date":  dates,
    })

    dirty_df = clean_df.copy()

    phone_idx = rng.choice(n, size=int(n * 0.6), replace=False)
    dirty_df.loc[phone_idx, "phone"] = [
        _scramble_phone(dirty_df.loc[i, "phone"], rng) for i in phone_idx
    ]

    date_idx = rng.choice(n, size=int(n * 0.6), replace=False)
    dirty_df.loc[date_idx, "listed_date"] = [
        _scramble_date(dirty_df.loc[i, "listed_date"], rng) for i in date_idx
    ]

    dup_idx  = rng.choice(n, size=15, replace=False)
    dup_rows = dirty_df.iloc[dup_idx].copy()
    dirty_df = pd.concat([dirty_df, dup_rows], ignore_index=True)

    return dirty_df.reset_index(drop=True), clean_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 3 — Customer database: full pipeline
# ---------------------------------------------------------------------------

def generate_task3_datasets():
    """Returns (dirty_df, clean_df) for Task 3."""
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    n = 300
    countries = ["USA", "UK", "Canada", "Australia", "Germany"]

    names            = [f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(n)]
    ages             = rng.integers(18, 75, size=n).astype(float)
    purchase_amounts = np.round(rng.uniform(10.0, 500.0, size=n), 2)
    countries_col    = rng.choice(countries, size=n)
    emails           = [f"user{i}@example.com" for i in range(1, n + 1)]
    days_offset      = rng.integers(0, 730, size=n)
    signup_dates     = [
        (pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in days_offset
    ]

    clean_df = pd.DataFrame({
        "name":            names,
        "age":             ages,
        "purchase_amount": purchase_amounts,
        "country":         countries_col,
        "email":           emails,
        "signup_date":     signup_dates,
    })

    dirty_df = clean_df.copy()

    for col, frac in [("age", 0.15), ("purchase_amount", 0.15),
                      ("country", 0.10), ("signup_date", 0.10)]:
        idx = rng.choice(n, size=int(n * frac), replace=False)
        dirty_df.loc[idx, col] = np.nan

    out_idx = rng.choice(n, size=int(n * 0.03), replace=False)
    dirty_df.loc[out_idx, "purchase_amount"] = (
        dirty_df.loc[out_idx, "purchase_amount"] * 10
    )

    case_idx = rng.choice(n, size=int(n * 0.40), replace=False)
    dirty_df.loc[case_idx, "country"] = dirty_df.loc[case_idx, "country"].str.lower()

    date_idx = rng.choice(n, size=int(n * 0.50), replace=False)
    valid_date_idx = [i for i in date_idx if pd.notna(dirty_df.loc[i, "signup_date"])]
    for i in valid_date_idx:
        dirty_df.loc[i, "signup_date"] = _scramble_date(dirty_df.loc[i, "signup_date"], rng)

    dup_idx  = rng.choice(n, size=20, replace=False)
    dup_rows = dirty_df.iloc[dup_idx].copy()
    dirty_df = pd.concat([dirty_df, dup_rows], ignore_index=True)

    return dirty_df.reset_index(drop=True), clean_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# OrgOS App Data Generators
# ---------------------------------------------------------------------------

def generate_jira_records(n: int = 50, seed: int = SEED) -> List[Dict]:
    """Generate synthetic Jira-like engineering tickets (canonical field names)."""
    random.seed(seed)
    priorities = ["p0", "p1", "p2", "p3"]
    statuses   = ["open", "in_progress", "in_review", "closed"]
    employees  = [f"EMP-{i:03d}" for i in range(1, 21)]
    accounts   = [f"ACME-{i:03d}" for i in range(1, 31)]
    titles = [
        "Login fails intermittently", "API timeout on checkout",
        "Dashboard charts not rendering", "Email notifications delayed",
        "Password reset broken", "Search returns no results",
        "Import fails for large files", "Session expires too quickly",
        "Reports missing data", "Webhook delivery failures",
    ]

    records = []
    for i in range(1, n + 1):
        records.append({
            "issue_id":       f"JIRA-{i:03d}",
            "title":          f"{random.choice(titles)} #{i}",
            "priority":       random.choices(priorities, weights=[5, 15, 50, 30])[0],
            "assignee":       random.choice(employees) if random.random() > 0.3 else None,
            "status":         random.choices(statuses, weights=[30, 40, 15, 15])[0],
            "reporter":       random.choice(employees),
            "customer_id":    random.choice(accounts),
            "linked_zendesk": None,
            "labels":         random.sample(["bug", "urgent", "customer-reported"], k=random.randint(0, 2)),
            "created_at":     "2026-04-20T09:00:00",
        })

    # Workflow A primary issue: JIRA-001 is unassigned, linked to ACME-001
    records[0].update({
        "title":          "Customer login fails intermittently",
        "priority":       "p1",
        "status":         "open",
        "customer_id":    "ACME-001",
        "assignee":       None,
        "linked_zendesk": None,
    })

    return records


def generate_zendesk_records(n: int = 40, seed: int = SEED) -> List[Dict]:
    """Generate synthetic Zendesk-like support tickets (canonical field names)."""
    random.seed(seed)
    urgencies = ["p0", "p1", "p2", "p3"]
    states    = ["new", "open", "pending", "resolved", "closed"]
    accounts  = [f"ACME-{i:03d}" for i in range(1, 31)]
    agents    = [f"agent{i}@company.com" for i in range(1, 6)]

    records = []
    for i in range(1, n + 1):
        records.append({
            "ticket_number": f"ZD-{i:03d}",
            "title":         f"Support request #{i}",
            "urgency":       random.choices(urgencies, weights=[3, 12, 55, 30])[0],
            "agent_email":   random.choice(agents) if random.random() > 0.4 else None,
            "state":         random.choices(states, weights=[20, 35, 20, 15, 10])[0],
            "customer_id":   random.choice(accounts),
            "channel":       random.choice(["email", "chat", "phone", "web"]),
            "created_at":    "2026-04-20T08:00:00",
            # Internal state tracking — stripped before agent sees record
            "_acknowledged": False,
            "_queried_accounts": [],
            "_profile_created": False,
        })

    # Workflow A primary: ZD-001 is unacknowledged, from ACME-001
    records[0].update({
        "title":         "Login issue — cannot access my account",
        "urgency":       "p1",
        "state":         "new",
        "customer_id":   "ACME-001",
        "_acknowledged": False,
    })

    # Workflow C: several tickets from ACME-003
    for i in [4, 11, 17]:
        if i < len(records):
            records[i]["customer_id"] = "ACME-003"

    return records


def generate_salesforce_records(n: int = 30, seed: int = SEED) -> List[Dict]:
    """Generate synthetic Salesforce-like CRM accounts (canonical field names)."""
    random.seed(seed)
    deal_stages  = ["prospect", "qualification", "negotiation", "closed_won", "closed_lost"]
    healths      = ["green", "yellow", "red"]
    territories  = ["west", "east", "central", "apac", "emea"]
    employees    = [f"EMP-{i:03d}" for i in range(1, 21)]
    companies    = [
        "Acme Corporation", "Globex Systems", "Initech Ltd", "Umbrella Corp",
        "Stark Industries", "Wayne Enterprises", "Hooli Inc", "Pied Piper",
        "Bluth Company", "Vandelay Industries",
    ]

    records = []
    for i in range(1, n + 1):
        records.append({
            "account_id":   f"ACME-{i:03d}",
            "company_name": f"{companies[(i-1) % len(companies)]} {i}",
            "deal_stage":   random.choice(deal_stages),
            "health":       random.choices(healths, weights=[60, 30, 10])[0],
            "owner":        random.choice(employees),
            "arr":          random.randint(5_000, 200_000),
            "is_paying":    random.random() > 0.3,
            "territory":    random.choice(territories),
            "industry":     random.choice(["tech", "finance", "healthcare", "retail"]),
            # Internal state tracking
            "_account_checked": False,
            "_churn_flagged":   False,
            "_team_assigned":   False,
            "_intervention_assigned": False,
        })

    # Workflow A: ACME-001 is a paying customer with yellow health
    records[0].update({
        "company_name": "Acme Corporation",
        "deal_stage":   "closed_won",
        "health":       "yellow",
        "is_paying":    True,
        "arr":          50_000,
        "territory":    "west",
    })

    # Workflow C: ACME-003 is at churn risk
    records[2].update({
        "company_name": "Globex Systems",
        "health":       "red",
        "deal_stage":   "negotiation",
        "is_paying":    True,
        "arr":          30_000,
        "_churn_flagged": False,
    })

    return records


def generate_workday_records(n: int = 20, seed: int = SEED) -> List[Dict]:
    """Generate synthetic Workday-like HR records (canonical field names)."""
    random.seed(seed)
    levels      = ["IC1", "IC2", "IC3", "IC4", "M1", "M2"]
    departments = ["engineering", "support", "sales", "hr", "data"]
    territories = ["west", "east", "central", "apac", "emea"]

    records = []
    for i in range(1, n + 1):
        records.append({
            "employee_id": f"EMP-{i:03d}",
            "name":        f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "level":       random.choice(levels),
            "manager_id":  f"EMP-{random.randint(1, min(i, 5)):03d}" if i > 1 else None,
            "status":      random.choices(["active", "pending"], weights=[90, 10])[0],
            "department":  random.choice(departments),
            "territory":   random.choice(territories),
            "email":       f"emp{i}@company.com",
            # Internal state tracking
            "_access_provisioned": {},  # app_name → bool
            "_sla_logged":         False,
            "_onboarding_created": False,
        })

    # Workflow B: one pending new hire to onboard
    records.append({
        "employee_id":         "EMP-NEW-001",
        "name":                "Jordan Riley",
        "level":               "IC2",
        "manager_id":          "EMP-001",
        "status":              "pending",
        "department":          "support",
        "territory":           "west",
        "email":               "jordan.riley@company.com",
        "_access_provisioned": {},
        "_sla_logged":         False,
        "_onboarding_created": False,
    })

    return records


def generate_episode_data(workflow_id: str, seed: int = SEED) -> Dict[str, List[Dict]]:
    """
    Generate correlated data for a full episode across all 4 apps.
    Cross-references are maintained: Zendesk customer_ids match Salesforce account_ids,
    Jira reporters are Workday employees, etc.
    """
    return {
        "jira":        generate_jira_records(n=50, seed=seed),
        "zendesk":     generate_zendesk_records(n=40, seed=seed),
        "salesforce":  generate_salesforce_records(n=30, seed=seed),
        "workday":     generate_workday_records(n=20, seed=seed),
    }
