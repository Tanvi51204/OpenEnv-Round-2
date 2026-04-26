"""Generate OrgOS architecture diagram → assets/orgos_architecture.png"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── canvas ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor("#ffffff")

# ── palette ────────────────────────────────────────────────────────────────────
ZD_C   = "#0284c7"
JR_C   = "#7c3aed"
SF_C   = "#059669"
WD_C   = "#d97706"
AGT_C  = "#1e3a8a"
SCH_C  = "#b91c1c"
POL_C  = "#c2410c"
ENV_BG = "#f8fafc"
ENV_BD = "#94a3b8"
TEXT_D = "#0f172a"
TEXT_M = "#475569"

# ── helpers ────────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec="none", lw=1.5, r=0.12, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def txt(x, y, s, sz=11, c="white", bold=False, z=5):
    ax.text(x, y, s, ha="center", va="center",
            fontsize=sz, fontweight="bold" if bold else "normal",
            color=c, zorder=z)

def arw(x1, y1, x2, y2, color, lw=2.8, dashed=False):
    style = (0, (6, 3)) if dashed else "solid"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, linestyle=style,
                                mutation_scale=30), zorder=6)

def pill(x, y, s, color, sz=8.5):
    ax.text(x, y, s, ha="center", va="center", fontsize=sz,
            color=color, zorder=7,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color, linewidth=1.6))

# ── title ──────────────────────────────────────────────────────────────────────
txt(8.4, 10.6, "OrgOS  —  Multi-App Enterprise RL Environment",
    sz=16, c=TEXT_D, bold=True)

# ── OrgOS Environment outer box ────────────────────────────────────────────────
rbox(3.0, 2.6, 10.8, 7.6, fc=ENV_BG, ec=ENV_BD, lw=2.2, r=0.2, z=1)
txt(8.4, 9.95, "OrgOS Environment", sz=14, c=TEXT_M, bold=True)

# ── 2 × 2 app boxes ────────────────────────────────────────────────────────────
AW, AH = 4.9, 2.65

# top-left: Zendesk
rbox(3.2, 6.95, AW, AH, ZD_C, r=0.14, z=3)
txt(3.2 + AW/2, 6.95 + 1.90, "Zendesk",        sz=16, bold=True)
txt(3.2 + AW/2, 6.95 + 1.25, "Support Tickets", sz=12, c="#bae6fd")
txt(3.2 + AW/2, 6.95 + 0.52, "8 operations",    sz=10, c="#7dd3fc")

# top-right: Jira
rbox(8.55, 6.95, AW, AH, JR_C, r=0.14, z=3)
txt(8.55 + AW/2, 6.95 + 1.90, "Jira",               sz=16, bold=True)
txt(8.55 + AW/2, 6.95 + 1.25, "Engineering Issues",  sz=12, c="#ddd6fe")
txt(8.55 + AW/2, 6.95 + 0.52, "9 operations",        sz=10, c="#c4b5fd")

# bottom-left: Salesforce
rbox(3.2, 3.85, AW, AH, SF_C, r=0.14, z=3)
txt(3.2 + AW/2, 3.85 + 1.90, "Salesforce",     sz=16, bold=True)
txt(3.2 + AW/2, 3.85 + 1.25, "CRM & Accounts", sz=12, c="#a7f3d0")
txt(3.2 + AW/2, 3.85 + 0.52, "7 operations",   sz=10, c="#6ee7b7")

# bottom-right: Workday
rbox(8.55, 3.85, AW, AH, WD_C, r=0.14, z=3)
txt(8.55 + AW/2, 3.85 + 1.90, "Workday",      sz=16, bold=True)
txt(8.55 + AW/2, 3.85 + 1.25, "HR & Access",  sz=12, c="#fef3c7")
txt(8.55 + AW/2, 3.85 + 0.52, "7 operations", sz=10, c="#fde68a")

# ── AI Agent box ───────────────────────────────────────────────────────────────
rbox(4.2, 0.2, 8.0, 1.3, AGT_C, r=0.14, z=3)
txt(8.2, 1.02, "AI Agent  (Qwen 2.5-3B-Instruct)",            sz=14, bold=True)
txt(8.2, 0.50, "Reads observation  ·  Sends one action per step", sz=11, c="#93c5fd")

# ── Action arrow  (Agent → Environment) ───────────────────────────────────────
arw(6.2, 1.52, 6.2, 2.62, AGT_C, lw=3.5)
pill(4.65, 2.07, "action\n{ app, op, args }", AGT_C, sz=9.5)

# ── Observation arrow  (Environment → Agent) ───────────────────────────────────
arw(10.4, 2.62, 10.4, 1.52, SF_C, lw=3.5)
pill(12.1, 2.07, "observation\n+ reward", SF_C, sz=9.5)

# ── Schema Drift box (left) ─────────────────────────────────────────────────────
rbox(0.05, 7.5, 2.0, 1.9, "#fff1f2", ec=SCH_C, lw=1.8, r=0.12, z=3)
txt(1.05, 9.17, "Schema Drift",        sz=11,  c=SCH_C, bold=True)
txt(1.05, 8.73, "field names shift",   sz=9.5, c=SCH_C)
txt(1.05, 8.35, "every episode",       sz=9.5, c=SCH_C)
txt(1.05, 7.92, "(3 versions / app)",  sz=8.5, c="#ef4444")
arw(2.07, 8.45, 2.98, 8.45, SCH_C, lw=2.5, dashed=True)

# ── Policy Drift box (left) ─────────────────────────────────────────────────────
rbox(0.05, 5.0, 2.0, 1.9, "#fff7ed", ec=POL_C, lw=1.8, r=0.12, z=3)
txt(1.05, 6.67, "Policy Drift",         sz=11,  c=POL_C, bold=True)
txt(1.05, 6.23, "SLA rules tighten",    sz=9.5, c=POL_C)
txt(1.05, 5.85, "every 3rd episode",    sz=9.5, c=POL_C)
txt(1.05, 5.42, "(no announcement)",    sz=8.5, c="#f97316")
arw(2.07, 5.95, 2.98, 5.95, POL_C, lw=2.5, dashed=True)

# ── save ───────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
out = os.path.join(os.path.dirname(__file__), "orgos_architecture.png")
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
