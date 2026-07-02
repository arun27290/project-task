"""
Azure Chat Bot (AWS Rebuild) — Program Dashboard  v2
=====================================================
Run  :  python app.py
Open :  http://localhost:5050

100% OFFLINE — no internet required.
Charts generated server-side with matplotlib (no Chart.js CDN).
No Google Fonts CDN — system fonts only. Single file, no external assets.

v2 adds: progress-over-time trend (local history file), presentation/print
mode, owner workload chart, upcoming deadlines widget, risk heat matrix,
change-frequency flag (from Change Log), and a Sync Log feed.
"""

import io, os, json, warnings, logging, re, difflib, base64
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from flask import Flask, request, render_template_string, jsonify

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

BLUE    = "#4a8cff"
RED     = "#ff4f6a"
GREEN   = "#30d988"
YELLOW  = "#ffc240"
PURPLE  = "#a78bfa"
CYAN    = "#22d3ee"
ORANGE  = "#fb923c"
GREY    = "#94a3b8"   # shared neutral — reads fine on both dark and light chart surfaces

THEMES = {
    "dark":  {"BG": "#07090f", "SURFACE": "#111827", "BORDER": "#1e2a40", "MUTED": "#7b8db0", "TEXT": "#edf2ff"},
    "light": {"BG": "#f8fafc", "SURFACE": "#ffffff", "BORDER": "#e2e8f0", "MUTED": "#64748b", "TEXT": "#0f172a"},
}

RAG_COLORS = {"Red": RED, "Amber": YELLOW, "Green": GREEN, "-": GREY}
STATUS_COLORS = {"Not Started": GREY, "In Progress": BLUE, "Completed": GREEN,
                  "Blocked": RED, "Delayed": ORANGE}
IMPACT_COLORS = {"High": RED, "Medium": YELLOW, "Low": GREEN}


def _apply_rc(theme_name):
    p = THEMES[theme_name]
    plt.rcParams.update({
        "figure.facecolor": p["BG"], "axes.facecolor": p["SURFACE"], "axes.edgecolor": p["BORDER"],
        "axes.labelcolor": p["MUTED"], "xtick.color": p["MUTED"], "ytick.color": p["MUTED"],
        "text.color": p["TEXT"], "grid.color": p["BORDER"], "grid.linewidth": 0.6,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.titlecolor": p["TEXT"], "axes.titlepad": 8,
        "legend.facecolor": p["SURFACE"], "legend.edgecolor": p["BORDER"], "legend.fontsize": 8,
    })
    return p


# Module-level defaults (dark) — chart-drawing functions below reference these
# as globals; build_theme_charts() reassigns them before each theme's render pass.
_dp = _apply_rc("dark")
BG, SURFACE, BORDER, MUTED, TEXT = _dp["BG"], _dp["SURFACE"], _dp["BORDER"], _dp["MUTED"], _dp["TEXT"]

def _flag(name):
    """Looks up a feature-toggle by name. If the line was commented out (so the
    variable no longer exists), treat it as disabled rather than raising an error."""
    return globals().get(name, False)


HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress_history.json")

# ══════════════════════════════════════════════════════════════════════════
#  DASHBOARD FEATURE TOGGLES
#  ------------------------------------------------------------------------
#  To disable any chart/widget, just comment out its line below (put a #
#  at the start of the line) or set it to False. No other code changes
#  needed — the chart won't be generated and its card won't appear.
# ══════════════════════════════════════════════════════════════════════════
ENABLE_STATUS_DONUT      = True   # Overview — "Milestones by Status" donut
ENABLE_RAG_DONUT         = True   # Overview — "RAG Health" donut
ENABLE_PHASE_BAR         = True   # Overview — "Phase-wise Progress" bar
ENABLE_OWNER_BAR         = True   # Overview — "Milestones by Owner" bar
ENABLE_UPCOMING_WIDGET   = True   # Overview — "Upcoming Deadlines" list
ENABLE_TREND_LINE        = True   # Overview — "Progress Over Time" (needs 2+ uploads)
ENABLE_SYNC_FEED         = True   # Overview — "Latest Sync Log Updates" feed
ENABLE_RISK_DONUT        = True   # Risks tab — "Open Risks by Impact" donut
ENABLE_RISK_MATRIX       = True   # Risks tab — "Risk Heat Matrix"

# Milestone Timeline — you can enable more than one; each appears as its own
# card, stacked top to bottom, so you can compare and decide which to keep.
ENABLE_TIMELINE_ROADMAP  = True   # single-line roadmap, fixed height, best for many milestones
ENABLE_TIMELINE_MONTHLY  = True   # compact stacked bar per month — always fits one page
ENABLE_TIMELINE_PHASE    = True   # one row per Phase instead of per milestone
ENABLE_TIMELINE_GANTT    = True   # original per-milestone Gantt bars (grows tall with many rows)


def load_history():
    try:
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_history_entry(entry):
    try:
        hist = load_history()
        hist = [h for h in hist if h["date"] != entry["date"]]
        hist.append(entry)
        hist = sorted(hist, key=lambda h: h["date"])[-60:]
        with open(HISTORY_PATH, "w") as f:
            json.dump(hist, f)
        return hist
    except Exception:
        log.exception("Could not persist progress history")
        return load_history()


def fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{data}"


# ── chart generators ──────────────────────────────────────────────────────────

def make_donut(labels, values, colors, title="", size=(4.2, 3.4)):
    nz = [(l, v, c) for l, v, c in zip(labels, values, colors) if v and v > 0]
    if not nz:
        return None
    legend_patches = [mpatches.Patch(color=c, label=f"{l} ({v})") for l, v, c in zip(labels, values, colors) if v is not None]
    nz_labels, nz_values, nz_cols = zip(*nz)
    fig, ax = plt.subplots(figsize=size)
    ax.pie(nz_values, colors=nz_cols, startangle=90, wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
    ax.text(0, 0, str(sum(values)), ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.set_title(title, color=TEXT, pad=6)
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0, fontsize=7.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_hbar(labels, values, colors=None, title="", xlabel="", unit="%", size=(6.2, None)):
    if not labels:
        return None
    n = len(labels)
    h = max(2.2, n * 0.42)
    fig, ax = plt.subplots(figsize=(size[0], h))
    y = range(n)
    cols = colors if colors else [BLUE] * n
    bars = ax.barh(list(y), values, color=cols, height=0.6, edgecolor=BG, linewidth=0.5)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="x", alpha=0.4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    maxv = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + maxv * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val}{unit}", va="center", fontsize=7.5, color=TEXT)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_timeline_gantt(rows, title="Milestone Timeline — Gantt", size=(9, None)):
    rows = [r for r in rows if r.get("target_date") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["target_date"])
    n = len(rows)
    h = max(3, n * 0.42)
    fig, ax = plt.subplots(figsize=(size[0], h))
    for i, r in enumerate(rows):
        target = r["target_date"]
        start = r.get("start_date") or (target - timedelta(days=10))
        color = RAG_COLORS.get(r.get("rag", "-"), GREY)
        width_days = max((target - start).days, 1)
        ax.barh(i, width_days, left=mdates.date2num(start), height=0.5,
                color=color, edgecolor=BG, linewidth=0.5, alpha=0.9)
        actual = r.get("actual_date")
        if actual:
            ax.scatter([mdates.date2num(actual)], [i], color=TEXT, s=26, zorder=5, marker="D")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["milestone"][:42] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    fig.autofmt_xdate(rotation=30)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right", "left"]].set_visible(False)
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in
                       [("On Track", GREEN), ("Due Soon / Late", YELLOW), ("Overdue / Blocked", RED)]]
    legend_patches.append(mpatches.Patch(facecolor="none", edgecolor="none", label="◆ = Actual completion"))
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0, -0.05 - 0.6 / h),
              ncol=2, framealpha=0, fontsize=7.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_timeline_roadmap(rows, title="Milestone Timeline — Roadmap", size=(11, 4.6)):
    """Single-line roadmap: one dot per milestone along a date axis, labels staggered
    above/below so the chart height never grows with milestone count."""
    rows = [r for r in rows if r.get("target_date") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["target_date"])
    n = len(rows)
    width = max(size[0], n * 0.85)
    fig, ax = plt.subplots(figsize=(width, size[1]))
    xs = [mdates.date2num(r["target_date"]) for r in rows]
    ax.axhline(0, color=BORDER, linewidth=1.4, zorder=1)
    levels = [1.6, -1.6, 2.9, -2.9]  # alternate stagger distances to reduce label overlap
    for i, (r, x) in enumerate(zip(rows, xs)):
        color = RAG_COLORS.get(r.get("rag", "-"), GREY)
        lvl = levels[i % len(levels)]
        ax.scatter([x], [0], s=90, color=color, edgecolor=BG, linewidth=1.4, zorder=5)
        ax.plot([x, x], [0, lvl * 0.78], color=BORDER, linewidth=0.8, zorder=2)
        va = "bottom" if lvl > 0 else "top"
        label = r["milestone"][:30] + ("…" if len(r["milestone"]) > 30 else "")
        date_lbl = r["target_date"].strftime("%d-%b")
        ax.text(x, lvl, f"{label}\n{date_lbl}", ha="center", va=va, fontsize=7.3, color=TEXT, linespacing=1.5)
    ax.set_ylim(-3.6, 3.6)
    ax.set_yticks([])
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y"))
    ax.set_title(title, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in
                       [("On Track", GREEN), ("Due Soon / Late", YELLOW), ("Overdue / Blocked", RED)]]
    ax.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, framealpha=0, fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_timeline_monthly(rows, title="Milestone Timeline — By Month", size=(9, 3.6)):
    """Compact fixed-height view: one stacked bar per month, coloured by RAG,
    so the chart never grows taller regardless of how many milestones exist."""
    rows = [r for r in rows if r.get("target_date") is not None]
    if not rows:
        return None
    buckets = {}
    for r in rows:
        key = r["target_date"].strftime("%Y-%m")
        buckets.setdefault(key, {"Green": 0, "Amber": 0, "Red": 0, "-": 0})
        buckets[key][r.get("rag", "-")] = buckets[key].get(r.get("rag", "-"), 0) + 1
    month_keys = sorted(buckets.keys())
    month_labels = [datetime.strptime(k, "%Y-%m").strftime("%b-%y") for k in month_keys]
    fig, ax = plt.subplots(figsize=size)
    bottoms = [0] * len(month_keys)
    for rag_key, color, label in [("Green", GREEN, "On Track"), ("Amber", YELLOW, "Due Soon / Late"),
                                   ("Red", RED, "Overdue / Blocked"), ("-", GREY, "No Target")]:
        vals = [buckets[k][rag_key] for k in month_keys]
        if sum(vals) == 0:
            continue
        ax.bar(month_labels, vals, bottom=bottoms, color=color, label=label, edgecolor=BG, linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title(title, color=TEXT)
    ax.set_ylabel("Milestones Due", color=MUTED)
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(loc="upper left", framealpha=0, fontsize=7.5, ncol=2)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_timeline_phase(rows, title="Milestone Timeline — By Phase", size=(9, None)):
    """One row per Phase (instead of per milestone) — bar spans that phase's earliest
    to latest target date, coloured by its worst RAG status, with a count badge."""
    rows = [r for r in rows if r.get("target_date") is not None]
    if not rows:
        return None
    phases = {}
    for r in rows:
        p = r.get("phase") or "Unassigned"
        phases.setdefault(p, []).append(r)
    if len(phases) <= 1:
        return None
    sev_order = {"Red": 3, "Amber": 2, "Green": 1, "-": 0}
    phase_rows = []
    for p, items in phases.items():
        targets = [it["target_date"] for it in items]
        starts = [it.get("start_date") or (it["target_date"] - timedelta(days=10)) for it in items]
        worst_rag = max((it.get("rag", "-") for it in items), key=lambda x: sev_order.get(x, 0))
        phase_rows.append({"phase": p, "start": min(starts), "end": max(targets),
                            "rag": worst_rag, "count": len(items)})
    phase_rows.sort(key=lambda x: x["start"])
    n = len(phase_rows)
    h = max(3, n * 0.55)
    fig, ax = plt.subplots(figsize=(size[0], h))
    for i, pr in enumerate(phase_rows):
        color = RAG_COLORS.get(pr["rag"], GREY)
        width_days = max((pr["end"] - pr["start"]).days, 1)
        ax.barh(i, width_days, left=mdates.date2num(pr["start"]), height=0.5,
                color=color, edgecolor=BG, linewidth=0.5, alpha=0.9)
        ax.text(mdates.date2num(pr["end"]) + 1, i, f'{pr["count"]} milestone{"s" if pr["count"]!=1 else ""}',
                va="center", fontsize=7.5, color=TEXT)
    ax.set_yticks(range(n))
    ax.set_yticklabels([pr["phase"][:32] for pr in phase_rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    fig.autofmt_xdate(rotation=30)
    ax.set_title(title, color=TEXT)
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right", "left"]].set_visible(False)
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in
                       [("On Track", GREEN), ("Due Soon / Late", YELLOW), ("Overdue / Blocked", RED)]]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0, -0.06 - 0.5 / h),
              ncol=3, framealpha=0, fontsize=7.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_risk_matrix(risk_rows, title="Risk Heat Matrix (Impact x Likelihood)", size=(5.6, 4.6)):
    order = {"Low": 1, "Medium": 2, "High": 3}
    grid = {}
    for r in risk_rows:
        imp = order.get(str(r.get("impact", "")).strip().title())
        like = order.get(str(r.get("likelihood", "")).strip().title())
        if imp is None or like is None:
            continue
        grid[(like, imp)] = grid.get((like, imp), 0) + 1
    if not grid:
        return None
    fig, ax = plt.subplots(figsize=size)
    for li in (1, 2, 3):
        for ii in (1, 2, 3):
            sev = li * ii
            color = RED if sev >= 6 else (YELLOW if sev >= 3 else GREEN)
            ax.add_patch(plt.Rectangle((li - 0.5, ii - 0.5), 1, 1, facecolor=color, alpha=0.10,
                                        edgecolor=BORDER, linewidth=0.6))
    for (li, ii), cnt in grid.items():
        sev = li * ii
        color = RED if sev >= 6 else (YELLOW if sev >= 3 else GREEN)
        ax.scatter([li], [ii], s=280 + cnt * 160, color=color, edgecolor=BG, linewidth=1.5, zorder=5, alpha=0.92)
        ax.text(li, ii, str(cnt), ha="center", va="center", fontsize=11, fontweight="bold", color=BG, zorder=6)
    ax.set_xlim(0.4, 3.6); ax.set_ylim(0.4, 3.6)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Likelihood", color=MUTED); ax.set_ylabel("Impact", color=MUTED)
    ax.set_title(title, color=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_trend_line(history, size=(9, 3.2)):
    if not history:
        return None
    dates = [h["date"] for h in history]
    pct = [h["overall_pct"] for h in history]
    health = [h["health"] for h in history]
    fig, ax = plt.subplots(figsize=size)
    ax.plot(dates, pct, color=CYAN, marker="o", linewidth=2, markersize=5, label="Overall % Complete")
    ax.plot(dates, health, color=PURPLE, marker="s", linewidth=2, markersize=5, label="Program Health")
    ax.set_ylim(0, 105)
    ax.set_title("Progress Over Time (across uploads)", color=TEXT)
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", framealpha=0, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── column detection helpers ──────────────────────────────────────────────────

ALIASES = {
    "sn": ["sn", "s.no", "serial number", "#"],
    "milestone": ["milestone", "milestones", "activity", "task", "task/item name", "name of activity"],
    "phase": ["phase", "category"],
    "task_owner": ["task owner", "owner"],
    "customer_owner": ["customer owner", "client owner"],
    "priority": ["priority"],
    "start_date": ["start date", "initiated", "initiation date"],
    "target_date": ["target date", "target", "due date"],
    "actual_date": ["actual date", "actual", "completion date"],
    "status": ["status"],
    "pct_complete": ["% complete", "percent complete", "progress"],
    "reason": ["reason for delay", "reason", "delay reason"],
    "remarks": ["remarks / next steps", "remarks", "next steps"],
    "rag": ["rag status", "rag"],
}
RISK_ALIASES = {
    "id": ["id"], "type": ["type"], "description": ["description"],
    "related_milestone": ["related milestone", "milestone"],
    "impact": ["impact"], "likelihood": ["likelihood"],
    "mitigation": ["mitigation / action plan", "mitigation"],
    "owner": ["owner"], "status": ["status"],
    "date_raised": ["date raised"], "date_resolved": ["date resolved"],
}
STAKE_ALIASES = {
    "name": ["name"], "role": ["role / title", "role"], "org": ["organization", "organisation"],
    "raci": ["raci"], "email": ["email"], "notes": ["notes"],
}
CHANGE_ALIASES = {
    "date": ["date"], "milestone": ["milestone"], "field": ["field changed", "field"],
    "old": ["old value"], "new": ["new value"], "reason": ["reason"], "changed_by": ["changed by"],
}
SYNC_ALIASES = {
    "date": ["date"], "milestone": ["milestone / workstream", "milestone", "workstream"],
    "owner": ["update owner", "owner"], "update": ["key update", "update"],
    "blockers": ["blockers / issues", "blockers"], "next_steps": ["next steps"],
    "next_sync": ["next sync date"],
}


def norm(h):
    return re.sub(r"\s+", " ", str(h).strip().lower())


def detect_columns(columns, alias_map):
    normed = {norm(c): c for c in columns}
    out = {}
    for field, aliases in alias_map.items():
        found = None
        for a in aliases:
            if a in normed:
                found = normed[a]; break
        if not found:
            match = difflib.get_close_matches(aliases[0], list(normed.keys()), n=1, cutoff=0.75)
            if match:
                found = normed[match[0]]
        out[field] = found
    return out


def to_date(v):
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, datetime):
        return v
    try:
        return pd.to_datetime(v).to_pydatetime()
    except Exception:
        return None


def to_num(v, default=None):
    try:
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def safe_str(row, key):
    if not key:
        return ""
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def compute_rag(status, target, actual, today):
    if str(status).strip().lower() == "blocked":
        return "Red"
    if actual is not None:
        if target is None or actual <= target:
            return "Green"
        return "Amber"
    if target is None:
        return "-"
    days = (target - today).days
    if days < 0:
        return "Red"
    if days <= 7:
        return "Amber"
    return "Green"


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse(sheets):
    today = datetime.now()

    tracker_df = None
    for name, df in sheets.items():
        if "tracker" in name.lower() or tracker_df is None:
            tracker_df = df
            if "tracker" in name.lower():
                break
    if tracker_df is None or tracker_df.empty:
        raise ValueError("No milestone data found in the uploaded file.")

    cols = detect_columns(tracker_df.columns, ALIASES)
    if not cols.get("milestone"):
        raise ValueError("Could not find a Milestone column. Check your headers match the template.")

    milestones = []
    for _, row in tracker_df.iterrows():
        name = row.get(cols["milestone"]) if cols["milestone"] else None
        if pd.isna(name) or str(name).strip() == "":
            continue
        target = to_date(row.get(cols["target_date"])) if cols["target_date"] else None
        actual = to_date(row.get(cols["actual_date"])) if cols["actual_date"] else None
        start = to_date(row.get(cols["start_date"])) if cols["start_date"] else None
        status = safe_str(row, cols["status"]) or "Not Started"
        pct = to_num(row.get(cols["pct_complete"]) if cols["pct_complete"] else None)
        rag_raw = safe_str(row, cols["rag"])
        rag = rag_raw if rag_raw in RAG_COLORS else None
        if not rag:
            rag = compute_rag(status, target, actual, today)
        milestones.append({
            "milestone": str(name).strip(),
            "phase": safe_str(row, cols["phase"]),
            "task_owner": safe_str(row, cols["task_owner"]),
            "customer_owner": safe_str(row, cols["customer_owner"]),
            "priority": safe_str(row, cols["priority"]),
            "start_date": start, "target_date": target, "actual_date": actual,
            "status": status,
            "pct_complete": pct if pct is not None else (100 if status == "Completed" else 0),
            "reason": safe_str(row, cols["reason"]),
            "remarks": safe_str(row, cols["remarks"]),
            "rag": rag,
            "date_changes": 0,
        })

    total = len(milestones)
    if total == 0:
        raise ValueError("No milestone rows with a name were found.")

    status_counts = {}
    for s in ["Not Started", "In Progress", "Completed", "Blocked", "Delayed"]:
        status_counts[s] = sum(1 for m in milestones if m["status"] == s)
    rag_counts = {"Green": 0, "Amber": 0, "Red": 0, "-": 0}
    for m in milestones:
        rag_counts[m["rag"]] = rag_counts.get(m["rag"], 0) + 1

    overdue = sum(1 for m in milestones if m["rag"] == "Red" and m["status"] != "Completed")
    due_soon = sum(1 for m in milestones if m["rag"] == "Amber" and m["actual_date"] is None)
    completed = status_counts["Completed"]
    blocked = status_counts["Blocked"]
    overall_pct = round(np.mean([m["pct_complete"] for m in milestones]), 0) if milestones else 0

    phase_data = {}
    for m in milestones:
        p = m["phase"] or "Unassigned"
        phase_data.setdefault(p, []).append(m["pct_complete"])
    phase_labels = list(phase_data.keys())
    phase_values = [round(np.mean(v), 0) for v in phase_data.values()]
    phase_colors = [GREEN if v >= 70 else (YELLOW if v >= 35 else RED) for v in phase_values]

    # ── risks ──
    risk_rows, risk_summary = [], {"total": 0, "open": 0, "high_open": 0}
    for name, df in sheets.items():
        if "risk" in name.lower() or "issue" in name.lower():
            rc = detect_columns(df.columns, RISK_ALIASES)
            if not rc.get("description"):
                continue
            for _, row in df.iterrows():
                desc = safe_str(row, rc["description"])
                if not desc:
                    continue
                status_r = safe_str(row, rc["status"])
                impact = safe_str(row, rc["impact"])
                risk_rows.append({
                    "type": safe_str(row, rc["type"]), "description": desc,
                    "milestone": safe_str(row, rc["related_milestone"]),
                    "impact": impact, "likelihood": safe_str(row, rc["likelihood"]),
                    "mitigation": safe_str(row, rc["mitigation"]), "owner": safe_str(row, rc["owner"]),
                    "status": status_r,
                })
                risk_summary["total"] += 1
                if status_r.lower() == "open":
                    risk_summary["open"] += 1
                    if impact.lower() == "high":
                        risk_summary["high_open"] += 1
            break

    # ── stakeholders ──
    stakeholder_rows = []
    for name, df in sheets.items():
        if "stakeholder" in name.lower():
            sc = detect_columns(df.columns, STAKE_ALIASES)
            if not sc.get("name") and not sc.get("role"):
                continue
            for _, row in df.iterrows():
                nm, role = safe_str(row, sc["name"]), safe_str(row, sc["role"])
                if not nm and not role:
                    continue
                stakeholder_rows.append({
                    "name": nm, "role": role, "org": safe_str(row, sc["org"]),
                    "raci": safe_str(row, sc["raci"]), "email": safe_str(row, sc["email"]),
                })
            break

    # ── change log: count Target Date revisions per milestone ──
    change_counts = {}
    for name, df in sheets.items():
        if "change" in name.lower():
            cc = detect_columns(df.columns, CHANGE_ALIASES)
            if not cc.get("milestone") or not cc.get("field"):
                continue
            for _, row in df.iterrows():
                mile = safe_str(row, cc["milestone"])
                field = safe_str(row, cc["field"])
                if mile and "target" in field.lower():
                    key = mile.strip().lower()
                    change_counts[key] = change_counts.get(key, 0) + 1
            break
    for m in milestones:
        m["date_changes"] = change_counts.get(m["milestone"].strip().lower(), 0)

    # ── sync log feed (most recent entries) ──
    sync_entries = []
    if _flag("ENABLE_SYNC_FEED"):
        for name, df in sheets.items():
            if "sync" in name.lower():
                sc = detect_columns(df.columns, SYNC_ALIASES)
                if not sc.get("update") and not sc.get("blockers"):
                    continue
                for _, row in df.iterrows():
                    upd = safe_str(row, sc["update"])
                    blk = safe_str(row, sc["blockers"])
                    if not upd and not blk:
                        continue
                    d = to_date(row.get(sc["date"])) if sc["date"] else None
                    sync_entries.append({
                        "date": d, "milestone": safe_str(row, sc["milestone"]),
                        "owner": safe_str(row, sc["owner"]), "update": upd, "blockers": blk,
                        "next_steps": safe_str(row, sc["next_steps"]),
                    })
                break
        sync_entries.sort(key=lambda e: e["date"] or datetime.min, reverse=True)
        sync_entries = sync_entries[:6]

    # ── owner workload ──
    owner_counts = {}
    for m in milestones:
        o = m["task_owner"] or "Unassigned"
        owner_counts[o] = owner_counts.get(o, 0) + 1
    owner_labels = list(owner_counts.keys())
    owner_values = list(owner_counts.values())

    # ── upcoming deadlines (top 5, open items only) ──
    upcoming_out = []
    if _flag("ENABLE_UPCOMING_WIDGET"):
        open_with_target = [m for m in milestones if m["target_date"] and not m["actual_date"] and m["status"] != "Completed"]
        upcoming = sorted(open_with_target, key=lambda m: m["target_date"])[:5]
        upcoming_out = [{
            "milestone": m["milestone"], "owner": m["task_owner"],
            "target_date": m["target_date"].strftime("%d-%b-%Y"),
            "days_left": (m["target_date"] - today).days, "rag": m["rag"],
        } for m in upcoming]

    # ── health score ──
    health = 100
    health -= min(overdue * 8, 40)
    health -= min(due_soon * 3, 15)
    health -= min(blocked * 6, 18)
    health -= min(risk_summary["high_open"] * 6, 18)
    health = int(max(0, min(100, health)))

    # ── progress history (persisted locally, across uploads) ──
    today_key = today.strftime("%Y-%m-%d")
    history = save_history_entry({"date": today_key, "overall_pct": int(overall_pct), "health": health})

    # ── charts (generated twice: once per theme) ──
    def build_theme_charts(theme_name):
        global BG, SURFACE, BORDER, MUTED, TEXT
        p = _apply_rc(theme_name)
        BG, SURFACE, BORDER, MUTED, TEXT = p["BG"], p["SURFACE"], p["BORDER"], p["MUTED"], p["TEXT"]

        charts = {}
        charts["status_donut"] = make_donut(list(status_counts.keys()), list(status_counts.values()),
                                             [STATUS_COLORS[s] for s in status_counts], title="Milestones by Status") \
            if _flag("ENABLE_STATUS_DONUT") else None
        charts["rag_donut"] = make_donut(["On Track", "Due Soon / Late", "Overdue / Blocked"],
                                          [rag_counts.get("Green", 0), rag_counts.get("Amber", 0), rag_counts.get("Red", 0)],
                                          [GREEN, YELLOW, RED], title="RAG Health") \
            if _flag("ENABLE_RAG_DONUT") else None
        charts["phase_bar"] = make_hbar(phase_labels, phase_values, phase_colors, title="Phase-wise Progress",
                                         xlabel="% Complete") \
            if _flag("ENABLE_PHASE_BAR") and (len(phase_labels) > 1 or (phase_labels and phase_labels[0] != "Unassigned")) else None
        charts["owner_bar"] = make_hbar(owner_labels, owner_values, [BLUE] * len(owner_labels),
                                         title="Milestones by Owner", xlabel="Milestones", unit="") \
            if _flag("ENABLE_OWNER_BAR") and (owner_labels and not (len(owner_labels) == 1 and owner_labels[0] == "Unassigned")) else None
        charts["trend_line"] = make_trend_line(history) \
            if _flag("ENABLE_TREND_LINE") and len(history) >= 2 else None
        charts["risk_donut"] = None
        charts["risk_matrix"] = None
        if risk_rows:
            imp_counts = {"High": 0, "Medium": 0, "Low": 0}
            for r in risk_rows:
                if r["impact"] in imp_counts:
                    imp_counts[r["impact"]] += 1
            if _flag("ENABLE_RISK_DONUT"):
                charts["risk_donut"] = make_donut(list(imp_counts.keys()), list(imp_counts.values()),
                                                   [IMPACT_COLORS[k] for k in imp_counts], title="Open Risks by Impact")
            if _flag("ENABLE_RISK_MATRIX"):
                charts["risk_matrix"] = make_risk_matrix(risk_rows)

        # Milestone Timeline — up to four independent views; enable any combination
        charts["timeline_roadmap"] = make_timeline_roadmap(milestones) if _flag("ENABLE_TIMELINE_ROADMAP") else None
        charts["timeline_monthly"] = make_timeline_monthly(milestones) if _flag("ENABLE_TIMELINE_MONTHLY") else None
        charts["timeline_phase"] = make_timeline_phase(milestones) if _flag("ENABLE_TIMELINE_PHASE") else None
        charts["timeline_gantt"] = make_timeline_gantt(milestones) if _flag("ENABLE_TIMELINE_GANTT") else None
        return charts

    charts_dark = build_theme_charts("dark")
    charts_light = build_theme_charts("light")

    def fmt(d):
        return d.strftime("%d-%b-%Y") if d else None

    milestones_out = [{
        **{k: v for k, v in m.items() if k not in ("start_date", "target_date", "actual_date")},
        "start_date": fmt(m["start_date"]), "target_date": fmt(m["target_date"]), "actual_date": fmt(m["actual_date"]),
    } for m in milestones]

    sync_entries_out = [{
        "date": e["date"].strftime("%d-%b-%Y") if e["date"] else "—",
        "milestone": e["milestone"], "owner": e["owner"], "update": e["update"],
        "blockers": e["blockers"], "next_steps": e["next_steps"],
    } for e in sync_entries]

    return {
        "total": total, "completed": completed, "overdue": overdue, "due_soon": due_soon,
        "blocked": blocked, "overall_pct": int(overall_pct), "health": health,
        "status_counts": status_counts, "rag_counts": rag_counts,
        "phase_labels": phase_labels, "phase_values": phase_values,
        "charts_dark": charts_dark, "charts_light": charts_light, "milestones": milestones_out,
        "risks": risk_rows, "risk_summary": risk_summary,
        "stakeholders": stakeholder_rows,
        "upcoming": upcoming_out, "sync_entries": sync_entries_out,
        "trend_points": len(history),
        "flags": {
            "upcoming": _flag("ENABLE_UPCOMING_WIDGET"),
            "sync_feed": _flag("ENABLE_SYNC_FEED"),
            "trend": _flag("ENABLE_TREND_LINE"),
        },
        "detected_cols": [c for c in cols.values() if c],
        "sheet_names": list(sheets.keys()),
        "generated_at": today.strftime("%d-%b-%Y %H:%M"),
    }


# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large — maximum 50 MB."}), 413


@app.errorhandler(Exception)
def handle_exc(e):
    log.exception("Unhandled")
    return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    log.info("Upload from %s", request.remote_addr)
    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "No file selected."}), 400
    fname = f.filename.lower().strip()
    try:
        fb = io.BytesIO(f.read())
    except Exception as e:
        return jsonify({"error": f"Could not receive file: {e}"}), 400

    sheets = {}
    try:
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            xls = pd.read_excel(fb, engine="openpyxl", sheet_name=None)
            for name, df in xls.items():
                if not df.empty:
                    sheets[name] = df
        elif fname.endswith(".csv"):
            sheets["Sheet1"] = pd.read_csv(fb)
        else:
            return jsonify({"error": f"Unsupported type '{f.filename}'. Use .xlsx, .xls, or .csv"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400

    if not sheets:
        return jsonify({"error": "File is empty or has no readable data."}), 400

    try:
        result = analyse(sheets)
        log.info("Analysed OK — %d milestones, health:%d", result["total"], result["health"])
    except Exception as e:
        log.exception("Analysis failed")
        return jsonify({"error": f"Analysis failed: {e}"}), 500

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE — zero external dependencies, 100% offline
# ─────────────────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Azure Chat Bot — Program Dashboard</title>
<style>
:root,[data-theme="dark"]{
  --bg:#07090f;--surf:#0d111e;--card:#111827;--card2:#161e30;
  --border:#1e2a40;--border2:#263047;
  --blue:#4a8cff;--red:#ff4f6a;--green:#30d988;--yellow:#ffc240;
  --purple:#a78bfa;--cyan:#22d3ee;--orange:#fb923c;
  --text:#edf2ff;--muted:#7b8db0;--dim:#3a4b6b;
}
[data-theme="light"]{
  --bg:#f8fafc;--surf:#ffffff;--card:#ffffff;--card2:#f1f5f9;
  --border:#e2e8f0;--border2:#cbd5e1;
  --blue:#2563eb;--red:#dc2626;--green:#16a34a;--yellow:#d97706;
  --purple:#7c3aed;--cyan:#0891b2;--orange:#ea580c;
  --text:#0f172a;--muted:#64748b;--dim:#94a3b8;
}
[data-theme="light"] .hdr{background:rgba(248,250,252,.96)}
[data-theme="light"] .tabs{background:var(--surf)}
[data-theme="light"] .cc,[data-theme="light"] .kpi,[data-theme="light"] .ucard{
  background:var(--card);border-color:var(--border);box-shadow:0 1px 3px rgba(15,23,42,.06)}
[data-theme="light"] .drop{background:var(--card2);border-color:var(--border2)}
[data-theme="light"] .drop:hover,[data-theme="light"] .drop.drag{background:rgba(37,99,235,.06);border-color:var(--blue)}
[data-theme="light"] .new-btn,[data-theme="light"] .print-btn,[data-theme="light"] #theme-btn{background:var(--card);border-color:var(--border)}
[data-theme="light"] thead th{background:var(--surf)}
[data-theme="light"] tbody tr:hover td{background:rgba(37,99,235,.04)}
[data-theme="light"] .search-box{background:var(--card);border-color:var(--border2);color:var(--text)}
[data-theme="light"] .tbl-toolbar{background:var(--surf);border-color:var(--border)}
[data-theme="light"] .mbar-bg{background:var(--border)}
[data-theme="light"] .health-pill{background:var(--card);border-color:var(--border)}
[data-theme="light"] #upload-section{background:radial-gradient(ellipse 70% 50% at 50% 30%,rgba(37,99,235,.06),transparent 70%)}
[data-theme="light"] .offline-badge{background:rgba(22,163,74,.1);border-color:rgba(22,163,74,.3);color:var(--green)}
[data-theme="light"] .fmt{background:var(--surf);border-color:var(--border);color:var(--muted)}
[data-theme="light"] .filter-chip{background:var(--card);border-color:var(--border2);color:var(--muted)}
[data-theme="light"] .filter-chip.on{background:var(--blue);border-color:var(--blue);color:#fff}
[data-theme="light"] .tab:hover{color:var(--text)}
[data-theme="light"] .tab.on{color:var(--blue);border-bottom-color:var(--blue)}
[data-theme="light"] .empty{color:var(--dim)}
[data-theme="light"] .chip.c-green{background:rgba(22,163,74,.12);color:var(--green)}
[data-theme="light"] .chip.c-red{background:rgba(220,38,38,.12);color:var(--red)}
[data-theme="light"] .chip.c-yellow{background:rgba(217,119,6,.14);color:var(--yellow)}
[data-theme="light"] .chip.c-blue{background:rgba(37,99,235,.1);color:var(--blue)}
[data-theme="light"] .chip.c-grey{background:rgba(100,116,139,.12);color:var(--muted)}
[data-theme="light"] .chip.c-orange{background:rgba(234,88,12,.12);color:var(--orange)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,'Segoe UI',sans-serif;min-height:100vh}
.hdr{position:sticky;top:0;z-index:500;background:rgba(7,9,15,.96);border-bottom:1px solid var(--border);
  padding:12px 26px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.brand{display:flex;align-items:center;gap:10px}
.brand-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--blue),var(--purple));
  display:flex;align-items:center;justify-content:center;font-size:15px}
.brand-name{font-size:.92rem;font-weight:800;letter-spacing:-.02em}
.brand-name span{color:var(--blue)}
.hdr-right{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.pill{background:var(--card);border:1px solid var(--border);padding:3px 10px;border-radius:18px;font-size:.65rem;color:var(--muted);font-family:monospace}
.pill.live{border-color:var(--green);color:var(--green)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green);display:inline-block;margin-right:4px;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.health-pill{display:flex;align-items:center;gap:5px;background:var(--card);border:1px solid var(--border);padding:3px 10px;border-radius:18px;font-size:.67rem;font-weight:700}
.print-btn{background:var(--card);border:1px solid var(--border);color:var(--muted);padding:4px 12px;border-radius:14px;
  font-size:.68rem;font-weight:700;cursor:pointer;font-family:inherit;transition:all .15s}
.print-btn:hover{border-color:var(--blue);color:var(--blue)}
#upload-section{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:calc(100vh - 58px);padding:40px 20px;
  background:radial-gradient(ellipse 70% 50% at 50% 30%,rgba(74,140,255,.07),transparent 70%)}
.ucard{width:100%;max-width:580px;background:var(--card);border:1px solid var(--border);border-radius:18px;padding:40px 36px;text-align:center;position:relative;overflow:hidden}
.ucard::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--blue),var(--purple),var(--cyan))}
.u-title{font-size:1.5rem;font-weight:800;letter-spacing:-.03em;margin-bottom:6px}
.u-title span{color:var(--blue)}
.u-sub{color:var(--muted);font-size:.83rem;margin-bottom:24px;line-height:1.65}
.offline-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(48,217,136,.1);border:1px solid rgba(48,217,136,.3);color:var(--green);padding:4px 12px;border-radius:20px;font-size:.72rem;font-weight:700;margin-bottom:18px}
.drop{border:2px dashed var(--border2);border-radius:12px;padding:34px 20px;cursor:pointer;transition:all .2s;position:relative;background:var(--card2)}
.drop:hover,.drop.drag{border-color:var(--blue);background:rgba(74,140,255,.06)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.fmts{margin-top:8px;display:flex;gap:6px;justify-content:center}
.fmt{background:var(--surf);border:1px solid var(--border);padding:2px 9px;border-radius:5px;font-size:.65rem;font-family:monospace;color:var(--muted)}
.ubtn{margin-top:18px;background:linear-gradient(135deg,var(--blue),var(--purple));border:none;color:#fff;padding:12px 32px;border-radius:11px;font-size:.9rem;font-weight:700;cursor:pointer;font-family:inherit;transition:opacity .2s,transform .15s;width:100%}
.ubtn:hover{opacity:.9;transform:translateY(-1px)}
.ubtn:disabled{opacity:.38;cursor:not-allowed;transform:none}
.fname{margin-top:9px;font-size:.75rem;color:var(--green);font-family:monospace}
.alert-box{background:rgba(255,79,106,.1);border:1px solid rgba(255,79,106,.3);border-radius:9px;padding:11px 14px;color:var(--red);font-size:.82rem;margin-top:10px}
#spin{display:none;position:fixed;inset:0;background:rgba(7,9,15,.88);z-index:900;align-items:center;justify-content:center;flex-direction:column;gap:12px}
#spin.show{display:flex}
.spinner{width:44px;height:44px;border:3px solid var(--border2);border-top-color:var(--blue);border-radius:50%;animation:rot .8s linear infinite}
.spin-sub{color:var(--muted);font-size:.82rem;margin-top:4px}
@keyframes rot{to{transform:rotate(360deg)}}
#dash{display:none}#dash.show{display:block}
.new-btn{display:inline-flex;align-items:center;gap:7px;background:var(--card);border:1px solid var(--border);color:var(--muted);padding:8px 16px;border-radius:9px;cursor:pointer;font-family:inherit;font-size:.78rem;font-weight:600;transition:all .2s;margin:14px 26px 0}
.new-btn:hover{border-color:var(--blue);color:var(--blue)}
.tabs{background:var(--surf);border-bottom:1px solid var(--border);padding:0 26px;display:flex;gap:2px;overflow-x:auto;position:sticky;top:58px;z-index:400}
.tab{padding:11px 16px;cursor:pointer;font-size:.78rem;font-weight:600;color:var(--muted);border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap;user-select:none}
.tab:hover{color:var(--text)}.tab.on{color:var(--blue);border-bottom-color:var(--blue)}
.page{display:none;padding:20px 26px}.page.on{display:block}
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 14px;position:relative;overflow:hidden;transition:transform .15s}
.kpi:hover{transform:translateY(-2px)}
.kpi-bar{position:absolute;top:0;left:0;right:0;height:2px}
.kpi-lbl{font-size:.61rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:7px}
.kpi-val{font-size:1.7rem;font-weight:800;font-family:monospace;line-height:1;letter-spacing:-.02em}
.sec{font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);margin:18px 0 10px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--border)}
.cc{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:17px}
.cc-hd{display:flex;align-items:center;gap:7px;margin-bottom:12px}
.cc-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.cc-title{font-size:.71rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.cc-note{font-size:.66rem;color:var(--dim);margin-top:8px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.full{margin-bottom:14px}
.chart-img{width:100%;height:auto;border-radius:8px;display:block}
.tbl-toolbar{padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--border);background:var(--surf);flex-wrap:wrap}
.search-box{flex:1;min-width:160px;background:var(--card);border:1px solid var(--border2);border-radius:7px;padding:5px 10px;color:var(--text);font-family:monospace;font-size:.75rem;outline:none}
.search-box:focus{border-color:var(--blue)}
.filter-chip{background:var(--card);border:1px solid var(--border2);color:var(--muted);padding:5px 12px;border-radius:16px;cursor:pointer;font-size:.7rem;font-weight:600;transition:all .15s}
.filter-chip.on{background:var(--blue);border-color:var(--blue);color:#fff}
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.76rem}
thead th{padding:8px 10px;background:var(--surf);color:var(--muted);font-weight:700;font-size:.63rem;text-transform:uppercase;letter-spacing:.06em;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
tbody td{padding:9px 10px;border-bottom:1px solid rgba(30,42,64,.55);vertical-align:middle}
tbody tr:hover td{background:rgba(74,140,255,.04)}
tbody tr:last-child td{border-bottom:none}
.chip{display:inline-block;padding:2px 8px;border-radius:7px;font-size:.66rem;font-weight:700;white-space:nowrap}
.c-green{background:rgba(48,217,136,.15);color:var(--green)}
.c-red{background:rgba(255,79,106,.15);color:var(--red)}
.c-yellow{background:rgba(255,194,64,.15);color:var(--yellow)}
.c-blue{background:rgba(74,140,255,.12);color:var(--blue)}
.c-grey{background:rgba(123,141,176,.12);color:var(--muted)}
.c-orange{background:rgba(251,146,60,.12);color:var(--orange)}
.mbar{display:flex;align-items:center;gap:6px;min-width:80px}
.mbar-bg{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.mbar-fill{height:100%;border-radius:3px}
.empty{padding:20px;color:var(--dim);font-size:.8rem;text-align:center}
.deadline-item{display:flex;align-items:center;gap:10px;padding:9px 4px;border-bottom:1px solid rgba(30,42,64,.55)}
.deadline-item:last-child{border-bottom:none}
.deadline-main{flex:1;min-width:0}
.deadline-name{font-size:.8rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.deadline-sub{font-size:.66rem;color:var(--muted);margin-top:2px}
.feed-item{padding:11px 4px;border-bottom:1px solid rgba(30,42,64,.55)}
.feed-item:last-child{border-bottom:none}
.feed-hd{display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap}
.feed-date{font-family:monospace;font-size:.68rem;color:var(--cyan)}
.feed-mile{font-size:.72rem;color:var(--muted)}
.feed-update{font-size:.78rem;line-height:1.5}
.feed-blockers{font-size:.72rem;color:var(--red);margin-top:3px}
.feed-next{font-size:.72rem;color:var(--muted);margin-top:2px}
footer{text-align:center;padding:18px;color:var(--dim);font-size:.67rem;border-top:1px solid var(--border);margin-top:4px;line-height:1.8}
@media(max-width:1100px){.kpi-row{grid-template-columns:repeat(3,1fr)}.g3{grid-template-columns:1fr 1fr}}
@media(max-width:800px){.g2,.g3{grid-template-columns:1fr}.kpi-row{grid-template-columns:1fr 1fr}}
@media(max-width:500px){.kpi-row{grid-template-columns:1fr}.page{padding:13px 11px}}
body.print-mode .page{display:block !important;margin-bottom:24px}
body.print-mode .tabs,body.print-mode .new-btn,body.print-mode #upload-section,body.print-mode .print-btn{display:none !important}
@media print{
  .hdr,.tabs,.new-btn,.print-btn,#upload-section{display:none !important}
  .page{padding:8px 4px}
  .kpi{break-inside:avoid}.cc{break-inside:avoid}
}
</style>
</head>
<body>

<header class="hdr">
  <div class="brand">
    <div class="brand-icon">☁️</div>
    <div class="brand-name">Azure Chat Bot <span>AWS Rebuild</span> — Program Dashboard</div>
  </div>
  <div class="hdr-right">
    <span class="pill live"><span class="dot"></span>Live</span>
    <span class="pill" style="color:var(--green);border-color:rgba(48,217,136,.3)">🔌 Offline</span>
    <span class="pill" id="file-pill">No file</span>
    <button class="print-btn" id="print-btn" style="display:none" onclick="printMode()">🖨️ Presentation Mode</button>
    <button id="theme-btn" onclick="toggleTheme()"
      style="display:flex;align-items:center;gap:5px;background:var(--card);border:1px solid var(--border);
      color:var(--muted);padding:4px 12px;border-radius:14px;font-size:.68rem;font-weight:700;cursor:pointer;
      font-family:inherit;transition:all .15s">
      <span id="theme-icon">☀️</span><span id="theme-label">Light</span>
    </button>
    <div class="health-pill" id="health-pill" style="display:none">Health: <span id="health-val">—</span></div>
  </div>
</header>

<div id="spin">
  <div class="spinner"></div>
  <div style="color:var(--text);font-weight:700;font-size:.95rem">Analysing Tracker…</div>
  <div class="spin-sub">Generating charts server-side (offline mode)</div>
</div>

<section id="upload-section">
  <div class="ucard">
    <div class="u-title">Program <span>Dashboard</span></div>
    <div class="offline-badge">🔌 100% Offline — No Internet Required</div>
    <div class="u-sub">Upload your milestone tracker workbook.<br>
      Reads Tracker, Sub-Tasks, Risk-Issue Log, Stakeholders, Change Log &amp; Sync Log sheets automatically.<br>
      All charts generated locally, no CDN required.</div>
    <div class="drop" id="drop">
      <input type="file" id="fi" accept=".xlsx,.xls,.csv"/>
      <div style="font-size:2rem;margin-bottom:9px">📂</div>
      <div style="font-size:.86rem;color:var(--muted)"><strong style="color:var(--text)">Click to browse</strong> or drag &amp; drop</div>
      <div class="fmts"><span class="fmt">.xlsx</span><span class="fmt">.xls</span><span class="fmt">.csv</span></div>
    </div>
    <div class="fname" id="fname"></div>
    <div id="uerr"></div>
    <button class="ubtn" id="abtn" disabled onclick="doUpload()">Build Dashboard →</button>
  </div>
</section>

<div id="dash">
  <button class="new-btn" onclick="reset()">← Upload New File</button>
  <nav class="tabs">
    <div class="tab on" onclick="go('ov',this)">📊 Overview</div>
    <div class="tab" onclick="go('ms',this)">🧩 Milestones</div>
    <div class="tab" onclick="go('rk',this)">⚠️ Risks &amp; Issues</div>
    <div class="tab" onclick="go('tm',this)">👥 Team</div>
  </nav>

  <section class="page on" id="pg-ov">
    <div class="kpi-row" id="kpi-row"></div>
    <div class="g3" id="ov-charts"></div>
    <div class="g2" id="ov-row2"></div>
    <div class="sec">Milestone Timeline</div>
    <div id="timeline-cards"></div>
    <div class="full cc" id="trend-card">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Progress Over Time</span></div>
      <div id="trend-chart"></div>
    </div>
    <div class="full cc" id="sync-card">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--orange)"></span><span class="cc-title">Recent Sync Updates</span></div>
      <div id="sync-feed"></div>
    </div>
  </section>

  <section class="page" id="pg-ms">
    <div class="cc full">
      <div class="tbl-toolbar">
        <input class="search-box" placeholder="Search milestones…" oninput="filterMilestones()" id="ms-search"/>
        <div id="status-filters" style="display:flex;gap:6px;flex-wrap:wrap"></div>
      </div>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Milestone</th><th>Phase</th><th>Owner</th><th>Target</th><th>Actual</th><th>Status</th><th>RAG</th><th>Progress</th><th>Revisions</th><th>Remarks</th></tr></thead>
        <tbody id="ms-tbody"></tbody>
      </table></div>
    </div>
  </section>

  <section class="page" id="pg-rk">
    <div class="kpi-row" id="risk-kpi-row" style="grid-template-columns:repeat(3,1fr)"></div>
    <div class="g2" id="risk-charts"></div>
    <div class="cc full">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--red)"></span><span class="cc-title">Risk / Issue Log</span></div>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Type</th><th>Description</th><th>Milestone</th><th>Impact</th><th>Likelihood</th><th>Owner</th><th>Status</th></tr></thead>
        <tbody id="risk-tbody"></tbody>
      </table></div>
    </div>
  </section>

  <section class="page" id="pg-tm">
    <div class="cc full">
      <div class="cc-hd"><span class="cc-dot" style="background:var(--purple)"></span><span class="cc-title">Stakeholders</span></div>
      <div class="tbl-wrap"><table>
        <thead><tr><th>Name</th><th>Role</th><th>Organization</th><th>RACI</th><th>Email</th></tr></thead>
        <tbody id="stake-tbody"></tbody>
      </table></div>
    </div>
  </section>

  <footer>Azure Chat Bot (AWS Rebuild) — Program Dashboard · Generated offline · Refresh by re-uploading your tracker</footer>
</div>

<script>
const $=id=>document.getElementById(id);
let DATA=null, activeStatusFilter='ALL';

const drop=$('drop'), fi=$('fi'), abtn=$('abtn'), fname=$('fname'), uerr=$('uerr');
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag');});
drop.addEventListener('dragleave',()=>drop.classList.remove('drag'));
drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag');if(e.dataTransfer.files.length){fi.files=e.dataTransfer.files;onFile();}});
fi.addEventListener('change',onFile);
function onFile(){
  if(!fi.files.length)return;
  fname.textContent='✓ '+fi.files[0].name;
  abtn.disabled=false;
  uerr.innerHTML='';
}
function doUpload(){
  if(!fi.files.length)return;
  $('spin').classList.add('show');
  uerr.innerHTML='';
  const fd=new FormData();
  fd.append('file',fi.files[0]);
  fetch('/upload',{method:'POST',body:fd})
    .then(r=>r.json())
    .then(d=>{
      $('spin').classList.remove('show');
      if(d.error){ uerr.innerHTML=`<div class="alert-box">${d.error}</div>`; return; }
      DATA=d;
      render(d);
      $('upload-section').style.display='none';
      $('dash').classList.add('show');
      $('file-pill').textContent=fi.files[0].name;
      $('print-btn').style.display='inline-block';
    })
    .catch(e=>{ $('spin').classList.remove('show'); uerr.innerHTML=`<div class="alert-box">Upload failed: ${e}</div>`; });
}
function reset(){
  $('dash').classList.remove('show');
  $('upload-section').style.display='flex';
  fi.value=''; fname.textContent=''; abtn.disabled=true; DATA=null;
  $('print-btn').style.display='none';
}
function go(id,el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('on'));
  el.classList.add('on'); $('pg-'+id).classList.add('on');
}
function printMode(){
  document.body.classList.add('print-mode');
  document.querySelectorAll('.page').forEach(p=>p.classList.add('on'));
  window.print();
}
window.addEventListener('afterprint', ()=>{
  document.body.classList.remove('print-mode');
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('on', i===0));
  $('pg-ov').classList.add('on');
});

function statusChip(s){
  const map={'Completed':'c-green','In Progress':'c-blue','Not Started':'c-grey','Blocked':'c-red','Delayed':'c-orange'};
  return `<span class="chip ${map[s]||'c-grey'}">${s}</span>`;
}
function ragChip(r){
  const map={'Green':'c-green','Amber':'c-yellow','Red':'c-red','-':'c-grey'};
  const label={'Green':'On Track','Amber':'Due Soon','Red':'At Risk','-':'—'};
  return `<span class="chip ${map[r]||'c-grey'}">${label[r]||r}</span>`;
}
function impactChip(i){
  const map={'High':'c-red','Medium':'c-yellow','Low':'c-green'};
  return i?`<span class="chip ${map[i]||'c-grey'}">${i}</span>`:'—';
}
function revisionChip(n){
  if(!n) return '<span class="chip c-grey">—</span>';
  const cls = n>=3?'c-red':n===2?'c-yellow':'c-blue';
  return `<span class="chip ${cls}">🔁 ×${n}</span>`;
}

// ── theme toggle ─────────────────────────────────────────────────────────────
function currentCharts(d){
  const t = document.documentElement.getAttribute('data-theme')||'dark';
  return (t==='light' ? d.charts_light : d.charts_dark) || {};
}
function applyChartTheme(){
  if(!DATA) return;
  const charts = currentCharts(DATA);
  document.querySelectorAll('img[data-chart-key]').forEach(img=>{
    const key = img.getAttribute('data-chart-key');
    if(charts[key]) img.src = charts[key];
  });
}
function setTheme(t, save){
  document.documentElement.setAttribute('data-theme', t);
  const icon=$('theme-icon'), label=$('theme-label');
  if(icon && label){
    if(t==='light'){ icon.textContent='🌙'; label.textContent='Dark'; }
    else { icon.textContent='☀️'; label.textContent='Light'; }
  }
  applyChartTheme();
  if(save){ try{ localStorage.setItem('pm_dashboard_theme', t); }catch(e){} }
}
function toggleTheme(){
  const cur = document.documentElement.getAttribute('data-theme')||'dark';
  setTheme(cur==='dark'?'light':'dark', true);
}
(function initTheme(){
  let saved='dark';
  try{ saved = localStorage.getItem('pm_dashboard_theme') || 'dark'; }catch(e){}
  setTheme(saved, false);
})();

function render(d){
  const charts = currentCharts(d);
  const kpis=[
    {l:'Total Milestones', v:d.total, c:'var(--blue)'},
    {l:'Completed', v:d.completed, c:'var(--green)'},
    {l:'Overdue', v:d.overdue, c:'var(--red)'},
    {l:'Due Soon', v:d.due_soon, c:'var(--yellow)'},
    {l:'Blocked', v:d.blocked, c:'var(--orange)'},
    {l:'Overall Progress', v:d.overall_pct+'%', c:'var(--cyan)'},
  ];
  $('kpi-row').innerHTML=kpis.map(k=>`<div class="kpi"><div class="kpi-bar" style="background:${k.c}"></div>
    <div class="kpi-lbl">${k.l}</div><div class="kpi-val" style="color:${k.c}">${k.v}</div></div>`).join('');

  $('health-pill').style.display='flex';
  const hv=$('health-val'); hv.textContent=d.health;
  hv.style.color = d.health>=70?'var(--green)':d.health>=40?'var(--yellow)':'var(--red)';

  let chartsHtml='';
  if(charts.status_donut) chartsHtml+=cardChart('Status Breakdown','var(--blue)',charts.status_donut,'status_donut');
  if(charts.rag_donut) chartsHtml+=cardChart('RAG Health','var(--red)',charts.rag_donut,'rag_donut');
  if(charts.phase_bar) chartsHtml+=cardChart('Phase-wise Progress','var(--purple)',charts.phase_bar,'phase_bar');
  $('ov-charts').innerHTML=chartsHtml || '<div class="empty">Not enough data for charts yet.</div>';

  let row2='';
  if(charts.owner_bar) row2 += cardChart('Milestones by Owner','var(--cyan)',charts.owner_bar,'owner_bar');
  if(d.flags.upcoming) row2 += upcomingCard(d.upcoming);
  $('ov-row2').innerHTML=row2;

  // ── Milestone Timeline — one card per enabled view, stacked so they're easy to compare ──
  const timelineViews=[
    {key:'timeline_roadmap', title:'Roadmap View', dot:'var(--cyan)'},
    {key:'timeline_monthly', title:'By Month', dot:'var(--blue)'},
    {key:'timeline_phase',   title:'By Phase',  dot:'var(--purple)'},
    {key:'timeline_gantt',   title:'Full Gantt (per milestone)', dot:'var(--green)'},
  ];
  let tlHtml='';
  timelineViews.forEach(v=>{
    if(charts[v.key]) tlHtml += cardChart(v.title, v.dot, charts[v.key], v.key);
  });
  $('timeline-cards').innerHTML = tlHtml || '<div class="cc empty">No target dates found to build a timeline.</div>';

  if(!d.flags.trend){
    $('trend-card').style.display='none';
  } else {
    $('trend-card').style.display='block';
    if(charts.trend_line){
      $('trend-chart').innerHTML = `<img src="${charts.trend_line}" class="chart-img" data-chart-key="trend_line"/>`;
    } else {
      $('trend-chart').innerHTML = `<div class="empty">Trend needs at least 2 uploads on different days — this dashboard remembers each upload locally, so come back after your next update.</div>`;
    }
  }

  if(!d.flags.sync_feed){
    $('sync-card').style.display='none';
  } else {
    $('sync-card').style.display='block';
    renderSyncFeed(d.sync_entries);
  }

  buildStatusFilters(d.status_counts);
  renderMilestones(d.milestones);

  const rk=[
    {l:'Total Risks/Issues', v:d.risk_summary.total, c:'var(--blue)'},
    {l:'Open', v:d.risk_summary.open, c:'var(--yellow)'},
    {l:'High Impact Open', v:d.risk_summary.high_open, c:'var(--red)'},
  ];
  $('risk-kpi-row').innerHTML=rk.map(k=>`<div class="kpi"><div class="kpi-bar" style="background:${k.c}"></div>
    <div class="kpi-lbl">${k.l}</div><div class="kpi-val" style="color:${k.c}">${k.v}</div></div>`).join('');
  let riskCharts='';
  if(charts.risk_donut) riskCharts += cardChart('Open Risks by Impact','var(--red)',charts.risk_donut,'risk_donut');
  if(charts.risk_matrix) riskCharts += cardChart('Risk Heat Matrix','var(--orange)',charts.risk_matrix,'risk_matrix');
  $('risk-charts').innerHTML = riskCharts || '<div class="empty">No risk data found yet.</div>';
  $('risk-tbody').innerHTML = d.risks.length ? d.risks.map(r=>`<tr>
    <td>${r.type||'—'}</td><td style="max-width:280px;white-space:normal">${r.description}</td>
    <td style="font-size:.7rem;color:var(--muted)">${r.milestone||'—'}</td>
    <td>${impactChip(r.impact)}</td><td>${r.likelihood||'—'}</td><td>${r.owner||'—'}</td>
    <td>${r.status||'—'}</td></tr>`).join('') : '<tr><td colspan="7" class="empty">No risks or issues logged yet.</td></tr>';

  $('stake-tbody').innerHTML = d.stakeholders.length ? d.stakeholders.map(s=>`<tr>
    <td>${s.name||'—'}</td><td>${s.role||'—'}</td><td>${s.org||'—'}</td>
    <td>${s.raci||'—'}</td><td style="font-size:.72rem;color:var(--muted)">${s.email||'—'}</td></tr>`).join('') : '<tr><td colspan="5" class="empty">No stakeholders logged yet.</td></tr>';
}

function cardChart(title,color,src,key){
  return `<div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:${color}"></span><span class="cc-title">${title}</span></div>
    <img src="${src}" class="chart-img"${key?` data-chart-key="${key}"`:''}/></div>`;
}

function upcomingCard(list){
  const body = list.length ? list.map(u=>{
    const overdue = u.days_left < 0;
    const lbl = overdue ? `Overdue by ${Math.abs(u.days_left)}d` : (u.days_left===0 ? 'Due today' : `Due in ${u.days_left}d`);
    return `<div class="deadline-item">
      <div class="deadline-main"><div class="deadline-name">${u.milestone}</div>
        <div class="deadline-sub">${u.owner||'Unassigned'} · ${u.target_date}</div></div>
      ${ragChip(u.rag)}
      <span class="chip ${overdue?'c-red':'c-blue'}" style="margin-left:4px">${lbl}</span>
    </div>`;
  }).join('') : '<div class="empty">Nothing upcoming — all clear ✓</div>';
  return `<div class="cc"><div class="cc-hd"><span class="cc-dot" style="background:var(--blue)"></span><span class="cc-title">Upcoming Deadlines</span></div>${body}</div>`;
}

function renderSyncFeed(entries){
  $('sync-feed').innerHTML = entries.length ? entries.map(e=>`<div class="feed-item">
    <div class="feed-hd"><span class="feed-date">${e.date}</span>${e.milestone?`<span class="feed-mile">· ${e.milestone}</span>`:''}${e.owner?`<span class="feed-mile">· ${e.owner}</span>`:''}</div>
    <div class="feed-update">${e.update||'—'}</div>
    ${e.blockers?`<div class="feed-blockers">⚠ ${e.blockers}</div>`:''}
    ${e.next_steps?`<div class="feed-next">→ ${e.next_steps}</div>`:''}
  </div>`).join('') : '<div class="empty">No sync log entries yet.</div>';
}

function buildStatusFilters(counts){
  const opts=['ALL',...Object.keys(counts)];
  activeStatusFilter='ALL';
  $('status-filters').innerHTML=opts.map(o=>{
    const n = o==='ALL' ? Object.values(counts).reduce((a,b)=>a+b,0) : counts[o];
    return `<div class="filter-chip ${o==='ALL'?'on':''}" data-s="${o}">${o==='ALL'?'All':o} (${n})</div>`;
  }).join('');
  document.querySelectorAll('.filter-chip').forEach(chip=>{
    chip.onclick=()=>{ activeStatusFilter=chip.dataset.s;
      document.querySelectorAll('.filter-chip').forEach(c=>c.classList.remove('on'));
      chip.classList.add('on'); renderMilestones(DATA.milestones); };
  });
}

function renderMilestones(list){
  const q=($('ms-search').value||'').toLowerCase();
  const filtered=list.filter(m=>
    (activeStatusFilter==='ALL'||m.status===activeStatusFilter) &&
    (!q || m.milestone.toLowerCase().includes(q) || (m.task_owner||'').toLowerCase().includes(q) || (m.phase||'').toLowerCase().includes(q))
  );
  $('ms-tbody').innerHTML = filtered.length ? filtered.map(m=>`<tr>
    <td style="font-weight:600;max-width:240px;white-space:normal">${m.milestone}</td>
    <td style="font-size:.72rem;color:var(--muted)">${m.phase||'—'}</td>
    <td>${m.task_owner||'—'}</td>
    <td style="font-family:monospace;font-size:.72rem">${m.target_date||'—'}</td>
    <td style="font-family:monospace;font-size:.72rem">${m.actual_date||'—'}</td>
    <td>${statusChip(m.status)}</td>
    <td>${ragChip(m.rag)}</td>
    <td><div class="mbar"><div class="mbar-bg"><div class="mbar-fill" style="width:${m.pct_complete}%;background:${m.rag==='Red'?'var(--red)':m.rag==='Amber'?'var(--yellow)':'var(--green)'}"></div></div><span style="font-size:.68rem;color:var(--muted)">${m.pct_complete}%</span></div></td>
    <td>${revisionChip(m.date_changes)}</td>
    <td style="font-size:.7rem;color:var(--muted);max-width:180px;white-space:normal">${m.remarks||m.reason||'—'}</td>
    </tr>`).join('') : '<tr><td colspan="10" class="empty">No milestones match.</td></tr>';
}
function filterMilestones(){ if(DATA) renderMilestones(DATA.milestones); }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); lan_ip = s.getsockname()[0]; s.close()
    except Exception:
        lan_ip = "YOUR_SERVER_IP"
    port = 5050
    print("\n" + "=" * 64)
    print("  Azure Chat Bot (AWS Rebuild) — Program Dashboard  v2")
    print("=" * 64)
    print(f"  Local   :  http://localhost:{port}")
    print(f"  Remote  :  http://{lan_ip}:{port}")
    print("=" * 64)
    print("  🔌 100% OFFLINE — No internet required")
    print("  Charts generated by Python matplotlib (server-side)")
    print("  Formats : .xlsx  .xls  .csv")
    print("  Press Ctrl+C to stop")
    print("=" * 64 + "\n")
    app.run(debug=False, port=port, host="0.0.0.0", threaded=True)
