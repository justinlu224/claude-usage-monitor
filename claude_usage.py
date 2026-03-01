#!/usr/bin/env python3
"""
Claude Code Usage Reporter

Reads local ~/.claude/ session data, groups by 5-hour windows,
and generates a Markdown report.

Usage:
    python3 claude_usage.py                        # All projects, last 30 days
    python3 claude_usage.py --days 7               # Last 7 days
    python3 claude_usage.py --project health-shop  # Filter by project
    python3 claude_usage.py --output ~/Desktop/report.md  # Save to file
"""

import argparse
import html
import json
import os
import re
import sys
from collections import defaultdict
from hook_logger import LOG_FILE
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_HOURS = 5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RateLimitEvent:
    """Plan rate limit event (detected via Hook)."""
    timestamp: datetime
    session_id: str
    message: str         # e.g. "You've hit your limit · resets 8pm (Asia/Taipei)"


@dataclass
class SessionRecord:
    session_id: str
    project_path: str
    project_name: str
    created: datetime
    modified: datetime
    message_count: int
    summary: str
    first_prompt: str
    git_branch: str
    # Token usage (from JSONL)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens +
                self.cache_read_tokens + self.cache_creation_tokens)

    @property
    def duration_minutes(self) -> float:
        """Session working duration in minutes.

        Outlier handling:
        - Over 5 hours (e.g. /resume across days) -> estimate by message count (~1 min each)
        - Under 1 minute but many messages (e.g. resumed after compression) -> estimate by message count
        """
        delta = self.modified - self.created
        mins = max(delta.total_seconds() / 60, 0)
        if mins > WINDOW_HOURS * 60:
            return float(self.message_count)
        if mins < 1 and self.message_count > 5:
            return float(self.message_count)
        return mins


@dataclass
class Window:
    anchor: datetime
    end: datetime
    sessions: list = field(default_factory=list)

    @property
    def total_messages(self) -> int:
        return sum(s.message_count for s in self.sessions)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.sessions)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.sessions)

    @property
    def total_cache_read(self) -> int:
        return sum(s.cache_read_tokens for s in self.sessions)

    @property
    def total_cache_creation(self) -> int:
        return sum(s.cache_creation_tokens for s in self.sessions)

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.sessions)

    @property
    def projects(self) -> list:
        return list(dict.fromkeys(s.project_name for s in self.sessions))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO 8601 timestamp to UTC datetime."""
    ts_str = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str)


def get_project_name(project_path: str) -> str:
    """Extract short project name from full path."""
    return Path(project_path).name if project_path else "unknown"


def _extract_user_text(rec: dict) -> str:
    """Extract text content from a user record."""
    content = rec.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                return c.get("text", "").strip()
    return ""


# System message prefixes — not real user input
_SYSTEM_PREFIXES = (
    "<",              # XML tags: <command-name>, <local-command-*>, <system-reminder>
    "Caveat:",        # Auto-generated caveat messages
    "Base directory", # Skill loading prompt
    "[Request ",      # [Request interrupted by user]
)


def _is_real_user_input(text: str) -> bool:
    """Check if text is real user input (not a system message)."""
    return not text.startswith(_SYSTEM_PREFIXES)


def parse_jsonl(jsonl_path: str) -> dict:
    """Parse a session JSONL file for token usage and metadata.

    Returns a dict with keys: input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, first_ts, last_ts, first_prompt, user_msg_count,
    git_branch, cwd, summary.
    """
    result = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "first_ts": None,
        "last_ts": None,
        "first_prompt": "",
        "user_msg_count": 0,
        "git_branch": "",
        "cwd": "",
        "summary": "",
    }

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rec_type = rec.get("type")

                # Track timestamps from any record that has one
                ts = rec.get("timestamp")
                if ts:
                    if not result["first_ts"]:
                        result["first_ts"] = ts
                    result["last_ts"] = ts

                # Extract summary from summary records
                if rec_type == "summary":
                    result["summary"] = rec.get("summary", "")

                # Count user messages and extract first prompt
                if rec_type == "user":
                    result["user_msg_count"] += 1
                    if not result["git_branch"]:
                        result["git_branch"] = rec.get("gitBranch", "")
                    if not result["cwd"]:
                        result["cwd"] = rec.get("cwd", "")
                    if not result["first_prompt"]:
                        text = _extract_user_text(rec)
                        if text and _is_real_user_input(text):
                            result["first_prompt"] = text[:80]

                # Sum token usage from assistant records
                if rec_type == "assistant":
                    msg_usage = rec.get("message", {}).get("usage", {})
                    if msg_usage:
                        result["input_tokens"] += msg_usage.get("input_tokens", 0)
                        result["output_tokens"] += msg_usage.get("output_tokens", 0)
                        result["cache_read_tokens"] += msg_usage.get("cache_read_input_tokens", 0)
                        result["cache_creation_tokens"] += msg_usage.get("cache_creation_input_tokens", 0)

    except (OSError, IOError):
        pass

    return result


def load_all_sessions(
    claude_dir: str,
    filter_project: str | None = None,
    days: int | None = None,
) -> list[SessionRecord]:
    """Load sessions from all projects under ~/.claude/projects/.

    Reads sessions-index.json for indexed sessions, then scans for
    unindexed JSONL files and parses them directly.
    """

    projects_dir = Path(claude_dir) / "projects"
    if not projects_dir.is_dir():
        print(f"Error: {projects_dir} not found", file=sys.stderr)
        return []

    cutoff = None
    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    sessions = []

    for proj_dir in projects_dir.iterdir():
        if not proj_dir.is_dir():
            continue

        # --- Phase 1: Load indexed sessions ---
        indexed_ids: set[str] = set()
        project_path_from_index = ""

        index_path = proj_dir / "sessions-index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                index_data = {}

            project_path_from_index = index_data.get("originalPath", "")

            for entry in index_data.get("entries", []):
                session_id = entry.get("sessionId", "")
                indexed_ids.add(session_id)

                project_path = entry.get("projectPath", "") or project_path_from_index
                project_name = get_project_name(project_path)

                if filter_project and filter_project.lower() not in project_name.lower():
                    continue

                created_str = entry.get("created")
                modified_str = entry.get("modified")
                if not created_str or not modified_str:
                    continue

                created = parse_timestamp(created_str)
                modified = parse_timestamp(modified_str)

                if cutoff and created < cutoff:
                    continue

                # Parse JSONL for token usage (validate path is within claude_dir)
                jsonl_path = entry.get("fullPath", "")
                claude_dir_real = os.path.realpath(claude_dir) + os.sep
                if jsonl_path and not os.path.realpath(jsonl_path).startswith(claude_dir_real):
                    jsonl_path = ""
                parsed = parse_jsonl(jsonl_path) if jsonl_path and os.path.exists(jsonl_path) else {}

                sessions.append(SessionRecord(
                    session_id=session_id,
                    project_path=project_path,
                    project_name=project_name,
                    created=created,
                    modified=modified,
                    message_count=entry.get("messageCount", 0),
                    summary=entry.get("summary", "") or parsed.get("summary", ""),
                    first_prompt=entry.get("firstPrompt", "")[:80],
                    git_branch=entry.get("gitBranch", ""),
                    input_tokens=parsed.get("input_tokens", 0),
                    output_tokens=parsed.get("output_tokens", 0),
                    cache_read_tokens=parsed.get("cache_read_tokens", 0),
                    cache_creation_tokens=parsed.get("cache_creation_tokens", 0),
                ))

        # --- Phase 2: Pick up unindexed JSONL files ---
        for jsonl_file in proj_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            if session_id in indexed_ids:
                continue

            parsed = parse_jsonl(str(jsonl_file))

            # Skip empty/invalid sessions
            if not parsed["first_ts"] or parsed["user_msg_count"] == 0:
                continue

            created = parse_timestamp(parsed["first_ts"])
            modified = parse_timestamp(parsed["last_ts"]) if parsed["last_ts"] else created

            # Determine project path from JSONL cwd or fall back to index
            project_path = parsed["cwd"] or project_path_from_index
            project_name = get_project_name(project_path)

            if filter_project and filter_project.lower() not in project_name.lower():
                continue

            if cutoff and created < cutoff:
                continue

            sessions.append(SessionRecord(
                session_id=session_id,
                project_path=project_path,
                project_name=project_name,
                created=created,
                modified=modified,
                message_count=parsed["user_msg_count"],
                summary=parsed["summary"],
                first_prompt=parsed["first_prompt"],
                git_branch=parsed["git_branch"],
                input_tokens=parsed["input_tokens"],
                output_tokens=parsed["output_tokens"],
                cache_read_tokens=parsed["cache_read_tokens"],
                cache_creation_tokens=parsed["cache_creation_tokens"],
            ))

    sessions.sort(key=lambda s: s.created)
    return sessions




def load_hook_rate_limits(hook_log_path: str | None = None) -> list[RateLimitEvent]:
    """Load plan rate limit events from Hook logs.

    Hook logs are written by hook_logger.py. A Stop event whose
    last_assistant_message starts with "You've hit your limit" indicates
    a plan rate limit.

    Match criteria (all must be met):
    1. hook_event == "Stop"
    2. last_assistant_message starts with "You've hit your limit"
    3. Message contains "resets" (fixed format, avoids false positives)
    """
    if not hook_log_path:
        hook_log_path = LOG_FILE

    events: list[RateLimitEvent] = []

    if not os.path.exists(hook_log_path):
        return events

    try:
        with open(hook_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if rec.get("hook_event") != "Stop":
                    continue

                data = rec.get("data", {})
                msg = data.get("last_assistant_message", "")

                # Strict match: must start with fixed prefix and contain "resets"
                if not (msg.startswith("You've hit your limit")
                        and "resets" in msg.lower()):
                    continue

                events.append(RateLimitEvent(
                    timestamp=parse_timestamp(rec["logged_at"]),
                    session_id=data.get("session_id", ""),
                    message=msg.strip(),
                ))
    except (OSError, IOError):
        pass

    return events



# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def group_into_windows(sessions: list[SessionRecord]) -> list[Window]:
    """Group sessions into rolling 5-hour windows."""
    if not sessions:
        return []

    windows = []
    anchor = sessions[0].created
    current: list[SessionRecord] = []

    for session in sessions:
        if (session.created - anchor).total_seconds() >= WINDOW_HOURS * 3600:
            windows.append(Window(
                anchor=anchor,
                end=anchor + timedelta(hours=WINDOW_HOURS),
                sessions=current,
            ))
            anchor = session.created
            current = []
        current.append(session)

    if current:
        windows.append(Window(
            anchor=anchor,
            end=anchor + timedelta(hours=WINDOW_HOURS),
            sessions=current,
        ))

    return windows


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def sanitize_text(text: str, max_len: int = 60) -> str:
    """Sanitize text for Markdown table cells (strip newlines, truncate)."""
    clean = text.replace("\n", " ").replace("\r", " ").replace("|", "/").strip()
    # Strip markdown syntax
    clean = re.sub(r"^#{1,6}\s+", "", clean)      # ## heading → heading (line-start only)
    clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", clean)  # **bold** → bold
    clean = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", clean)       # ![alt](url) → remove images
    clean = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", clean)   # [text](url) → text
    clean = re.sub(r"\s{2,}", " ", clean)         # collapse whitespace
    clean = clean.strip()
    if len(clean) > max_len:
        clean = clean[:max_len] + "…"
    return html.escape(clean)


def fmt_tokens(n: int) -> str:
    """Format token count for display."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_time(dt: datetime) -> str:
    """Format datetime as local time string."""
    local = dt.astimezone()
    return local.strftime("%Y-%m-%d %H:%M")


def fmt_time_short(dt: datetime) -> str:
    local = dt.astimezone()
    return local.strftime("%H:%M")


# ---------------------------------------------------------------------------
# Task filtering
# ---------------------------------------------------------------------------

# Keywords for non-substantive work, excluded from task list
_SKIP_TASK_KEYWORDS = (
    "Summarize this",
    "claude install",
    "npm run",
    "git ",
)


def _is_meaningful_task(label: str) -> bool:
    """Check if a task summary represents meaningful work."""
    if not label or len(label) < 5:
        return False
    for kw in _SKIP_TASK_KEYWORDS:
        if label.lower().startswith(kw.lower()):
            return False
    return True


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def _prepare_report_data(
    sessions: list[SessionRecord],
    windows: list[Window],
    hook_events: list[RateLimitEvent] | None,
) -> dict:
    """Prepare aggregated data used by all report sections."""

    total_messages = sum(s.message_count for s in sessions)
    total_duration = sum(
        min(sum(s.duration_minutes for s in w.sessions), WINDOW_HOURS * 60)
        for w in windows
    )

    session_map = {s.session_id: s for s in sessions}
    matched_hook_events = [
        e for e in (hook_events or [])
        if e.session_id in session_map
    ]

    meaningful_tasks = []
    for s in sessions:
        label = sanitize_text(s.summary or s.first_prompt, max_len=80)
        if _is_meaningful_task(label):
            meaningful_tasks.append((label, s.project_name, s))

    days_map: dict[str, list[SessionRecord]] = defaultdict(list)
    for s in sessions:
        day = fmt_time(s.created).split(" ")[0]
        days_map[day].append(s)

    rl_by_date: dict[str, list[RateLimitEvent]] = defaultdict(list)
    for evt in matched_hook_events:
        day = fmt_time(evt.timestamp).split(" ")[0]
        rl_by_date[day].append(evt)

    active_dates = set()
    for s in sessions:
        label = sanitize_text(s.summary or s.first_prompt, max_len=80)
        if _is_meaningful_task(label):
            active_dates.add(fmt_time(s.created).split(" ")[0])
    for evt in matched_hook_events:
        active_dates.add(fmt_time(evt.timestamp).split(" ")[0])

    return {
        "total_messages": total_messages,
        "total_duration": total_duration,
        "matched_hook_events": matched_hook_events,
        "meaningful_tasks": meaningful_tasks,
        "days_map": days_map,
        "rl_by_date": rl_by_date,
        "active_days": len(active_dates),
    }


def _render_header(sessions: list[SessionRecord], days: int | None) -> list[str]:
    """Render report title and period line."""
    if sessions:
        start_date = fmt_time(sessions[0].created).split(" ")[0]
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        start_date = end_date = "N/A"

    period = f"Last {days} days ({start_date} — {end_date})" if days else f"{start_date} — {end_date}"

    return [
        "# Claude Code Usage Report",
        "",
        f"**Period:** {period}",
        "",
    ]


def _render_executive_summary(data: dict) -> list[str]:
    """Render the Executive Summary section."""
    lines = [
        "## Executive Summary",
        "",
        f"During this period, Claude Code AI assisted development across "
        f"**{data['active_days']}** active days, completing **{len(data['meaningful_tasks'])}** tasks "
        f"with **{data['total_messages']}** AI interactions "
        f"over approximately **{data['total_duration'] / 60:.1f} hours** of total working time.",
    ]
    if data["matched_hook_events"]:
        lines.append(
            f"The plan usage limit was hit **{len(data['matched_hook_events'])} time(s)**, "
            f"forcing work interruptions while waiting for reset."
        )
    lines.append("")
    return lines


def _render_usage_overview(data: dict) -> list[str]:
    """Render the Usage Overview metrics table."""
    active_days = data["active_days"]
    total_messages = data["total_messages"]
    matched_hook_events = data["matched_hook_events"]
    meaningful_tasks = data["meaningful_tasks"]

    lines = [
        "---",
        "",
        "## Usage Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Active Days | {active_days} |",
        f"| Total AI Interactions | {total_messages} |",
        f"| Total Working Hours | {data['total_duration'] / 60:.1f} hrs |",
        f"| Avg. Interactions / Day | {total_messages / max(active_days, 1):.0f} |",
        f"| Plan Limit Hits | {len(matched_hook_events)} |",
    ]
    projects_with_tasks = list(dict.fromkeys(proj for _, proj, _ in meaningful_tasks))
    if projects_with_tasks:
        lines.append(f"| Projects | {', '.join(sanitize_text(p) for p in projects_with_tasks)} |")
    lines.append("")
    return lines


def _render_completed_tasks(data: dict) -> list[str]:
    """Render the Completed Tasks section grouped by project."""
    meaningful_tasks = data["meaningful_tasks"]
    lines = ["---", "", "## Completed Tasks", ""]

    if meaningful_tasks:
        tasks_by_project: dict[str, list[str]] = defaultdict(list)
        for label, proj, _ in meaningful_tasks:
            tasks_by_project[proj].append(label)

        for proj, tasks in tasks_by_project.items():
            lines.append(f"### {sanitize_text(proj)}")
            lines.append("")
            for t in tasks:
                lines.append(f"- {t}")
            lines.append("")
    else:
        lines.append("(No identifiable work items recorded)")
        lines.append("")

    return lines


def _render_work_log(data: dict) -> list[str]:
    """Render the AI Work Log with daily timeline and rate limit events."""
    days_map = data["days_map"]
    rl_by_date = data["rl_by_date"]
    lines = ["---", "", "## AI Work Log", ""]

    for day, day_sessions in days_map.items():
        day_tasks = []
        for s in day_sessions:
            label = sanitize_text(s.summary or s.first_prompt, max_len=80)
            if _is_meaningful_task(label):
                day_tasks.append(label)

        day_rl = rl_by_date.get(day, [])

        if not day_tasks and not day_rl:
            continue

        session_count = len(day_sessions)
        raw_dur = sum(s.duration_minutes for s in day_sessions)
        day_dur = min(raw_dur, WINDOW_HOURS * 60)

        last_modified = max(s.modified for s in day_sessions)
        last_mod_day = fmt_time(last_modified).split(" ")[0]

        if last_mod_day != day:
            dur_str = f"continued to {last_mod_day}"
        elif day_dur >= 60:
            dur_str = f"{day_dur / 60:.1f} hrs"
        else:
            dur_str = f"{day_dur:.0f} min"

        lines.append(f"### {day} ({session_count} sessions, {dur_str})")
        lines.append("")
        for t in day_tasks:
            lines.append(f"- {t}")

        for evt in day_rl:
            msg = evt.message
            reset_part = ""
            if "resets" in msg.lower():
                idx = msg.lower().index("resets")
                reset_part = msg[idx:].strip()
                if "(" in reset_part:
                    reset_part = reset_part[:reset_part.index("(")].strip()
            lines.append(
                f"- Hit plan limit — work interrupted, waiting for reset ({html.escape(reset_part)})"
            )

        lines.append("")

    return lines


def _render_conclusion(data: dict) -> list[str]:
    """Render the Conclusion & Recommendations section."""
    matched_hook_events = data["matched_hook_events"]
    lines = ["---", "", "## Conclusion & Recommendations", ""]

    if matched_hook_events:
        lines.append(
            f"1. **Plan limit reached:** Hit the plan usage limit "
            f"{len(matched_hook_events)} time(s) during this period, "
            f"forcing work interruptions and directly impacting development efficiency."
        )
        lines.append(
            f"2. **Recommend plan upgrade:** Upgrading to a higher plan would eliminate "
            f"work interruptions and improve AI tool ROI."
        )
    else:
        lines.append(
            "No plan usage limits were hit during this period. "
            "Current quota is sufficient for the workload."
        )
        lines.append(
            "Recommend continued monitoring of usage trends. "
            "Consider upgrading if limits are reached and work is interrupted."
        )
    lines.append("")
    return lines


def generate_markdown_report(
    sessions: list[SessionRecord],
    windows: list[Window],
    days: int | None,
    hook_events: list[RateLimitEvent] | None = None,
) -> str:
    """Generate a Markdown report for management."""
    data = _prepare_report_data(sessions, windows, hook_events)
    lines = []
    lines.extend(_render_header(sessions, days))
    lines.extend(_render_executive_summary(data))
    lines.extend(_render_usage_overview(data))
    lines.extend(_render_completed_tasks(data))
    lines.extend(_render_work_log(data))
    lines.extend(_render_conclusion(data))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claude Code Usage Reporter — analyze session data and generate reports"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Analyze last N days of data (default: 30, 0 = all)"
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Filter by project name (fuzzy match)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write output to file (default: stdout)"
    )
    parser.add_argument(
        "--claude-dir", type=str, default=None,
        help="Claude data directory (default: ~/.claude)"
    )

    args = parser.parse_args()

    claude_dir = args.claude_dir or os.path.expanduser("~/.claude")

    # days=0 means show all
    effective_days = args.days if args.days else None

    # Load data
    sessions = load_all_sessions(claude_dir, args.project, effective_days)

    if not sessions:
        print("No matching session data found.", file=sys.stderr)
        print(f"  Search directory: {claude_dir}/projects/", file=sys.stderr)
        if args.project:
            print(f"  Project filter: {args.project}", file=sys.stderr)
        if effective_days:
            print(f"  Date range: last {effective_days} days", file=sys.stderr)
        else:
            print("  Date range: all", file=sys.stderr)
        sys.exit(1)

    # Group into windows
    windows = group_into_windows(sessions)

    # Load hook rate limits
    hook_events = load_hook_rate_limits()

    # Generate report
    report = generate_markdown_report(sessions, windows, effective_days, hook_events)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {output_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
