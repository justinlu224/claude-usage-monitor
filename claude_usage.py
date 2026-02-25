#!/usr/bin/env python3
"""
Claude Code Pro 用量監控腳本

讀取本機 ~/.claude/ 的 session 資料，按 5 小時週期分組，
產出 Markdown 或 CSV 報告。

用法：
    python3 claude_usage.py                        # 全部專案，近 30 天
    python3 claude_usage.py --days 7               # 近 7 天
    python3 claude_usage.py --project health-shop  # 篩選特定專案
    python3 claude_usage.py --format  csv --output ~/Desktop/report.csv #CSV 格式
    python3 claude_usage.py --output ~/Desktop/report.md  # 存檔
"""

import argparse
import csv
import io
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RateLimitEvent:
    """單次 API 錯誤（限速/過載）事件。"""
    timestamp: datetime
    status: int          # 429=rate_limit, 529=overloaded
    error_type: str      # e.g. "overloaded_error", "rate_limit_error"
    retry_attempt: int
    retry_wait_ms: float


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
    # Rate limit events
    rate_limit_events: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return (self.input_tokens + self.output_tokens +
                self.cache_read_tokens + self.cache_creation_tokens)

    @property
    def duration_minutes(self) -> float:
        delta = self.modified - self.created
        return max(delta.total_seconds() / 60, 0)

    @property
    def rate_limit_count(self) -> int:
        return len(self.rate_limit_events)

    @property
    def total_retry_wait_ms(self) -> float:
        return sum(e.retry_wait_ms for e in self.rate_limit_events)


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

    @property
    def total_rate_limits(self) -> int:
        return sum(s.rate_limit_count for s in self.sessions)

    @property
    def total_retry_wait_seconds(self) -> float:
        return sum(s.total_retry_wait_ms for s in self.sessions) / 1000

    @property
    def has_rate_limit(self) -> bool:
        return self.total_rate_limits > 0


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


def parse_jsonl(jsonl_path: str) -> dict:
    """Parse a session JSONL file for token usage and metadata.

    Returns a dict with keys: input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, first_ts, last_ts, first_prompt, user_msg_count,
    git_branch, cwd, summary, rate_limit_events.
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
        "rate_limit_events": [],
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
                        content = rec.get("message", {}).get("content", "")
                        if isinstance(content, str):
                            # Strip XML-like tags from auto-generated prompts
                            text = content
                            if text.startswith("<"):
                                # Try to find actual text after tags
                                import re
                                cleaned = re.sub(r"<[^>]+>", "", text).strip()
                                text = cleaned if cleaned else text
                            result["first_prompt"] = text[:80]
                        elif isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    result["first_prompt"] = c.get("text", "")[:80]
                                    break

                # Sum token usage from assistant records
                if rec_type == "assistant":
                    msg_usage = rec.get("message", {}).get("usage", {})
                    if msg_usage:
                        result["input_tokens"] += msg_usage.get("input_tokens", 0)
                        result["output_tokens"] += msg_usage.get("output_tokens", 0)
                        result["cache_read_tokens"] += msg_usage.get("cache_read_input_tokens", 0)
                        result["cache_creation_tokens"] += msg_usage.get("cache_creation_input_tokens", 0)

                # Capture rate limit / overload errors
                if (rec_type == "system"
                        and rec.get("subtype") == "api_error"
                        and ts):
                    err = rec.get("error", {})
                    inner = err.get("error", {}).get("error", {})
                    result["rate_limit_events"].append(RateLimitEvent(
                        timestamp=parse_timestamp(ts),
                        status=err.get("status", 0),
                        error_type=inner.get("type", "unknown"),
                        retry_attempt=rec.get("retryAttempt", 0),
                        retry_wait_ms=rec.get("retryInMs", 0),
                    ))
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

                # Parse JSONL for token usage
                jsonl_path = entry.get("fullPath", "")
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
                    rate_limit_events=parsed.get("rate_limit_events", []),
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
                rate_limit_events=parsed.get("rate_limit_events", []),
            ))

    sessions.sort(key=lambda s: s.created)
    return sessions


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

WINDOW_HOURS = 5

# 高負載定義：滿足以下任一條件即為高負載週期
# 1. 週期內 sessions 數 ≥ 3
# 2. 週期內總訊息數 ≥ 50
# 3. 週期內總 output tokens ≥ 50K
HEAVY_MIN_SESSIONS = 3
HEAVY_MIN_MESSAGES = 50
HEAVY_MIN_OUTPUT_TOKENS = 50_000


def is_heavy_window(w) -> bool:
    """判定是否為高負載週期。"""
    return (
        len(w.sessions) >= HEAVY_MIN_SESSIONS
        or w.total_messages >= HEAVY_MIN_MESSAGES
        or w.total_output_tokens >= HEAVY_MIN_OUTPUT_TOKENS
    )


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
    if len(clean) > max_len:
        clean = clean[:max_len] + "…"
    return clean


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
# Report generators
# ---------------------------------------------------------------------------

def generate_markdown_report(
    sessions: list[SessionRecord],
    windows: list[Window],
    days: int | None,
) -> str:
    """Generate a Markdown report for management."""
    lines = []

    # Header
    if sessions:
        start_date = fmt_time(sessions[0].created).split(" ")[0]
        end_date = fmt_time(sessions[-1].created).split(" ")[0]
    else:
        start_date = end_date = "N/A"

    lines.append("# Claude Code Pro 使用報告")
    lines.append("")
    period = f"近 {days} 天（{start_date} — {end_date}）" if days else f"{start_date} — {end_date}"
    lines.append(f"**期間：** {period}")
    lines.append(
        f"**5 小時週期數：** {len(windows)}  |  "
        f"**總 session 數：** {len(sessions)}  |  "
        f"**總對話數：** {sum(s.message_count for s in sessions)}"
    )
    lines.append("")

    # Summary table
    lines.append("---")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append("| 指標 | 數值 |")
    lines.append("|------|------|")
    lines.append(f"| 活躍 5 小時週期 | {len(windows)} |")

    heavy_windows = sum(1 for w in windows if is_heavy_window(w))
    lines.append(f"| 高負載週期 | {heavy_windows} |")

    total_output = sum(s.output_tokens for s in sessions)
    lines.append(f"| 總 output tokens | {fmt_tokens(total_output)} |")

    total_input = sum(s.input_tokens for s in sessions)
    lines.append(f"| 總 input tokens | {fmt_tokens(total_input)} |")

    total_cache = sum(s.cache_read_tokens + s.cache_creation_tokens for s in sessions)
    lines.append(f"| 總 cache tokens | {fmt_tokens(total_cache)} |")

    total_rl = sum(s.rate_limit_count for s in sessions)
    rl_windows = sum(1 for w in windows if w.has_rate_limit)
    total_rl_wait = sum(s.total_retry_wait_ms for s in sessions) / 1000
    lines.append(f"| 限速/過載次數 | {total_rl} 次（影響 {rl_windows} 個週期） |")
    if total_rl_wait > 0:
        lines.append(f"| 限速等待總時間 | {total_rl_wait:.1f} 秒 |")

    # Most active day
    day_msgs: dict[str, int] = defaultdict(int)
    day_sessions: dict[str, int] = defaultdict(int)
    for s in sessions:
        day_key = fmt_time(s.created).split(" ")[0]
        day_msgs[day_key] += s.message_count
        day_sessions[day_key] += 1

    if day_msgs:
        busiest = max(day_msgs, key=day_msgs.get)
        lines.append(
            f"| 最活躍日 | {busiest}"
            f"（{day_sessions[busiest]} sessions, {day_msgs[busiest]} 則訊息） |"
        )

    # Projects
    all_projects = list(dict.fromkeys(s.project_name for s in sessions))
    lines.append(f"| 涵蓋專案 | {', '.join(all_projects)} |")
    lines.append("")

    # Tasks completed
    lines.append("## 完成的主要任務")
    lines.append("")
    for s in sessions:
        label = sanitize_text(s.summary or s.first_prompt, max_len=80)
        if label:
            lines.append(f"- {label} ({s.project_name})")
    lines.append("")

    # Window details
    lines.append("---")
    lines.append("")
    lines.append("## 各 5 小時週期明細")
    lines.append("")

    for i, w in enumerate(windows, 1):
        tags = []
        if is_heavy_window(w):
            tags.append("HIGH")
        if w.has_rate_limit:
            tags.append("RATE-LIMITED")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"### 週期 {i} — {fmt_time(w.anchor)} ~ {fmt_time_short(w.end)}{tag_str}"
        )
        stats_parts = [
            f"**Sessions:** {len(w.sessions)}",
            f"**訊息:** {w.total_messages}",
            f"**Output tokens:** {fmt_tokens(w.total_output_tokens)}",
        ]
        if w.has_rate_limit:
            stats_parts.append(
                f"**限速:** {w.total_rate_limits} 次（等待 {w.total_retry_wait_seconds:.1f}s）"
            )
        lines.append("  |  ".join(stats_parts))
        lines.append("")
        lines.append("| 時間 | 任務摘要 | 專案 | 訊息數 | Output Tokens | 限速 |")
        lines.append("|------|---------|------|--------|---------------|------|")
        for s in w.sessions:
            label = sanitize_text(s.summary or s.first_prompt)
            rl_mark = f"{s.rate_limit_count}次" if s.rate_limit_count > 0 else "-"
            lines.append(
                f"| {fmt_time_short(s.created)} | {label} | "
                f"{s.project_name} | {s.message_count} | {fmt_tokens(s.output_tokens)} | {rl_mark} |"
            )
        lines.append("")

    # Token breakdown
    lines.append("---")
    lines.append("")
    lines.append("## Token 用量明細")
    lines.append("")
    lines.append("| 日期 | 週期 | Sessions | Input | Output | Cache Read | Cache Create | Total | 狀態 |")
    lines.append("|------|------|----------|-------|--------|------------|--------------|-------|------|")
    for i, w in enumerate(windows, 1):
        day = fmt_time(w.anchor).split(" ")[0]
        time_range = f"{fmt_time_short(w.anchor)}~{fmt_time_short(w.end)}"
        status_parts = []
        if is_heavy_window(w):
            status_parts.append("HIGH")
        if w.has_rate_limit:
            status_parts.append(f"限速{w.total_rate_limits}次")
        status_mark = ", ".join(status_parts) if status_parts else "-"
        lines.append(
            f"| {day} | {time_range} | {len(w.sessions)} | "
            f"{fmt_tokens(w.total_input_tokens)} | {fmt_tokens(w.total_output_tokens)} | "
            f"{fmt_tokens(w.total_cache_read)} | {fmt_tokens(w.total_cache_creation)} | "
            f"{fmt_tokens(w.total_tokens)} | {status_mark} |"
        )
    lines.append("")

    # Rate limit analysis
    lines.append("---")
    lines.append("")
    lines.append("## 限制分析")
    lines.append("")

    heavy_list = [
        (i, w) for i, w in enumerate(windows, 1)
        if is_heavy_window(w)
    ]

    if heavy_list:
        lines.append(f"共 **{len(heavy_list)}** 個高負載週期（佔比 {len(heavy_list)}/{len(windows)}）：")
        lines.append("")
        for idx, w in heavy_list:
            actual_span = (w.sessions[-1].modified - w.sessions[0].created).total_seconds() / 3600
            reasons = []
            if len(w.sessions) >= HEAVY_MIN_SESSIONS:
                reasons.append(f"{len(w.sessions)} sessions")
            if w.total_messages >= HEAVY_MIN_MESSAGES:
                reasons.append(f"{w.total_messages} 則訊息")
            if w.total_output_tokens >= HEAVY_MIN_OUTPUT_TOKENS:
                reasons.append(f"output {fmt_tokens(w.total_output_tokens)}")
            lines.append(
                f"- **[HIGH] 週期 {idx}** ({fmt_time(w.anchor)})："
                f"跨度 {actual_span:.1f} 小時，"
                f"觸發條件：{', '.join(reasons)}"
            )
        lines.append("")
    else:
        lines.append("目前各週期使用量較為分散，尚未出現高負載使用情形。")
        lines.append("")

    # Rate limit summary
    rl_windows = [(i, w) for i, w in enumerate(windows, 1) if w.has_rate_limit]
    if rl_windows:
        lines.append(f"### 限速事件")
        lines.append("")
        lines.append(
            f"共 **{sum(w.total_rate_limits for _, w in rl_windows)}** 次限速/過載，"
            f"影響 **{len(rl_windows)}** 個週期："
        )
        lines.append("")
        for idx, w in rl_windows:
            all_events = []
            for s in w.sessions:
                all_events.extend(s.rate_limit_events)
            error_types = list(dict.fromkeys(e.error_type for e in all_events))
            lines.append(
                f"- **[RATE-LIMITED] 週期 {idx}** ({fmt_time(w.anchor)})："
                f"{w.total_rate_limits} 次，"
                f"等待 {w.total_retry_wait_seconds:.1f}s，"
                f"類型：{', '.join(error_types)}"
            )
        lines.append("")
    else:
        lines.append("### 限速事件")
        lines.append("")
        lines.append("本期間內未偵測到限速/過載事件。")
        lines.append("")

    lines.append("> **建議：** 若出現 [RATE-LIMITED] 標記，代表已實際觸及方案上限，")
    lines.append("> 工作被迫中斷等待，建議升級方案以提升效率。")
    lines.append("")

    # Criteria explanation
    lines.append("---")
    lines.append("")
    lines.append("## 標記定義")
    lines.append("")
    lines.append("### [HIGH] 高負載")
    lines.append("週期滿足以下**任一**條件：")
    lines.append("")
    lines.append(f"| 條件 | 門檻值 |")
    lines.append(f"|------|--------|")
    lines.append(f"| 週期內 session 數 | >= {HEAVY_MIN_SESSIONS} |")
    lines.append(f"| 週期內總訊息數 | >= {HEAVY_MIN_MESSAGES} |")
    lines.append(f"| 週期內總 output tokens | >= {fmt_tokens(HEAVY_MIN_OUTPUT_TOKENS)} |")
    lines.append("")
    lines.append("### [RATE-LIMITED] 限速")
    lines.append("週期內偵測到 API 回傳 429 (rate limit) 或 529 (overloaded) 錯誤，代表已觸及方案用量上限。")
    lines.append("")

    return "\n".join(lines)


def generate_csv_report(sessions: list[SessionRecord], windows: list[Window]) -> str:
    """Generate CSV report, one row per session."""
    # Build session -> window mapping
    session_window: dict[str, int] = {}
    for i, w in enumerate(windows, 1):
        for s in w.sessions:
            session_window[s.session_id] = i

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "date", "start_time", "end_time", "project", "summary",
        "messages", "duration_min", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_creation_tokens", "total_tokens",
        "rate_limit_count", "rate_limit_wait_ms",
        "window_id", "git_branch",
    ])

    for s in sessions:
        writer.writerow([
            fmt_time(s.created).split(" ")[0],
            fmt_time(s.created),
            fmt_time(s.modified),
            s.project_name,
            s.summary or s.first_prompt[:80],
            s.message_count,
            round(s.duration_minutes, 1),
            s.input_tokens,
            s.output_tokens,
            s.cache_read_tokens,
            s.cache_creation_tokens,
            s.total_tokens,
            s.rate_limit_count,
            round(s.total_retry_wait_ms, 0),
            session_window.get(s.session_id, 0),
            s.git_branch,
        ])

    return output.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claude Code Pro 用量監控 — 分析 session 資料並產出報告"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="分析近 N 天的資料（預設 30，設 0 顯示全部）"
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="篩選特定專案名稱（模糊比對）"
    )
    parser.add_argument(
        "--format", choices=["markdown", "csv"], default="markdown",
        help="輸出格式（預設 markdown）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="輸出到檔案（預設印到 stdout）"
    )
    parser.add_argument(
        "--claude-dir", type=str, default=None,
        help="Claude 資料目錄（預設 ~/.claude）"
    )

    args = parser.parse_args()

    claude_dir = args.claude_dir or os.path.expanduser("~/.claude")

    # days=0 means show all
    effective_days = args.days if args.days else None

    # Load data
    sessions = load_all_sessions(claude_dir, args.project, effective_days)

    if not sessions:
        print("找不到符合條件的 session 資料。", file=sys.stderr)
        print(f"  搜尋目錄: {claude_dir}/projects/", file=sys.stderr)
        if args.project:
            print(f"  專案篩選: {args.project}", file=sys.stderr)
        if effective_days:
            print(f"  天數範圍: 近 {effective_days} 天", file=sys.stderr)
        else:
            print("  天數範圍: 全部", file=sys.stderr)
        sys.exit(1)

    # Group into windows
    windows = group_into_windows(sessions)

    # Generate report
    if args.format == "csv":
        report = generate_csv_report(sessions, windows)
    else:
        report = generate_markdown_report(sessions, windows, effective_days)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"報告已輸出到: {output_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
