# claude-usage-monitor

A local usage monitoring tool for Claude Code. It reads session data from `~/.claude/`, analyzes usage patterns, and generates Markdown reports — ideal for reporting AI tool utilization to management.

## Features

- Scans all project sessions under `~/.claude/projects/`
- Tracks AI interactions, working hours, and token usage
- Detects plan rate limit events via Claude Code Hook mechanism
- Generates management-oriented Markdown reports (executive summary, task overview, daily work log)

## Files

| File | Description |
|------|-------------|
| `claude_usage.py` | Main script — parses session data and generates reports |
| `hook_logger.py` | Hook event logger — called automatically by Claude Code Hooks |

## Usage

```bash
# Default: all projects, last 30 days
python3 claude_usage.py

# Last 7 days
python3 claude_usage.py --days 7

# Filter by project name
python3 claude_usage.py --project my-project

# Save report to file
python3 claude_usage.py --output ~/Desktop/report.md

# Show all data (no day limit)
python3 claude_usage.py --days 0
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--days` | 30 | Analyze last N days of data (0 = all) |
| `--project` | none | Filter by project name (fuzzy match) |
| `--output` | stdout | Write output to file |
| `--claude-dir` | `~/.claude` | Claude data directory path |

## Hook Setup

`hook_logger.py` works with Claude Code's Hook mechanism to log session events, particularly for detecting plan rate limit hits.

Add the following to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/hook_logger.py"
          }
        ]
      }
    ]
  }
}
```

Hook logs are stored at `~/claude-usage-monitor/logs/hook_events.jsonl` with automatic 30-day retention.

## Requirements

- Python 3.10+
- Claude Code CLI (provides the `~/.claude/` data directory)