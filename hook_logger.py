#!/usr/bin/env python3
"""
Claude Code Hook Logger
記錄 Claude Code 的 Hook 事件，用於偵測限速等狀態。

使用方式：
  由 Claude Code Hook 機制自動呼叫，不需手動執行。
  Hook 會透過 stdin 傳入 JSON 資料。

日誌位置：
  ~/claude-usage-monitor/logs/hook_events.jsonl
"""

import json
import sys
import os
from datetime import datetime, timezone

# 日誌目錄
LOG_DIR = os.path.expanduser("~/claude-usage-monitor/logs")
LOG_FILE = os.path.join(LOG_DIR, "hook_events.jsonl")


def main():
    # 確保日誌目錄存在
    os.makedirs(LOG_DIR, exist_ok=True)

    # 從 stdin 讀取 Hook 傳入的 JSON
    try:
        raw = sys.stdin.read()
        hook_data = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, Exception):
        hook_data = {"raw_input": raw[:500] if raw else "empty"}

    # 加上時間戳
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "hook_event": hook_data.get("hook_event_name", "unknown"),
        "session_id": hook_data.get("session_id", ""),
        "data": hook_data,
    }

    # 寫入日誌（append 模式）
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Exit 0 = 允許動作繼續
    sys.exit(0)


if __name__ == "__main__":
    main()
