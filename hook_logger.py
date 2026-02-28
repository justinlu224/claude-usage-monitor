#!/usr/bin/env python3
"""
Claude Code Hook Logger
記錄 Claude Code 的 Hook 事件，用於偵測限速等狀態。

使用方式：
  由 Claude Code Hook 機制自動呼叫，不需手動執行。
  Hook 會透過 stdin 傳入 JSON 資料。

日誌位置：
  ~/claude-usage-report/logs/hook_events.jsonl
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

# 日誌目錄
LOG_DIR = os.path.expanduser("~/claude-usage-report/logs")
LOG_FILE = os.path.join(LOG_DIR, "hook_events.jsonl")

# 保留天數（與報告預設一致）
RETENTION_DAYS = 30


def cleanup_old_logs():
    """Remove log entries older than RETENTION_DAYS using streaming write."""
    if not os.path.exists(LOG_FILE):
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=LOG_DIR,
            prefix=".hook_cleanup_", suffix=".tmp", delete=False,
        ) as tmp:
            tmp_path = tmp.name
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        ts_str = rec.get("logged_at", "")
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts < cutoff:
                            continue
                    except (json.JSONDecodeError, ValueError):
                        pass  # Keep unparseable lines to avoid data loss
                    tmp.write(line)
        os.replace(tmp_path, LOG_FILE)
    except (OSError, IOError):
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


def main():
    # 確保日誌目錄存在
    os.makedirs(LOG_DIR, exist_ok=True)

    # 從 stdin 讀取 Hook 傳入的 JSON（限制 1MB 避免磁碟空間耗盡）
    MAX_INPUT_BYTES = 1_000_000
    try:
        raw = sys.stdin.read(MAX_INPUT_BYTES)
        hook_data = json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, Exception):
        hook_data = {}

    # 每天最多清理一次（用 marker 檔記錄上次清理日期）
    marker = os.path.join(LOG_DIR, ".last_cleanup")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    need_cleanup = True
    if os.path.exists(marker):
        try:
            with open(marker, "r") as f:
                need_cleanup = f.read().strip() != today
        except OSError:
            pass
    if need_cleanup:
        cleanup_old_logs()
        with open(marker, "w") as f:
            f.write(today)

    # 僅保留必要欄位，避免將完整對話內容寫入日誌
    record = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "hook_event": hook_data.get("hook_event_name", "unknown"),
        "session_id": hook_data.get("session_id", ""),
        "data": {
            "session_id": hook_data.get("session_id", ""),
            "last_assistant_message": hook_data.get("last_assistant_message", ""),
        },
    }

    # 寫入日誌（append 模式）
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Exit 0 = 允許動作繼續
    sys.exit(0)


if __name__ == "__main__":
    main()
