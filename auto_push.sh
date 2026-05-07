#!/usr/bin/env bash
# auto_push.sh — watches the repo for changes and pushes them to origin automatically.
# Usage:  bash auto_push.sh [commit_message]
#         If no message is given, a timestamped default is used.
# Run in background:  nohup bash auto_push.sh &

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="origin"
BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
POLL_INTERVAL=10   # seconds between checks when running as a watcher

# ---------- one-shot push (called internally or directly) ----------
do_push() {
  local msg="${1:-auto: $(date '+%Y-%m-%d %H:%M:%S')}"
  cd "$REPO_DIR" || exit 1

  if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "[auto_push] No changes to commit."
    return 0
  fi

  git add -A
  if ! git commit -m "$msg"; then
    echo "[auto_push] ERROR: commit failed." >&2
    return 1
  fi
  if git push "$REMOTE" "$BRANCH"; then
    echo "[auto_push] Pushed to $REMOTE/$BRANCH at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "[auto_push] ERROR: push failed. Check credentials/network." >&2
    return 1
  fi
}

# ---------- entry point ----------
if [ "$1" = "--watch" ]; then
  # Continuous watcher mode: poll every POLL_INTERVAL seconds
  echo "[auto_push] Watching $REPO_DIR every ${POLL_INTERVAL}s. Press Ctrl+C to stop."
  while true; do
    do_push
    sleep "$POLL_INTERVAL"
  done
else
  # One-shot mode: commit & push immediately with optional message
  do_push "$1"
fi
