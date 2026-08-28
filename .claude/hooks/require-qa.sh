#!/bin/bash
# Blocks the session from stopping while any file exists under .pending-qa/ --
# one file per implementer run, created with `mktemp` (atomic -- the old
# read-modify-write counter lost updates under parallel increments, proven
# with a 100-parallel test: only 23/100 survived). Incremented via
# data-ai-engineer's SubagentStop hook, decremented (one file) by
# check-qa-result.sh only on a validated "pass".

INPUT=$(cat)
PENDING_DIR="$CLAUDE_PROJECT_DIR/.claude/.pending-qa"
BLOCK_COUNT_FILE="$CLAUDE_PROJECT_DIR/.claude/.stop-block-count"

PENDING=0
if [ -d "$PENDING_DIR" ] && [ -n "$(ls -A "$PENDING_DIR" 2>/dev/null)" ]; then
  PENDING=1
fi

if [ "$PENDING" -eq 0 ]; then
  rm -f "$BLOCK_COUNT_FILE"
  exit 0
fi

# Claude Code allows a Stop hook to force at most 8 consecutive continuations
# before overriding it. Rather than conceding on the first stop_hook_active,
# we track our own block count and keep blocking up to a smaller budget
# first. This narrows the escape window; it does not close it.
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active // false')" = "true" ]; then
  BLOCKS=$(cat "$BLOCK_COUNT_FILE" 2>/dev/null || echo 0)
  if [ "$BLOCKS" -lt 3 ] 2>/dev/null; then
    echo $((BLOCKS + 1)) > "$BLOCK_COUNT_FILE"
    echo "qa still pending after a forced continuation ($((BLOCKS + 1))/3 retries) -- invoke qa, don't just retry Stop." >&2
    exit 2
  fi
  rm -f "$BLOCK_COUNT_FILE"
  exit 0
fi

echo "data-ai-engineer made changes this session that qa hasn't independently verified (pass) yet. Invoke qa before finishing." >&2
exit 2
