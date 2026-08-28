#!/bin/bash
# Blocks the session from stopping while pending-qa-count > 0. A counter, not a
# boolean, so two implementer runs in the same session don't let one qa pass
# clear both. Incremented by data-ai-engineer's SubagentStop hook, decremented
# by check-qa-result.sh only on a validated "pass".

INPUT=$(cat)

# Known limitation: a second Stop attempt right after a block can slip through
# without qa actually running again -- Claude Code's own anti-loop design
# requires this check. Don't treat this gate as unconditional.
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active // false')" = "true" ]; then
  exit 0
fi

COUNT_FILE="$CLAUDE_PROJECT_DIR/.claude/.pending-qa-count"
COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)

if [ "$COUNT" -gt 0 ] 2>/dev/null; then
  echo "data-ai-engineer made $COUNT change(s) this session that qa hasn't independently verified (pass) yet. Invoke qa before finishing." >&2
  exit 2
fi

exit 0
