#!/bin/bash
# Blocks the session from stopping if data-ai-engineer made changes this session
# that qa hasn't verified yet. Marker set/cleared by SubagentStop hooks in
# .claude/settings.json. This repo has no test suite (CI only byte-compiles), so
# this is the closest thing to a real safety net that exists.

INPUT=$(cat)

if [ "$(echo "$INPUT" | jq -r '.stop_hook_active // false')" = "true" ]; then
  exit 0
fi

MARKER="$CLAUDE_PROJECT_DIR/.claude/.pending-qa"
if [ -f "$MARKER" ]; then
  echo "data-ai-engineer made changes this session that the qa subagent hasn't independently verified yet. Invoke qa before finishing." >&2
  exit 2
fi

exit 0
