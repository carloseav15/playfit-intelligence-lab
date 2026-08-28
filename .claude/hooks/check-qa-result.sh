#!/bin/bash
# Runs when the qa subagent finishes (SubagentStop, matcher "qa"). Removes
# every file under .pending-qa/ whose content is exactly PASS.
#
# No shared .qa-result file anymore -- the old design raced under
# concurrency (100 parallel qa passes left 92-97 pending instead of 0,
# reproduced and confirmed against this repo's own copy too). qa now writes
# its verdict INTO the specific pending file it verified (see
# .claude/agents/qa.md), so there's no shared mutable state between
# concurrent qa runs. This hook just sweeps for anything marked PASS.

PENDING_DIR="$CLAUDE_PROJECT_DIR/.claude/.pending-qa"

if [ -d "$PENDING_DIR" ]; then
  for f in "$PENDING_DIR"/*; do
    [ -f "$f" ] || continue
    if [ "$(cat "$f" 2>/dev/null)" = "PASS" ]; then
      rm -f "$f"
    fi
  done
fi

exit 0
