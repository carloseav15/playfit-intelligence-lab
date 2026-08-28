#!/bin/bash
# Runs when the qa subagent finishes (SubagentStop, matcher "qa"). Decrements
# .pending-qa-count by 1 only if qa wrote a well-formed {"result":"pass"} --
# validated with jq, not a substring grep. See .claude/agents/qa.md.

RESULT_FILE="$CLAUDE_PROJECT_DIR/.claude/.qa-result"
COUNT_FILE="$CLAUDE_PROJECT_DIR/.claude/.pending-qa-count"
COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)

PASSED=false
if [ -f "$RESULT_FILE" ]; then
  RESULT=$(jq -r '.result // empty' "$RESULT_FILE" 2>/dev/null)
  if [ "$RESULT" = "pass" ]; then
    PASSED=true
  fi
fi

if [ "$PASSED" = true ] && [ "$COUNT" -gt 0 ] 2>/dev/null; then
  echo $((COUNT - 1)) > "$COUNT_FILE"
else
  echo "qa finished without a validated pass result -- pending-qa-count stays at $COUNT." >&2
fi

rm -f "$RESULT_FILE"
exit 0
