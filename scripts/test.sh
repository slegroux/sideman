#!/usr/bin/env bash
# Run both suites.
#
#   tests/test_mcp.py  unit, no Live required     -> CI-able, ~1s
#   tests/smoke.py     integration, needs Live    -> skipped if unreachable, ~70s
#
# The integration suite is slow because it does real work: six scratch-track
# create/delete cycles and a genuine device load through the browser. That is
# the point - it exercises Live, not a mock. Do not mistake it for a hang.
#
# The integration suite creates a scratch MIDI track named __lomtest and
# deletes it. It refuses to delete a track whose name changed underneath it,
# so a rename mid-run leaves the track rather than removing the wrong one.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY=python3
rc=0

echo "== unit: MCP layer (no Live needed) =="
"$PY" "$REPO/tests/test_mcp.py" "$@" || rc=1

echo
if "$REPO/scripts/lomcli.py" ping >/dev/null 2>&1; then
  echo "== integration: engine against live Ableton =="
  "$PY" "$REPO/tests/smoke.py" "$@" || rc=1
else
  echo "== integration: SKIPPED (Live not reachable on :9878) =="
  echo "   Start Live with AbletonLOM enabled to run the engine tests."
fi

echo
[ $rc -eq 0 ] && echo "SUITES PASS" || echo "SUITES FAILED"
exit $rc
