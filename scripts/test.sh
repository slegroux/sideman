#!/usr/bin/env bash
# Run both suites.
#
#   tests/test_mcp.py  unit,  no Live required    -> CI-able, ~1s
#   tests/test_e2e.py  e2e,   needs Live          -> MCP layer -> socket -> Live
#   tests/smoke.py     engine, needs Live         -> raw socket, ~70s
#
# The three cover different seams. test_mcp stubs the socket so it never touches
# Live; smoke uses a raw socket so it never loads the MCP layer. Only test_e2e
# exercises the path Claude actually takes, where a response the tool cannot
# serialise or an error escaping as the wrong type would finally show up.
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

echo "== docs: generated reference up to date =="
"$PY" "$REPO/scripts/gen_docs.py" --check || rc=1

# The README's superset claim rests on this. It was previously never run by the
# suite, which is how it drifted to a failing state unnoticed. Exit 2 means the
# competitor checkouts are absent (a fresh clone) - that is a skip, not a
# failure. Exit 1 means a real unresolved attribute.
echo
echo "== coverage: superset claim still holds =="
"$PY" "$REPO/scripts/coverage_harness.py" >/dev/null 2>&1
case $? in
  0) echo "   superset claim verified" ;;
  2) echo "   SKIPPED (competitor checkouts absent from ~/Projects/vendor)" ;;
  *) echo "   FAILED - run scripts/coverage_harness.py --verbose"; rc=1 ;;
esac

echo
echo "== unit: MCP layer (no Live needed) =="
"$PY" "$REPO/tests/test_mcp.py" "$@" || rc=1

echo
if "$REPO/scripts/lomcli.py" ping >/dev/null 2>&1; then
  echo "== e2e: MCP layer -> Live =="
  "$PY" "$REPO/tests/test_e2e.py" "$@" || rc=1
  echo
  echo "== integration: engine against live Ableton =="
  "$PY" "$REPO/tests/smoke.py" "$@" || rc=1
else
  echo "== e2e + integration: SKIPPED (Live not reachable on :9878) =="
  echo "   Start Live with AbletonLOM enabled to run the engine tests."
fi

echo
[ $rc -eq 0 ] && echo "SUITES PASS" || echo "SUITES FAILED"
exit $rc
