#!/usr/bin/env bash
# Run the docs, lock and coverage checks, then all four suites. Which seam
# each suite covers: see README.
#
#   tests/test_mcp.py     unit,   no Live required -> CI-able, ~1s
#   tests/test_e2e.py     e2e,    needs Live       -> MCP layer -> socket -> Live
#   tests/test_client.py  client, needs Live       -> sideman.client -> Live
#   tests/smoke.py        engine, needs Live       -> raw socket, ~70s
#
# The Live-dependent suites are slow because they do real work: six scratch-track
# create/delete cycles and a genuine device load through the browser. Do not
# mistake it for a hang. They create a MIDI track named __lomtest and delete it,
# refusing to delete one whose name changed underneath them.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY=python3
rc=0

echo "== docs: generated reference up to date =="
"$PY" "$REPO/scripts/gen_docs.py" --check || rc=1

# The README's superset claim rests on this. It was previously never run by the
# suite, which is how it drifted to a failing state unnoticed. The harness exit
# codes are the contract:
#   0  superset verified
#   1  a real unresolved attribute
#   2  competitor checkouts absent (a fresh clone) - a skip, not a failure
#   3  no census - the setup itself is broken
echo
echo "== env: installed versions match constraints.txt =="
# The suites prove only the versions they run on. A venv that drifted from
# the lock tests one mcp while the package ships another.
"$PY" - "$REPO/constraints.txt" <<'PYEOF' || rc=1
import importlib.metadata as m, re, sys
try:
    from packaging.markers import Marker
except ImportError:  # not in the lock; pip vendors it
    from pip._vendor.packaging.markers import Marker
bad = []
for line in open(sys.argv[1]):
    hit = re.match(r"([A-Za-z0-9_.-]+)==([^\s;]+)\s*(?:;(.*))?", line)
    if not hit or (hit[3] and not Marker(hit[3].strip()).evaluate()):
        continue  # a pin for another Python or platform
    try:
        have = m.version(hit[1])
    except m.PackageNotFoundError:
        continue  # not installed here
    if have != hit[2]:
        bad.append("%s %s (locked %s)" % (hit[1], have, hit[2]))
if bad:
    print("   FAILED - " + ", ".join(bad))
    print("   sync: VIRTUAL_ENV=.venv uv pip install -e . -c constraints.txt")
    sys.exit(1)
print("   in sync")
PYEOF

echo
echo "== coverage: superset claim still holds =="
HARNESS="$REPO/scripts/coverage_harness.py"
# Checked before running because CPython also exits 2 when it cannot open the
# script, which is indistinguishable from the harness's own "sources absent"
# 2 - i.e. a missing harness would report as a benign skip.
if [ ! -f "$HARNESS" ]; then
  echo "   FAILED - harness missing at $HARNESS"; rc=1
else
  # stderr is kept: the harness explains both of its refusals there, and
  # swallowing it is how a broken setup gets reported as a benign skip.
  "$PY" "$HARNESS" >/dev/null
  case $? in
    0) echo "   superset claim verified" ;;
    2) echo "   SKIPPED (competitor checkouts absent from ~/Projects/vendor)" ;;
    3) echo "   FAILED - no census; run scripts/census.py"; rc=1 ;;
    *) echo "   FAILED - run scripts/coverage_harness.py --verbose"; rc=1 ;;
  esac
fi

echo
echo "== unit: MCP layer (no Live needed) =="
"$PY" "$REPO/tests/test_mcp.py" "$@" || rc=1

echo
if "$REPO/scripts/lomcli.py" ping >/dev/null 2>&1; then
  echo "== e2e: MCP layer -> Live =="
  "$PY" "$REPO/tests/test_e2e.py" "$@" || rc=1
  echo
  echo "== client: sideman.client -> Live =="
  "$PY" "$REPO/tests/test_client.py" "$@" || rc=1
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
