"""pytest entry point. The suites are scripts: run them with `scripts/test.sh`.

pytest must not collect them one function at a time:

- test_mcp.py, test_handlers.py: check() records a failure in a list instead
  of raising, so under pytest a failing check reports as a pass. (test_mcp's
  socket stub is installed at import, so it can no longer reach Live, but the
  results would still be meaningless.)
- test_e2e.py, test_client.py, smoke.py: these need Live open on a scratch Set,
  and they create and delete tracks.

So pytest collects only test_pytest_entry.py, which runs the two unit
suites (test_mcp.py, test_handlers.py) through their own main() and fails if
either does.
"""
collect_ignore = ["test_mcp.py", "test_handlers.py", "test_e2e.py", "test_client.py", "smoke.py"]
