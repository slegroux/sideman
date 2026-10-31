"""The unit suites (no Live needed), run through their own main().

See conftest.py for why pytest does not collect the suites directly.
"""
import test_handlers
import test_mcp


def test_mcp_layer_unit_suite():
    assert test_mcp.main() == 0, test_mcp.FAILURES


def test_engine_unit_suite():
    assert test_handlers.main([]) == 0, test_handlers.FAILURES
