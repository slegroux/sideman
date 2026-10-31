"""The MCP-layer unit suite (no Live needed), run through its own main().

See conftest.py for why pytest does not collect the suites directly.
"""
import test_mcp


def test_mcp_layer_unit_suite():
    assert test_mcp.main() == 0, test_mcp.FAILURES
