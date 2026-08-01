"""AbletonLOM MCP server - generic Live Object Model access.

Unlike every other Ableton MCP server, this does not hand-wrap individual
properties. It exposes the Live Object Model itself, so any property Live has
is reachable without shipping new code.
"""
from __future__ import annotations

import json
import socket
from typing import Any

from mcp.server.mcpserver import MCPServer

HOST, PORT = "127.0.0.1", 9878
TIMEOUT = 25.0

PATH_HELP = """
Paths are space-separated, zero-indexed, unquoted:
    live_set tracks 0 mixer_device volume
    live_set tracks 2 clip_slots 0 clip
    live_set scenes 1

Roots: live_set (the Song), live_app (the Application), app_view.
Use `lom_describe` to discover what exists at any path - never guess member names.
""".strip()

mcp = MCPServer(
    "ableton-lom",
    instructions=(
        "Generic access to Ableton Live's Object Model. There is no fixed tool "
        "per feature - navigate the object graph instead.\n\n"
        + PATH_HELP
        + "\n\nWorkflow: lom_describe to discover, then lom_get/lom_set/lom_call. "
        "Availability is per-instance: a member listed for a type may still be "
        "unavailable on a given object (see the `unavailable` map)."
    ),
)


class LiveError(RuntimeError):
    pass


def _request(op: str, params: dict[str, Any] | None = None) -> Any:
    try:
        s = socket.create_connection((HOST, PORT), timeout=TIMEOUT)
    except ConnectionRefusedError:
        raise LiveError(
            f"Nothing listening on {HOST}:{PORT}. Ableton Live must be running "
            "with 'AbletonLOM' selected as a Control Surface in "
            "Preferences > Link, Tempo & MIDI."
        )
    try:
        s.sendall(json.dumps({"id": "mcp", "op": op,
                              "params": params or {}}).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                raise LiveError("connection closed by Live before a full reply")
            buf += chunk
    except socket.timeout:
        raise LiveError(f"Live did not respond within {TIMEOUT}s")
    finally:
        s.close()

    resp = json.loads(buf.split(b"\n", 1)[0].decode())
    if not resp.get("ok"):
        err = resp.get("error") or {}
        raise LiveError(f"{err.get('type', 'Error')}: {err.get('message', resp)}")
    return resp.get("result")


@mcp.tool()
def lom_describe(path: str = "live_set", include_values: bool = True) -> str:
    """Discover everything at a Live Object Model path: properties (with current
    values), navigable children (with counts), and callable functions.

    START HERE. This is how you find out what exists rather than guessing.

    Returns an `unavailable` map for members that exist on the type but not on
    THIS instance - e.g. a MIDI track has no `input_meter_left`, only the main
    track has a `crossfader`. That is correct Live behaviour, not an error.

    Paths are space-separated and zero-indexed, e.g.
    "live_set tracks 0 mixer_device volume". Roots: live_set, live_app, app_view.
    """
    return json.dumps(_request("describe", {"path": path,
                                            "include_values": include_values}),
                      indent=2)


@mcp.tool()
def lom_get(path: str, property: str) -> str:
    """Read one property from a Live object.

    Example: lom_get("live_set", "tempo") -> 120.0
             lom_get("live_set tracks 0", "name") -> "1-MIDI"

    Use lom_describe first to discover valid member names; do not guess.
    """
    return json.dumps(_request("get", {"path": path, "property": property}),
                      indent=2)


@mcp.tool()
def lom_set(path: str, property: str, value: Any) -> str:
    """Write one property on a Live object. Wrapped in a native undo step, so
    the user can revert it with Cmd-Z.

    Example: lom_set("live_set", "tempo", 128)
             lom_set("live_set tracks 0", "name", "Drums")

    Use lom_describe first to discover valid member names; do not guess.
    """
    return json.dumps(_request("set", {"path": path, "property": property,
                                       "value": value}), indent=2)


@mcp.tool()
def lom_call(path: str, function: str, args: list[Any] | None = None,
             confirm: bool = False) -> str:
    """Call a function on a Live object. Wrapped in a native undo step.

    Example: lom_call("live_set", "create_midi_track", [-1])
             lom_call("live_set tracks 0 clip_slots 0", "fire")

    Destructive functions (delete_track, delete_scene, remove_all_notes, ...)
    refuse to run unless confirm=True. Ask the user before setting it.
    """
    return json.dumps(_request("call", {"path": path, "function": function,
                                        "args": args or [],
                                        "confirm": confirm}), indent=2)


@mcp.tool()
def lom_count(path: str, child: str) -> str:
    """Count items in a Live collection, e.g. lom_count("live_set", "tracks")."""
    return json.dumps(_request("count", {"path": path, "child": child}),
                      indent=2)


@mcp.tool()
def lom_types() -> str:
    """Census of every LOM type Live exposes and every member of each.

    This is the map of the entire addressable surface. Large - prefer
    lom_describe for day-to-day navigation.
    """
    return json.dumps(_request("types"), indent=2)


@mcp.tool()
def lom_ping() -> str:
    """Check the connection to Ableton Live and whether the engine loaded."""
    return json.dumps(_request("ping"), indent=2)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
