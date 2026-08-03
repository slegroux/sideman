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
def lom_get(path: str, property: str, offset: int | None = None,
            limit: int | None = None) -> str:
    """Read one property from a Live object.

    Example: lom_get("live_set", "tempo") -> 120.0
             lom_get("live_set tracks 0", "name") -> "1-MIDI"

    A list longer than 64 reports only {"__vector__": true, "count": N}, which
    hides exactly the long lists worth reading. Pass offset/limit to page
    through one - Drift's 66 parameters, a rack's chains:

        lom_get("live_set tracks 0 devices 0", "parameters", limit=25)
        lom_get("live_set tracks 0 devices 0", "parameters", offset=25, limit=25)

    The window returns `items` plus `count`, `returned` and `truncated`.
    Max 512 per page; a limit above that is clamped, not refused.

    Use lom_describe first to discover valid member names; do not guess.
    """
    p: dict[str, Any] = {"path": path, "property": property}
    if offset is not None:
        p["offset"] = offset
    if limit is not None:
        p["limit"] = limit
    return json.dumps(_request("get", p), indent=2)


@mcp.tool()
def lom_set(path: str, property: str, value: Any) -> str:
    """Write one property on a Live object. Wrapped in a native undo step, so
    the user can revert it with Cmd-Z.

    Example: lom_set("live_set", "tempo", 128)
             lom_set("live_set tracks 0", "name", "Drums")

    For a property that holds a Live OBJECT rather than a scalar, name it by
    path with {"__path__": "<lom path>"}:

        lom_set("live_set view", "selected_track",
                {"__path__": "live_set tracks 3"})

    Use lom_describe first to discover valid member names; do not guess.
    """
    return json.dumps(_request("set", {"path": path, "property": property,
                                       "value": value}), indent=2)


@mcp.tool()
def lom_call(path: str, function: str, args: list[Any] | None = None,
             confirm: bool = False, offset: int | None = None,
             limit: int | None = None) -> str:
    """Call a function on a Live object. Wrapped in a native undo step.

    Example: lom_call("live_set", "create_midi_track", [-1])
             lom_call("live_set tracks 0 clip_slots 0", "fire")

    Some functions take a Live OBJECT, not a scalar - a path string reaches them
    as a str and Live rejects the call ("did not match C++ signature"). Pass
    {"__path__": "<lom path>"} for those and it is resolved inside Live:

        lom_call("live_set", "move_device",
                 [{"__path__": "live_set tracks 5 devices 0"},
                  {"__path__": "live_set tracks 7"}, 0])

    Markers work anywhere in args, including nested in lists/dicts. A plain
    string is never reinterpreted as a path, so ordinary string args are safe.

    When the return value is a long list it reports only its count. Page it
    with offset/limit - this is how you read a plugin's real parameter list,
    which Live knows even while `parameters` exposes one entry:

        lom_call(dev, "get_parameter_names", limit=50)   -> 50 of 2362 names

    Destructive functions refuse to run unless confirm=True - anything named
    delete_*, remove_* or clear_*, plus crop. Ask the user before setting it.
    """
    p: dict[str, Any] = {"path": path, "function": function,
                         "args": args or [], "confirm": confirm}
    if offset is not None:
        p["offset"] = offset
    if limit is not None:
        p["limit"] = limit
    return json.dumps(_request("call", p), indent=2)


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


# --------------------------------------------------------------------- notes
# Typed wrappers: notes cannot round-trip through generic get/set, because
# Live hands back MidiNote objects and expects MidiNoteSpecification on write.


@mcp.tool()
def clip_get_notes(path: str, from_pitch: int = 0, pitch_span: int = 128,
                   from_time: float = 0.0, time_span: float | None = None) -> str:
    """Read MIDI notes from a clip, with note_id, pitch, start_time, duration,
    velocity, mute, probability, velocity_deviation and release_velocity.

    path is a clip path, e.g. "live_set tracks 0 clip_slots 0 clip".
    Defaults cover the whole clip. Keep the note_ids - clip_modify_notes needs them.
    """
    p: dict[str, Any] = {"path": path, "from_pitch": from_pitch,
                         "pitch_span": pitch_span, "from_time": from_time}
    if time_span is not None:
        p["time_span"] = time_span
    return json.dumps(_request("notes_get", p), indent=2)


@mcp.tool()
def clip_add_notes(path: str, notes: list[dict[str, Any]]) -> str:
    """Add MIDI notes to a clip. Undoable.

    Each note: {"pitch": 60, "start_time": 0.0, "duration": 1.0,
                "velocity": 100, "mute": false}
    pitch is a MIDI note number (60 = C3); times are in beats.
    """
    return json.dumps(_request("notes_add", {"path": path, "notes": notes}),
                      indent=2)


@mcp.tool()
def clip_modify_notes(path: str, notes: list[dict[str, Any]]) -> str:
    """Edit existing notes in place. Undoable.

    Each entry needs "note_id" (from clip_get_notes) plus the fields to change,
    e.g. {"note_id": 3, "pitch": 62, "velocity": 80}.
    Returns `unmatched_note_ids` for ids that no longer exist rather than failing.
    """
    return json.dumps(_request("notes_modify", {"path": path, "notes": notes}),
                      indent=2)


@mcp.tool()
def clip_remove_notes(path: str, from_pitch: int = 0, pitch_span: int = 128,
                      from_time: float = 0.0,
                      time_span: float | None = None) -> str:
    """Remove notes in a pitch/time range. Undoable. Defaults remove ALL notes."""
    p: dict[str, Any] = {"path": path, "from_pitch": from_pitch,
                         "pitch_span": pitch_span, "from_time": from_time}
    if time_span is not None:
        p["time_span"] = time_span
    return json.dumps(_request("notes_remove", p), indent=2)


# ------------------------------------------------------------------- browser


@mcp.tool()
def browser_list(path: str = "") -> str:
    """Browse Live's library. Empty path lists the roots (instruments, sounds,
    drums, audio_effects, midi_effects, plugins, clips, samples, packs,
    user_library, current_project, max_for_live).

    Then descend by name with "/", e.g. "instruments/Drift" or "plugins/VST3".
    """
    return json.dumps(_request("browser_list", {"path": path}), indent=2)


@mcp.tool()
def browser_load(path: str, track_index: int | None = None) -> str:
    """Load a browser item (instrument, effect, plugin, sample) onto a track.

    path is a browser path from browser_list, e.g. "instruments/Drift".
    Loads onto the selected track unless track_index is given. Undoable.
    """
    p: dict[str, Any] = {"path": path}
    if track_index is not None:
        p["track_index"] = track_index
    return json.dumps(_request("browser_load", p), indent=2)


# --------------------------------------------------------------- automation


@mcp.tool()
def clip_envelope_get(path: str, parameter: str, samples: int = 8,
                      times: list[float] | None = None) -> str:
    """Sample a clip's automation envelope for one device/mixer parameter.

    path      - clip path, e.g. "live_set tracks 0 clip_slots 0 clip"
    parameter - parameter path, e.g. "live_set tracks 0 mixer_device volume"
                or "live_set tracks 0 devices 0 parameters 1"
    """
    p: dict[str, Any] = {"path": path, "parameter": parameter,
                         "samples": samples}
    if times:
        p["times"] = times
    return json.dumps(_request("envelope_get", p), indent=2)


@mcp.tool()
def clip_envelope_insert_step(path: str, parameter: str, time: float,
                              length: float, value: float) -> str:
    """Write a flat automation step into a clip envelope, creating the envelope
    if it does not exist. Times are in beats. Undoable."""
    return json.dumps(_request("envelope_insert_step",
                               {"path": path, "parameter": parameter,
                                "time": time, "length": length,
                                "value": value}), indent=2)


@mcp.tool()
def clip_envelope_clear(path: str, parameter: str | None = None) -> str:
    """Clear one parameter's envelope, or ALL envelopes on the clip if
    parameter is omitted. Undoable."""
    p: dict[str, Any] = {"path": path}
    if parameter:
        p["parameter"] = parameter
    return json.dumps(_request("envelope_clear", p), indent=2)


# -------------------------------------------------------------- arrangement


@mcp.tool()
def arrangement_list_clips(path: str) -> str:
    """List clips in a track's Arrangement, e.g. path "live_set tracks 0"."""
    return json.dumps(_request("arrangement_list", {"path": path}), indent=2)


@mcp.tool()
def arrangement_create_clip(path: str, start_time: float,
                            length: float = 4.0, kind: str = "midi",
                            file_path: str | None = None) -> str:
    """Create a clip directly in the Arrangement view. Undoable.

    path       - track path, e.g. "live_set tracks 0"
    start_time - position in beats
    kind       - "midi" (uses length) or "audio" (requires file_path)
    """
    p: dict[str, Any] = {"path": path, "start_time": start_time,
                         "length": length, "kind": kind}
    if file_path:
        p["file_path"] = file_path
    return json.dumps(_request("arrangement_create_clip", p), indent=2)


@mcp.tool()
def arrangement_duplicate_clip(path: str, clip: str,
                               destination_time: float) -> str:
    """Copy a session clip into the Arrangement. Undoable.

    path  - track path, e.g. "live_set tracks 0"
    clip  - source clip path, e.g. "live_set tracks 0 clip_slots 0 clip"
    """
    return json.dumps(_request("arrangement_duplicate_clip",
                               {"path": path, "clip": clip,
                                "destination_time": destination_time}), indent=2)


# ------------------------------------------------------------- warp markers


@mcp.tool()
def clip_set_warp_markers(path: str, markers: list[list[float]],
                          warp_mode: int | None = None) -> str:
    """Replace an audio clip's warp map, turning warping on. Undoable.

    path    - audio clip path, e.g. "live_set tracks 0 clip_slots 0 clip"
    markers - [[beat_time, sample_time], ...], at least 2, beat_times strictly
              increasing. beat_time is beats from the sample start;
              sample_time is SECONDS from the sample start, NOT frames -
              do not scale by the sample rate.

    Cannot be done with lom_call: Live wants a C++ WarpMarker object, which
    JSON cannot express, so it must be built inside Live.

    Existing markers the new map does not occupy are removed. A fresh clip
    carries a marker Live refuses to delete, so `remove_failed` may be 1.
    """
    p: dict[str, Any] = {"path": path, "markers": markers}
    if warp_mode is not None:
        p["warp_mode"] = warp_mode
    return json.dumps(_request("warp_markers_set", p), indent=2)


@mcp.tool()
def lom_search(query: str | None = None, type: str | None = None,
               root: str = "live_set", max_results: int = 50,
               max_depth: int = 6) -> str:
    """Find LOM paths by name and/or type, without walking the tree by hand.

    query - substring of the object's name, case-insensitive ("bass", "drums")
    type  - substring of the type ("Track", "Clip", "DeviceParameter")

    Give at least one. Examples:
      lom_search(query="bass")                -> live_set tracks 3
      lom_search(type="DeviceParameter")      -> every automatable parameter

    Check `truncated` - the walk is bounded so it cannot freeze Live's UI.
    The browser is deliberately excluded (6000+ samples); use browser_list.
    """
    p: dict[str, Any] = {"root": root, "max_results": max_results,
                         "max_depth": max_depth}
    if query:
        p["query"] = query
    if type:
        p["type"] = type
    return json.dumps(_request("search", p), indent=2)


@mcp.tool()
def lom_canonical_path(path: str) -> str:
    """Resolve an alias path to where the object actually lives.

    Many paths point at the same object: "live_set view selected_track" is
    whichever track is selected right now. This returns the stable form
    ("live_set tracks 3") which is what you want to store or reuse.
    `is_alias` says whether the input was one.
    """
    return json.dumps(_request("canonical_path", {"path": path}), indent=2)


# ------------------------------------------------------ batch + transaction


@mcp.tool()
def lom_get_batch(specs: list[dict[str, Any]]) -> str:
    """Read many properties in ONE round trip instead of N calls.

    specs: [{"path": "live_set", "property": "tempo"},
            {"path": "live_set tracks 0", "property": "name"}, ...]

    Never fails as a whole - each result carries its own ok/error, so one
    unavailable property does not lose the other reads.
    """
    return json.dumps(_request("get_batch", {"specs": specs}), indent=2)


@mcp.tool()
def lom_set_batch(specs: list[dict[str, Any]],
                  stop_on_error: bool = True) -> str:
    """Write many properties inside one undo step.

    specs: [{"path": ..., "property": ..., "value": ...}, ...]
    See lom_transaction for the undo-grouping caveat.
    """
    return json.dumps(_request("set_batch", {"specs": specs,
                                             "stop_on_error": stop_on_error}),
                      indent=2)


@mcp.tool()
def lom_transaction(ops: list[dict[str, Any]], stop_on_error: bool = True,
                    rollback_on_error: bool = False) -> str:
    """Run several ops as one undo step, so the user gets one Cmd-Z rather than N.

    ops: [{"op": "set", "path": ..., "property": ..., "value": ...},
          {"op": "call", "path": ..., "function": ..., "args": [...]}]

    UNDO CAVEAT (measured on Live 12.2.7, not assumed): grouping does NOT cover
    automatable parameters. tempo, mixer volume, mute and device parameters
    each form their own undo step in Live even inside an explicit one. Track
    name, colour and time signature do group. The `undo` field in the result
    restates this.

    NOT a database transaction - Live has no intra-step rollback, so if op 5
    fails, ops 1-4 have already applied. `rollback_on_error` defaults to FALSE
    on purpose: it is a mutation on an error path, and if the step captured
    nothing it would revert whatever the user did beforehand. Prefer to
    inspect the result and let the user press Cmd-Z.
    """
    return json.dumps(_request("transaction",
                               {"ops": ops, "stop_on_error": stop_on_error,
                                "rollback_on_error": rollback_on_error}),
                      indent=2)


# --------------------------------------------------------------- observers
# MCP has no server->client push, so this is a pull with a ring buffer behind
# it: Live accumulates change events, you collect them on demand.


@mcp.tool()
def lom_observe(path: str, property: str) -> str:
    """Watch a property and record every change Live makes to it - including
    changes the USER makes in the GUI, not just ones made through this server.

    Then call lom_poll_events to collect them.

    Not every property is observable. If there is no add_<property>_listener,
    this says so; lom_describe lists what an object actually has.
    Example: lom_observe("live_set", "tempo")
    """
    return json.dumps(_request("observe_add", {"path": path,
                                               "property": property}), indent=2)


@mcp.tool()
def lom_unobserve(path: str, property: str) -> str:
    """Stop watching one property.

    Returns `outcome`: "removed", or "object_gone" if the underlying track/clip
    was deleted (harmless - Live discarded its listeners with the object).
    """
    return json.dumps(_request("observe_remove", {"path": path,
                                                  "property": property}),
                      indent=2)


@mcp.tool()
def lom_observers() -> str:
    """List active observers, buffered event count, and whether each observed
    object is still valid (`object_valid: false` means it was deleted)."""
    return json.dumps(_request("observe_list"), indent=2)


@mcp.tool()
def lom_unobserve_all() -> str:
    """Remove every observer. Reports `leaked` - the number that could NOT be
    detached and are still firing inside Live. Should always be 0."""
    return json.dumps(_request("observe_clear"), indent=2)


@mcp.tool()
def lom_poll_events(since: int | None = None, limit: int = 500,
                    consume: bool = False) -> str:
    """Collect change events recorded by lom_observe.

    Pass `since` = the previous `latest_seq` to get only new events; that is
    cheaper and avoids re-reading. `consume=true` empties the buffer instead.

    Check `dropped_events`: the buffer holds 2000 events and drops oldest
    first, so a nonzero value means changes were lost between polls.
    """
    p: dict[str, Any] = {"limit": limit, "consume": consume}
    if since is not None:
        p["since"] = since
    return json.dumps(_request("observe_poll", p), indent=2)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
