"""AbletonLOM MCP server - generic Live Object Model access.

Unlike every other Ableton MCP server, this does not hand-wrap individual
properties. It exposes the Live Object Model itself, so any property Live has
is reachable without shipping new code.
"""
from __future__ import annotations

import json
import os
import socket
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

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
    "sideman",
    instructions=(
        "Generic access to Ableton Live's Object Model. There is no fixed tool "
        "per feature - navigate the object graph instead.\n\n"
        + PATH_HELP
        + "\n\nWorkflow: lom_describe to discover, then lom_get/lom_set/lom_call. "
        "Availability is per-instance: a member listed for a type may still be "
        "unavailable on a given object (see the `unavailable` map)."
    ),
)


class LiveError(ToolError):
    """Every failure a tool reports. It must be a ToolError: the MCP SDK (after
    2.0.0) treats any other exception as a crash and sends the client only "Error executing
    tool <name>", dropping the diagnosis."""


def _request(op: str, params: dict[str, Any] | None = None) -> Any:
    try:
        s = socket.create_connection((HOST, PORT), timeout=TIMEOUT)
    except ConnectionRefusedError:
        raise LiveError(
            f"Nothing listening on {HOST}:{PORT}. Ableton Live must be running "
            "with 'AbletonLOM' selected as a Control Surface in "
            "Preferences > Link, Tempo & MIDI."
        )
    except socket.timeout:
        raise LiveError(f"Live did not accept a connection on {HOST}:{PORT} "
                        f"within {TIMEOUT}s; its UI may be blocked by a dialog")
    except OSError as e:
        raise LiveError(f"cannot connect to Live on {HOST}:{PORT}: {e!r}")
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
    except OSError as e:
        raise LiveError(f"connection to Live lost mid-reply: {e!r}")
    finally:
        s.close()

    line = buf.split(b"\n", 1)[0]
    try:
        resp = json.loads(line.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LiveError(f"malformed reply from Live ({e}): {line[:200]!r}")
    if not isinstance(resp, dict):
        raise LiveError(f"malformed reply from Live (not an object): {line[:200]!r}")
    if not resp.get("ok"):
        err = resp.get("error") or {}
        if not isinstance(err, dict):
            raise LiveError(f"Error: {err}")
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
             limit: int | None = None,
             expect: str | dict[str, Any] | None = None) -> str:
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

    When the call returns a Live object that sits directly under `path` (a new
    track, a new clip), `result_path` gives its path, e.g.
    create_midi_track -> "live_set tracks 5".

    Calls that remove or overwrite work (delete_*, remove_*, clear_*, crop,
    set_notes, Looper clear, a clip copied onto a full slot, ...) refuse to run
    without `expect`: the name of what they will remove, as you just read it.
    It is checked inside Live before the call, so an index that shifted (a
    track inserted above) is refused with what is there now instead of
    deleting the wrong thing:

        lom_call("live_set", "delete_track", [3], expect="Bass")

    The note removers need the clip's note count too, the `total` that
    clip_get_notes reports: expect={"name": "Verse", "note_count": 16}. A name
    that is empty or Live's positional default ("3-MIDI", "A-Reverb")
    renumbers when tracks move, so it is refused on its own: rename the
    target, or add its id, expect={"id": <lom_get(path, "_live_ptr")>}.
    The result's `target` says what was removed. confirm=True alone still
    runs, with a warning, and never overrides a wrong expect. Ask the user
    before removing their work either way.
    """
    p: dict[str, Any] = {"path": path, "function": function,
                         "args": args or [], "confirm": confirm}
    if expect is not None:
        p["expect"] = expect
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
    Defaults cover the clip's [0, length) window; `count` is the notes read,
    `total` every note in the clip (the note_count a note remover's expect
    takes). Keep the note_ids - clip_modify_notes needs them.
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

    Returns `clip_path`, e.g. "live_set tracks 0 arrangement_clips 2".
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

    UNDO (measured, not assumed): on Live 12.4+ the whole batch is one Cmd-Z,
    tempo, volume and device parameters included. On older Live those
    automatable parameters can need extra Cmd-Z. The `undo`
    field in the result says which applies to the running Live.

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

    The buffer is shared with every earlier session, so start from the
    `latest_seq` that lom_observe returned. Then pass `since` = the previous
    `next_since` to get only new events, oldest first. When `truncated` is
    true, more are waiting: poll again with since=next_since. `reset` true
    means Live restarted the sequence: the page starts over from the oldest
    buffered event. `gap` true means events after `since` are gone (dropped
    or consumed by another client) and the page skips them. `limit` must be
    at least 1. `consume=true` removes exactly the returned events.

    Check `dropped_events`: the buffer holds 2000 events and drops oldest
    first, so a nonzero value means changes were lost between polls.
    """
    p: dict[str, Any] = {"limit": limit, "consume": consume}
    if since is not None:
        p["since"] = since
    return json.dumps(_request("observe_poll", p), indent=2)


# ------------------------------------------------------------------ listening
# These run in this process on audio files; they send nothing to Live except,
# for a clip path, one `get` of its file_path. numpy and soundfile come from the
# optional `listen` extra and are imported only when a listening tool runs.

LISTEN_HINT = ("the listening tools need the optional `listen` extra (numpy, "
               "soundfile): pip install 'sideman[listen]', or from a checkout "
               "uv pip install -e '.[listen]'")


def _listen():
    try:
        from sideman import listen
    except ImportError as e:
        raise LiveError("%s (%s)" % (LISTEN_HINT, e))
    return listen


def _oneshot():
    try:
        from sideman import oneshot
    except ImportError as e:
        raise LiveError("%s (%s)" % (LISTEN_HINT, e))
    return oneshot


def _audio_file(source: str) -> str:
    """An absolute file path, from a path or an audio clip's LOM path."""
    source = source.strip()
    if not source:
        raise LiveError("no audio source given: pass a file path or an audio clip's LOM path")
    if source.split(" ", 1)[0] in ("live_set", "live_app", "app_view"):
        got = _request("get", {"path": source, "property": "file_path"})
        path = got.get("value") if isinstance(got, dict) else got
        if not path:
            raise LiveError("%s has no file_path; only audio clips do" % source)
    else:
        path = os.path.abspath(os.path.expanduser(source))
    if not os.path.isfile(path):
        raise LiveError("no such audio file: %s" % path)
    return path


def _analyze(source: str) -> dict:
    listen = _listen()
    path = _audio_file(source)
    try:
        return listen.analyze(path)
    except MemoryError:
        raise LiveError("not enough memory to analyse %s; bounce a shorter section" % path)
    except (OSError, RuntimeError, ValueError) as e:   # LibsndfileError is a RuntimeError
        raise LiveError("cannot measure %s: %s" % (path, e))


@mcp.tool()
def audio_analyze(source: str) -> str:
    """Measure an audio file, locally: loudness, peaks, dynamics, spectrum, stereo.

    `source` is a mono or stereo WAV/AIFF/FLAC path, or an audio clip's LOM
    path such as "live_set tracks 3 clip_slots 0 clip". A clip path measures
    the clip's whole source file: its start/end, loop, warping, clip gain and
    the track's devices are not applied. Nothing leaves the machine.

    Returns integrated / short-term max / momentary max LUFS (BS.1770-4),
    true_peak_dbtp, sample_peak_db, rms_db, crest_db, centroid_hz, `bands_db`
    (mean power per band in dBFS: sub 20-60, bass 60-250, low_mid 250-500, mid
    500-2k, high_mid 2k-6k, high 6k-20k), `side_minus_mid_db` per band,
    `width` (0 mono .. 1 all side) and `low_end_correlation` (L/R below 120 Hz:
    1 is mono-safe, below 0 cancels in mono). A value that cannot be measured
    (loudness of a file under 0.4 s, short-term under 3 s, or silence) is null.

    Needs the `listen` extra; without it the tool says how to install it.
    """
    return json.dumps(_analyze(source), indent=2)


@mcp.tool()
def audio_compare(a: str, b: str) -> str:
    """Compare two audio files, `a` (usually the Set's bounce or capture) against
    `b` (usually a reference track). Each is a path or an audio clip's LOM path.

    Band balance is compared after removing the overall level offset (the
    median band difference), so a quieter mix does not read as a darker one.
    `flagged_bands` names bands still more than 3 dB away from the reference;
    `diff` gives a - b for LUFS, true peak, crest, width, low-end correlation
    and centroid. Local only; needs the `listen` extra.
    """
    listen = _listen()
    return json.dumps(listen.compare(_analyze(a), _analyze(b)), indent=2)


def _profile(source: str, hit: int | None) -> dict:
    oneshot = _oneshot()
    path = _audio_file(source)
    try:
        return oneshot.profile(path, hit)
    except MemoryError:
        raise LiveError("not enough memory to profile %s; trim it to the hits" % path)
    except (OSError, RuntimeError, ValueError) as e:   # LibsndfileError is a RuntimeError
        raise LiveError("cannot measure %s: %s" % (path, e))


@mcp.tool()
def sample_analyze(source: str, hit: int | None = None) -> str:
    """Profile a drum one-shot, locally, in the terms a synth is programmed in.

    `source` is a mono or stereo file path or an audio clip's LOM path (the
    clip's whole source file). Every measure is relative to the hit itself:
    times from its onset (the first sample within 40 dB of its peak), levels
    against its own peak or fundamental, so level, leading silence and sample
    rate do not change them. For a mix or a whole track use audio_analyze.

    A take may hold several hits: `hits` counts them, `onsets_ms` places
    them, and `hit` (from 0) picks one; by default the first that is not
    `truncated` (cut by the next hit or the file's end before it falls
    40 dB; its falls past that point are null). A hit is split off only
    when it rises 15 dB within 10 ms over a tail that has fallen that far,
    so a tremolo or a tom's beating stays one hit, and so does a fast roll.

    Envelope: peak_db (dBFS), attack_ms (onset to the envelope peak, good to
    half a period of the pitch), fall_10_ms / fall_20_ms / fall_40_ms (from the
    peak to the last moment above peak - 10/20/40 dB), duration_ms (to -60 dB).
    Pitch: `pitched` (false for noise or an inharmonic cymbal or bell: then no
    pitch and no harmonics, only `partials`), pitch_body_hz (40-150 ms, the
    fundamental: the lowest strong partial, not the loudest), pitch_start_hz
    (read about 1.5 periods in, at pitch_start_at_ms, so a glide faster than
    that reads low), pitch_drop_st (start minus body), pitch_settle_ms (when it
    comes within 0.5 st of the body pitch and stays 10 ms), pitch_track_hz at
    0, 5, 10, 20, 40, 80, 150, 300 ms.
    Tone, on the body: harmonics_db (2-8, dB against the fundamental) with
    harmonics_floor_db (the spectral floor around each), `partials` (strongest
    non-harmonic peaks, Hz and dB), noise_db (energy outside the tonal peaks,
    dB against the total) and flatness (0 tone .. 0.56 white noise) for attack
    (0-30 ms) and body, click_db (attack power above 2 kHz against the
    attack's total: a click or any bright attack), bands_db (six bands as
    audio_analyze, against the hit's total) and width (0 mono .. 1 all side).
    A value that cannot be measured is null and `notes` says why.

    Needs the `listen` extra; without it the tool says how to install it.
    """
    return json.dumps(_profile(source, hit), indent=2)


@mcp.tool()
def sample_compare(a: str, b: str, hit_a: int | None = None, hit_b: int | None = None) -> str:
    """Compare two drum hits in synth terms: `a` the candidate (usually a synth
    take), `b` the reference sample. Each is a file path or an audio clip's
    LOM path; hit_a / hit_b pick a hit from a take holding several (see
    sample_analyze; by default the first that is not cut off).

    Returns both sample_analyze profiles, `diff` (a - b: pitch in semitones,
    times as {"ms", "ratio" a/b}, levels in dB, width) and `readings`: one
    plain line per difference past its threshold, largest first, such as
    "body pitch 2.1 st below the reference" or "falls 20 dB in 80 ms,
    reference 140 ms (43 % shorter)". Thresholds: body pitch 0.3 st, pitch
    drop 1 st, attack / falls / duration / settle 15 % and 2 ms, harmonics
    and noise share 3 dB, click 6 dB, width 0.1; one side pitched and the
    other not is a reading of its own. Levels at the method's floor on both
    sides are not read (harmonics under -50 dB or near the noise, noise share
    under -25 dB, click under -40 dB). Level, leading silence and sample rate
    give no reading. Change the patch, record again, compare again until
    `readings` is empty. Local only; needs the `listen` extra.
    """
    oneshot = _oneshot()
    return json.dumps(oneshot.compare(_profile(a, hit_a), _profile(b, hit_b)), indent=2)


@mcp.tool()
def mix_report(manifest: dict[str, Any] | str, reference: str | None = None,
               out_dir: str | None = None) -> str:
    """Per-section mix report of a capture: what to fix, section by section.

    `manifest` (a dict, or the path of a JSON file) places time-aligned files
    on the Set's bar grid:
        {"tempo": 122, "beats_per_bar": 4, "offset_s": 0.0,
         "sources": {"master": "/abs/master.wav", "kick": "...", "bass": "..."},
         "sections": [{"name": "INTRO A", "start_bar": 0, "bars": 8}, ...]}
    start_bar counts from the files' bar 0 (at offset_s). "master" is required;
    a "kick" (or "drums") plus a "bass" source adds masking.

    Per section: integrated / short-term max LUFS, true peak, crest, width,
    low-end correlation, bands and side-minus-mid; with `reference` (a file
    path) the bands more than 3 dB off its balance; with kick and bass, their
    40-120 Hz overlap (0 separate or sidechained .. 1 fully stacked; flagged
    above 0.5 unless the quieter part is more than 20 dB down). Over the whole capture, the spectral change between 4-bar
    blocks, flagging runs of more than 8 bars where nothing changes.

    Writes mix_report.json and mix_report.md (default: next to the master
    file) and returns the report. Local only; needs the `listen` extra.
    """
    listen = _listen()
    ref = _audio_file(reference) if reference else None
    try:
        return json.dumps(listen.mix_report(manifest, ref, out_dir), indent=2)
    except MemoryError:
        raise LiveError("not enough memory for this capture; report fewer sections at a time")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, ArithmeticError) as e:
        raise LiveError("cannot build the mix report: %s: %s" % (type(e).__name__, e))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
