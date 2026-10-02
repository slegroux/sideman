"""Python client for Sideman's wire protocol - the notebook/REPL way in.

Three clients speak this protocol: the MCP server (for models), scripts/lomcli.py
(for the shell), and this module (for Python). Same engine, same paths, same
vocabulary - see docs/PROTOCOL.md.

    from sideman import Live
    live = Live()
    live.ping()
    live.describe("live_set")
    live.get("live_set", "tempo")
    live.call("live_set", "create_midi_track", [-1])

Deliberately verbs-only. A Pythonic object layer (live.tracks[0].volume = ...)
would recreate the hand-written-wrapper trap this project exists to escape:
coverage bounded by human labour, stale each Live release. The path grammar is
the interface; anything Live has is reachable through these verbs.

Paths are space-separated, zero-indexed, unquoted:
    live_set tracks 0 mixer_device volume
Roots: live_set, live_app, app_view. Use describe() to discover what exists -
never guess member names.
"""
from __future__ import annotations

import json
import socket
from typing import Any

HOST, PORT = "127.0.0.1", 9878
TIMEOUT = 25.0


class LiveError(RuntimeError):
    """A refusal from Live or the engine. `type` keeps the Python exception
    class name from inside Live, which is load-bearing: PermissionError is the
    destructive-call guard, AttributeError means the member does not exist."""

    def __init__(self, type_: str, message: str):
        super().__init__("%s: %s" % (type_, message))
        self.type = type_


class Live:
    """One Ableton Live instance, addressed by the LOM path grammar.

    Connects per request, like the MCP layer: there is no socket to go stale
    between two notebook cells run minutes apart.
    """

    def __init__(self, host: str = HOST, port: int = PORT,
                 timeout: float = TIMEOUT):
        self.host, self.port, self.timeout = host, port, timeout

    def request(self, op: str, params: dict[str, Any] | None = None) -> Any:
        """Send one wire-protocol op and return its unwrapped result.

        The escape hatch: every named method below is one line of sugar over
        this, and any op docs/PROTOCOL.md lists works here by name.
        """
        try:
            s = socket.create_connection((self.host, self.port),
                                         timeout=self.timeout)
        except ConnectionRefusedError:
            raise LiveError(
                "ConnectionRefused",
                "nothing listening on %s:%d. Ableton Live must be running "
                "with 'AbletonLOM' selected as a Control Surface in "
                "Preferences > Link, Tempo & MIDI." % (self.host, self.port))
        try:
            s.sendall(json.dumps({"id": "py", "op": op,
                                  "params": params or {}}).encode() + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(65536)
                if not chunk:
                    raise LiveError("ConnectionClosed",
                                    "Live closed the connection mid-reply")
                buf += chunk
        except socket.timeout:
            raise LiveError("Timeout",
                            "Live did not respond within %.1fs" % self.timeout)
        finally:
            s.close()

        resp = json.loads(buf.split(b"\n", 1)[0].decode())
        if not resp.get("ok"):
            err = resp.get("error") or {}
            raise LiveError(err.get("type", "Error"),
                            err.get("message", str(resp)))
        return resp.get("result")

    # ------------------------------------------------------- generic verbs

    def ping(self):
        """Check the connection and whether the engine loaded.

        Answered by the shell rather than the engine, so it still replies when
        handlers.py failed to import - the load error is in the result."""
        return self.request("ping")

    def reload(self):
        """Hot-reload the engine after editing handlers.py. No Live restart.

        Tears every observer down first, so anything being watched - by any
        client - is no longer watched afterwards."""
        return self.request("reload")

    def describe(self, path: str = "live_set", include_values: bool = True):
        """Discover what exists at a path instead of guessing. Start here.

        Returns `properties` (with current values unless include_values is
        False), `children` with their counts, `functions`, and `unavailable`:
        members the type has that THIS instance does not."""
        return self.request("describe", {"path": path,
                                         "include_values": include_values})

    def get(self, path: str, property: str, **params):
        """Read one property. Returns {path, property, value, type}.

        A list longer than 64 reports only {"__vector__": true, "count": N};
        optional offset/limit page through one, 512 per page at most, and the
        window adds `items`, `returned` and `truncated`."""
        return self.request("get", dict(params, path=path, property=property))

    def set(self, path: str, property: str, value):
        """Write one property, inside a native undo step. Returns the value
        read back, which is not always the value sent: Live stores parameter
        values as 32-bit floats.

        For a property holding a Live object rather than a scalar, pass
        {"__path__": "<lom path>"} as the value."""
        return self.request("set", {"path": path, "property": property,
                                    "value": value})

    def call(self, path: str, function: str, args: list | None = None,
             confirm: bool = False, **params):
        """Call a function on a Live object, inside a native undo step.
        Returns {path, function, result}.

        {"__path__": "<lom path>"} anywhere in args names a Live object by
        path; a plain string is never reinterpreted as one. Destructive
        functions (delete_*, remove_*, clear_*, crop) raise
        LiveError("PermissionError") unless confirm=True. Optional
        offset/limit page a long return value."""
        return self.request("call", dict(params, path=path, function=function,
                                         args=args or [], confirm=confirm))

    def count(self, path: str, child: str):
        """Length of a Live collection. Returns {path, child, count}."""
        return self.request("count", {"path": path, "child": child})

    def types(self):
        """Census of every LOM type Live exposes and every member of each.

        The whole addressable surface, and large - this is the coverage
        baseline, not a navigation tool. Use describe() to get around."""
        return self.request("types")

    def search(self, query: str | None = None, type: str | None = None,
               root: str = "live_set", **params):
        """Find paths by name and/or type, both matched as case-insensitive
        substrings. Give at least one. Returns {count, nodes_visited,
        truncated, results: [{path, type, name}]}.

        The walk is bounded - optional max_results, max_depth, max_nodes - so
        it cannot freeze Live's main thread; check `truncated`. The browser is
        excluded, because `samples` alone has thousands of children."""
        p = dict(params, root=root)
        if query is not None:
            p["query"] = query
        if type is not None:
            p["type"] = type
        return self.request("search", p)

    def canonical_path(self, path: str):
        """Resolve an alias path to where the object actually lives.

        "live_set view selected_track" is whichever track is selected right
        now; this returns the stable form, "live_set tracks 3", which is what
        to store or reuse. `is_alias` says whether the input was one."""
        return self.request("canonical_path", {"path": path})

    def get_batch(self, specs: list):
        """Read many properties in one round trip.

        specs: [{"path": ..., "property": ...}, ...]. Returns {count,
        ok_count, results}; each result carries its own ok/error, so one
        unavailable property does not lose the other reads."""
        return self.request("get_batch", {"specs": specs})

    def set_batch(self, specs: list, **params):
        """Write many properties inside one undo step.

        specs: [{"path": ..., "property": ..., "value": ...}, ...]. Optional
        stop_on_error (default true) and rollback_on_error (default false).
        See transaction() for the undo-grouping caveat."""
        return self.request("set_batch", dict(params, specs=specs))

    def transaction(self, ops: list, **params):
        """Run several ops as one undo step, so the user gets one Cmd-Z.

        ops: [{"op": "set", "path": ..., "property": ..., "value": ...},
              {"op": "call", "path": ..., "function": ..., "args": [...]}]

        Grouping does not cover automatable parameters: tempo, mixer volume,
        mute and device parameters each form their own undo step in Live even
        inside an explicit one. Not a database transaction either - Live has
        no intra-step rollback, so if op 5 fails, ops 1-4 have applied."""
        return self.request("transaction", dict(params, ops=ops))

    # ------------------------- typed ops (JSON cannot express these shapes)

    def notes_get(self, path: str, **window):
        """Read a MIDI clip's notes. Returns {path, count, notes}, each note
        carrying note_id, pitch, start_time, duration, velocity, mute,
        probability, velocity_deviation and release_velocity.

        Times are in beats, pitch is a MIDI number (60 = C3). The optional
        window - from_pitch, pitch_span, from_time, time_span - defaults to
        the whole clip. Keep the note_ids; notes_modify() needs them."""
        return self.request("notes_get", dict(window, path=path))

    def notes_add(self, path: str, notes: list):
        """Add notes to a MIDI clip. Undoable. Returns {path, added}.

        Each note: {"pitch": 60, "start_time": 0.0, "duration": 1.0,
        "velocity": 100, "mute": false}. Times are in beats."""
        return self.request("notes_add", {"path": path, "notes": notes})

    def notes_modify(self, path: str, notes: list):
        """Edit existing notes in place. Undoable. Returns {path, modified,
        unmatched_note_ids} - ids that no longer exist are reported rather
        than failing the call.

        Each entry needs "note_id" from notes_get() plus the fields to
        change, e.g. {"note_id": 3, "pitch": 62, "velocity": 80}."""
        return self.request("notes_modify", {"path": path, "notes": notes})

    def notes_remove(self, path: str, **window):
        """Remove notes in a pitch/time window. Undoable. Returns {path,
        removed}. The window is the same as notes_get()'s and defaults to the
        whole clip, so no window removes every note."""
        return self.request("notes_remove", dict(window, path=path))

    def browser_list(self, path: str = ""):
        """Browse Live's library. Returns {path, item, items}.

        An empty path lists the roots - instruments, sounds, drums,
        audio_effects, midi_effects, plugins, clips, samples, packs,
        user_library, current_project, max_for_live - and everything below
        descends by name with "/"."""
        return self.request("browser_list", {"path": path})

    def browser_load(self, path: str, track_index: int | None = None):
        """Load a browser item onto a track. Undoable. Returns {loaded,
        track}.

        A device lands at the END of the track's device chain; a sample lands
        in the highlighted clip slot of the selected track, and only while
        Live is showing the Session view. track_index selects the track and
        decides nothing else."""
        p: dict[str, Any] = {"path": path}
        if track_index is not None:
            p["track_index"] = track_index
        return self.request("browser_load", p)

    def envelope_get(self, path: str, parameter: str, **params):
        """Sample a clip's automation envelope for one parameter. Both
        arguments are paths: a clip, and the DeviceParameter or mixer control
        it automates.

        Returns {exists, parameter_name, min, max, points: [{time, value}]},
        sampled at `times` or at `samples` points across the clip (8 by
        default). Live evaluates a step edge as the end of the step before it,
        so sample inside a step rather than on its boundary."""
        return self.request("envelope_get",
                            dict(params, path=path, parameter=parameter))

    def envelope_insert_step(self, path: str, parameter: str, time: float,
                             length: float, value: float):
        """Write a flat automation step, creating the envelope if the clip has
        none. Undoable. Time and length are in beats; value is in the
        parameter's own units, so read its min/max first."""
        return self.request("envelope_insert_step",
                            {"path": path, "parameter": parameter,
                             "time": time, "length": length, "value": value})

    def envelope_clear(self, path: str, parameter: str | None = None):
        """Clear one parameter's envelope, or every envelope on the clip if
        parameter is omitted. Undoable."""
        p: dict[str, Any] = {"path": path}
        if parameter is not None:
            p["parameter"] = parameter
        return self.request("envelope_clear", p)

    def arrangement_list(self, path: str):
        """List a track's Arrangement clips. Returns {path, count, clips},
        each clip carrying index, name, start_time, end_time and is_midi.
        Times are in beats."""
        return self.request("arrangement_list", {"path": path})

    def arrangement_create_clip(self, path: str, start_time: float, **params):
        """Create a clip directly in the Arrangement, from nothing or from a
        file. Undoable. start_time is in beats.

        kind="midi" (the default) requires length=, in beats; kind="audio"
        requires file_path=."""
        return self.request("arrangement_create_clip",
                            dict(params, path=path, start_time=start_time))

    def arrangement_duplicate_clip(self, path: str, clip: str,
                                   destination_time: float):
        """Copy a session clip into a track's Arrangement at a beat position.
        Undoable.

        An arrangement clip cannot be lengthened afterwards - `end_time` has
        no setter - so a long section is tiled, not resized."""
        return self.request("arrangement_duplicate_clip",
                            {"path": path, "clip": clip,
                             "destination_time": destination_time})

    def warp_markers_set(self, path: str, markers: list,
                         warp_mode: int | None = None):
        """Replace an audio clip's warp map, turning warping on. Undoable.

        markers: [[beat_time, sample_time], ...], at least two, beat_times
        strictly increasing. beat_time is beats from the sample start;
        sample_time is SECONDS from the sample start, not frames - do not
        scale by the sample rate.

        Typed because Live wants a C++ WarpMarker, which JSON cannot express.
        Existing markers the new map does not occupy are removed, except the
        one every clip carries that Live refuses to delete, so
        `remove_failed` may be 1."""
        p: dict[str, Any] = {"path": path, "markers": markers}
        if warp_mode is not None:
            p["warp_mode"] = warp_mode
        return self.request("warp_markers_set", p)

    # ----------------------------------------------------------- observers

    def observe(self, path: str, property: str):
        """Watch a property and record every change Live makes to it,
        including changes the user makes by hand in the GUI. Collect them
        with poll_events().

        Not every property is observable; describe() lists what an object
        actually has. The registry lives inside Live and is shared by every
        client connected to it."""
        return self.request("observe_add", {"path": path, "property": property})

    def unobserve(self, path: str, property: str):
        """Stop watching one property. `outcome` is "removed", or
        "object_gone" if the track or clip was deleted first - harmless, Live
        discarded its listeners with the object."""
        return self.request("observe_remove",
                            {"path": path, "property": property})

    def observers(self):
        """List active observers, the buffered and dropped event counts, and
        whether each observed object is still valid."""
        return self.request("observe_list")

    def unobserve_all(self):
        """Remove every observer, including ones another client created.
        Reports `leaked`: listeners that could not be detached and are still
        firing inside Live. Should always be 0."""
        return self.request("observe_clear")

    def poll_events(self, **params):
        """Collect the change events observe() recorded, oldest first. Returns
        {count, truncated, reset, next_since, latest_seq, dropped_events,
        active_listeners, events}.

        The buffer is shared, so start from the latest_seq observe() returned,
        then pass since=<the previous next_since> for only what is new. When
        truncated, more are waiting: poll again with since=next_since.
        reset means Live restarted the sequence, so the page starts over.
        Optional limit (500 by default) and consume, which removes the
        returned events from the buffer.
        It holds 2000 events and drops the oldest first, so a nonzero
        `dropped_events` means changes were lost between polls."""
        return self.request("observe_poll", params)
