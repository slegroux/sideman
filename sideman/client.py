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
        return self.request("ping")

    def reload(self):
        """Hot-reload the engine after editing handlers.py. No Live restart."""
        return self.request("reload")

    def describe(self, path: str = "live_set", include_values: bool = True):
        return self.request("describe", {"path": path,
                                         "include_values": include_values})

    def get(self, path: str, property: str, **params):
        """Optionals: offset/limit page a long vector (docs/TOOLS.md)."""
        return self.request("get", dict(params, path=path, property=property))

    def set(self, path: str, property: str, value):
        return self.request("set", {"path": path, "property": property,
                                    "value": value})

    def call(self, path: str, function: str, args: list | None = None,
             confirm: bool = False, **params):
        """{"__path__": "<lom path>"} in args names a Live object by path.
        Destructive functions (delete_*/remove_*/clear_*/crop) need
        confirm=True."""
        return self.request("call", dict(params, path=path, function=function,
                                         args=args or [], confirm=confirm))

    def count(self, path: str, child: str):
        return self.request("count", {"path": path, "child": child})

    def types(self):
        return self.request("types")

    def search(self, query: str | None = None, type: str | None = None,
               root: str = "live_set", **params):
        p = dict(params, root=root)
        if query is not None:
            p["query"] = query
        if type is not None:
            p["type"] = type
        return self.request("search", p)

    def canonical_path(self, path: str):
        return self.request("canonical_path", {"path": path})

    def get_batch(self, specs: list):
        return self.request("get_batch", {"specs": specs})

    def set_batch(self, specs: list, **params):
        return self.request("set_batch", dict(params, specs=specs))

    def transaction(self, ops: list, **params):
        return self.request("transaction", dict(params, ops=ops))

    # ------------------------- typed ops (JSON cannot express these shapes)

    def notes_get(self, path: str, **window):
        return self.request("notes_get", dict(window, path=path))

    def notes_add(self, path: str, notes: list):
        return self.request("notes_add", {"path": path, "notes": notes})

    def notes_modify(self, path: str, notes: list):
        return self.request("notes_modify", {"path": path, "notes": notes})

    def notes_remove(self, path: str, **window):
        return self.request("notes_remove", dict(window, path=path))

    def browser_list(self, path: str = ""):
        return self.request("browser_list", {"path": path})

    def browser_load(self, path: str, track_index: int | None = None):
        p: dict[str, Any] = {"path": path}
        if track_index is not None:
            p["track_index"] = track_index
        return self.request("browser_load", p)

    def envelope_get(self, path: str, parameter: str, **params):
        return self.request("envelope_get",
                            dict(params, path=path, parameter=parameter))

    def envelope_insert_step(self, path: str, parameter: str, time: float,
                             length: float, value: float):
        return self.request("envelope_insert_step",
                            {"path": path, "parameter": parameter,
                             "time": time, "length": length, "value": value})

    def envelope_clear(self, path: str, parameter: str | None = None):
        p: dict[str, Any] = {"path": path}
        if parameter is not None:
            p["parameter"] = parameter
        return self.request("envelope_clear", p)

    def arrangement_list(self, path: str):
        return self.request("arrangement_list", {"path": path})

    def arrangement_create_clip(self, path: str, start_time: float, **params):
        return self.request("arrangement_create_clip",
                            dict(params, path=path, start_time=start_time))

    def arrangement_duplicate_clip(self, path: str, clip: str,
                                   destination_time: float):
        return self.request("arrangement_duplicate_clip",
                            {"path": path, "clip": clip,
                             "destination_time": destination_time})

    def warp_markers_set(self, path: str, markers: list,
                         warp_mode: int | None = None):
        p: dict[str, Any] = {"path": path, "markers": markers}
        if warp_mode is not None:
            p["warp_mode"] = warp_mode
        return self.request("warp_markers_set", p)

    # ----------------------------------------------------------- observers

    def observe(self, path: str, property: str):
        return self.request("observe_add", {"path": path, "property": property})

    def unobserve(self, path: str, property: str):
        return self.request("observe_remove",
                            {"path": path, "property": property})

    def observers(self):
        return self.request("observe_list")

    def unobserve_all(self):
        return self.request("observe_clear")

    def poll_events(self, **params):
        """Optionals: since=<last latest_seq>, limit, consume."""
        return self.request("observe_poll", params)
