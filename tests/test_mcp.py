#!/usr/bin/env python3
"""Unit tests for the MCP tool layer. Runs WITHOUT Ableton Live.

The socket is stubbed, so this covers the part smoke.py cannot reach: how each
tool maps its arguments onto a wire op, and how transport errors are
translated. A bug here - a mistyped op name, a dropped optional parameter -
passes every integration test, because smoke.py talks to the socket directly
and never loads this module.

  ./tests/test_mcp.py        run
  ./tests/test_mcp.py -v     list each assertion
"""
import asyncio
import json
import pathlib
import socket
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import mcp_server.server as S  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

VERBOSE = "-v" in sys.argv
FAILURES = []
CALLS = []
REAL_REQUEST = S._request      # main() stubs S._request; the transport tests
                               # need the genuine one back.


def check(name, cond, detail=""):
    if cond:
        if VERBOSE:
            print("  ok   %s" % name)
    else:
        FAILURES.append("%s %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))


def fake_request(op, params=None):
    CALLS.append((op, params or {}))
    return {"echo": op}


def call(fn, *a, **kw):
    """Invoke a tool and return (op, params) it would have sent."""
    del CALLS[:]
    out = fn(*a, **kw)
    json.loads(out)  # every tool must return valid JSON
    return CALLS[-1] if CALLS else (None, None)


# --------------------------------------------------------------- op mapping

EXPECTED_OP = {
    "lom_describe": "describe", "lom_get": "get", "lom_set": "set",
    "lom_call": "call", "lom_count": "count", "lom_types": "types",
    "lom_ping": "ping", "lom_search": "search",
    "lom_canonical_path": "canonical_path",
    "lom_get_batch": "get_batch", "lom_set_batch": "set_batch",
    "lom_transaction": "transaction",
    "clip_get_notes": "notes_get", "clip_add_notes": "notes_add",
    "clip_modify_notes": "notes_modify", "clip_remove_notes": "notes_remove",
    "browser_list": "browser_list", "browser_load": "browser_load",
    "clip_envelope_get": "envelope_get",
    "clip_envelope_insert_step": "envelope_insert_step",
    "clip_envelope_clear": "envelope_clear",
    "arrangement_list_clips": "arrangement_list",
    "arrangement_create_clip": "arrangement_create_clip",
    "arrangement_duplicate_clip": "arrangement_duplicate_clip",
    "clip_set_warp_markers": "warp_markers_set",
    "lom_observe": "observe_add", "lom_unobserve": "observe_remove",
    "lom_observers": "observe_list", "lom_unobserve_all": "observe_clear",
    "lom_poll_events": "observe_poll",
}

MINIMAL_ARGS = {
    "lom_get": ("live_set", "tempo"), "lom_set": ("live_set", "tempo", 120),
    "lom_call": ("live_set", "start_playing"),
    "lom_count": ("live_set", "tracks"),
    "lom_search": ("bass",), "lom_canonical_path": ("live_set",),
    "lom_get_batch": ([{"path": "live_set", "property": "tempo"}],),
    "lom_set_batch": ([{"path": "live_set", "property": "tempo",
                        "value": 1}],),
    "lom_transaction": ([{"op": "set", "path": "live_set",
                          "property": "tempo", "value": 1}],),
    "clip_get_notes": ("clip",), "clip_add_notes": ("clip", []),
    "clip_modify_notes": ("clip", []), "clip_remove_notes": ("clip",),
    "browser_list": (), "browser_load": ("instruments/Drift",),
    "clip_envelope_get": ("clip", "param"),
    "clip_envelope_insert_step": ("clip", "param", 0.0, 1.0, 0.5),
    "clip_envelope_clear": ("clip",),
    "arrangement_list_clips": ("live_set tracks 0",),
    "arrangement_create_clip": ("live_set tracks 0", 0.0),
    "arrangement_duplicate_clip": ("live_set tracks 0", "clip", 0.0),
    "clip_set_warp_markers": ("clip", [[0.0, 0.0], [4.0, 2.0]]),
    "lom_observe": ("live_set", "tempo"),
    "lom_unobserve": ("live_set", "tempo"),
    "lom_observers": (), "lom_unobserve_all": (), "lom_poll_events": (),
    "lom_describe": (), "lom_types": (), "lom_ping": (),
}


def test_every_tool_maps_to_the_right_op():
    """A mistyped op name is invisible to the integration tests."""
    missing = [n for n in EXPECTED_OP if not hasattr(S, n)]
    check("all expected tools exist", not missing, missing)
    for name, want in sorted(EXPECTED_OP.items()):
        fn = getattr(S, name, None)
        if fn is None:
            continue
        op, _ = call(fn, *MINIMAL_ARGS.get(name, ()))
        check("%s -> %s" % (name, want), op == want, "got %r" % op)


def test_tool_count_matches_registry():
    import asyncio
    tools = asyncio.run(S.mcp.list_tools())
    check("every registered tool is covered here",
          {t.name for t in tools} == set(EXPECTED_OP),
          "registry-only: %s" % ({t.name for t in tools} - set(EXPECTED_OP)))
    check("no tool has an empty description",
          all((t.description or "").strip() for t in tools))


# ------------------------------------------------------- parameter handling

def test_optional_params_omitted_when_none():
    """Sending time_span=None would override the engine's whole-clip default."""
    _, p = call(S.clip_get_notes, "clip")
    check("clip_get_notes omits time_span when unset", "time_span" not in p, p)
    _, p = call(S.clip_get_notes, "clip", time_span=4.0)
    check("clip_get_notes forwards time_span when given",
          p.get("time_span") == 4.0, p)

    _, p = call(S.browser_load, "instruments/Drift")
    check("browser_load omits track_index when unset", "track_index" not in p, p)
    _, p = call(S.browser_load, "instruments/Drift", track_index=2)
    check("browser_load forwards track_index", p.get("track_index") == 2, p)

    _, p = call(S.lom_poll_events)
    check("lom_poll_events omits since when unset", "since" not in p, p)
    _, p = call(S.lom_poll_events, since=7)
    check("lom_poll_events forwards since", p.get("since") == 7, p)

    _, p = call(S.clip_set_warp_markers, "clip", [[0.0, 0.0], [4.0, 2.0]])
    check("warp_markers omits warp_mode when unset", "warp_mode" not in p, p)
    _, p = call(S.clip_set_warp_markers, "clip", [[0.0, 0.0], [4.0, 2.0]],
                warp_mode=1)
    check("warp_markers forwards warp_mode", p.get("warp_mode") == 1, p)

    # offset=0 is falsy but meaningful - a truthiness test would drop it and
    # silently return the unwindowed summary instead of the first page.
    _, p = call(S.lom_get, "live_set", "tracks")
    check("lom_get omits offset/limit when unset",
          "offset" not in p and "limit" not in p, p)
    _, p = call(S.lom_get, "live_set", "tracks", offset=0, limit=10)
    check("lom_get forwards offset=0", p.get("offset") == 0, p)
    check("lom_get forwards limit", p.get("limit") == 10, p)
    _, p = call(S.lom_call, "live_set tracks 0 devices 0",
                "get_parameter_names", offset=0)
    check("lom_call forwards offset=0", p.get("offset") == 0, p)

    _, p = call(S.clip_envelope_clear, "clip")
    check("envelope_clear omits parameter -> clears all",
          "parameter" not in p, p)
    _, p = call(S.clip_envelope_clear, "clip", parameter="live_set tracks 0")
    check("envelope_clear forwards parameter",
          p.get("parameter") == "live_set tracks 0", p)


def test_search_requires_a_criterion():
    _, p = call(S.lom_search, query="bass")
    check("search forwards query", p.get("query") == "bass", p)
    check("search omits empty type", "type" not in p, p)
    _, p = call(S.lom_search, type="Track")
    check("search forwards type", p.get("type") == "Track", p)
    check("search omits empty query", "query" not in p, p)


def test_destructive_default_is_safe():
    _, p = call(S.lom_call, "live_set", "delete_track", [0])
    check("lom_call defaults confirm to False", p.get("confirm") is False, p)
    _, p = call(S.lom_transaction, [], rollback_on_error=False)
    check("transaction defaults rollback_on_error to False",
          p.get("rollback_on_error") is False, p)
    _, p = call(S.lom_transaction, [])
    check("transaction rollback default is False even when unspecified",
          p.get("rollback_on_error") is False, p)


def test_args_default_to_empty_list():
    _, p = call(S.lom_call, "live_set", "start_playing")
    check("lom_call sends [] not None for args", p.get("args") == [], p)


# ------------------------------------------------------ error translation

class _Sock:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc

    def sendall(self, _b):
        if self.exc:
            raise self.exc

    def recv(self, _n):
        if self.exc:
            raise self.exc
        out, self.payload = self.payload, b""
        return out

    def close(self):
        pass


def test_error_translation():
    """Exercises the real _request against a stubbed socket."""
    real = S.socket.create_connection
    stub, S._request = S._request, REAL_REQUEST
    try:
        _error_translation_body(real)
    finally:
        S._request = stub
        S.socket.create_connection = real


def _error_translation_body(real):

    def refuse(*_a, **_k):
        raise ConnectionRefusedError()
    S.socket.create_connection = refuse
    try:
        S._request("ping")
        check("refused connection raises LiveError", False, "no raise")
    except S.LiveError as e:
        check("refused connection names the fix",
              "Control Surface" in str(e), str(e)[:60])
    except Exception as e:
        check("refused connection raises LiveError", False, type(e).__name__)
    finally:
        S.socket.create_connection = real

    S.socket.create_connection = lambda *a, **k: _Sock(exc=socket.timeout())
    try:
        S._request("ping")
        check("timeout raises LiveError", False, "no raise")
    except S.LiveError as e:
        check("timeout mentions the deadline", "did not respond" in str(e))
    except Exception as e:
        check("timeout raises LiveError", False, type(e).__name__)
    finally:
        S.socket.create_connection = real

    err = json.dumps({"ok": False, "error": {"type": "PermissionError",
                                             "message": "nope"}}).encode()
    S.socket.create_connection = lambda *a, **k: _Sock(payload=err + b"\n")
    try:
        S._request("call")
        check("engine error raises LiveError", False, "no raise")
    except S.LiveError as e:
        check("engine error preserves type and message",
              "PermissionError" in str(e) and "nope" in str(e), str(e)[:60])
    finally:
        S.socket.create_connection = real

    good = json.dumps({"ok": True, "result": {"v": 1}}).encode()
    S.socket.create_connection = lambda *a, **k: _Sock(payload=good + b"\n")
    try:
        check("success unwraps result", S._request("ping") == {"v": 1})
    finally:
        S.socket.create_connection = real


def test_errors_reach_the_client():
    """Through the real MCP dispatch, not _request alone. mcp 2.x replaces the
    message of any exception that is not a ToolError with a bare "Error
    executing tool <name>" (since after 2.0.0), so a diagnosis that _request gets right can still
    never reach the model."""
    real = S.socket.create_connection
    stub, S._request = S._request, REAL_REQUEST

    def refuse(*_a, **_k):
        raise ConnectionRefusedError()

    cases = [
        ("refused connection", refuse, "Control Surface"),
        ("reset mid-reply",
         lambda *a, **k: _Sock(exc=ConnectionResetError()), "connection"),
        ("malformed reply",
         lambda *a, **k: _Sock(payload=b"not json\n"), "malformed"),
        ("reply not an object",
         lambda *a, **k: _Sock(payload=b"[]\n"), "not an object"),
        ("error not an object",
         lambda *a, **k: _Sock(payload=b'{"ok": false, "error": "boom"}\n'),
         "boom"),
        ("connect timeout",
         lambda *a, **k: (_ for _ in ()).throw(socket.timeout()),
         "did not accept a connection"),
        ("engine error",
         lambda *a, **k: _Sock(payload=json.dumps(
             {"ok": False, "error": {"type": "KeyError", "message": "no such path"}}
         ).encode() + b"\n"), "no such path"),
    ]
    try:
        for name, conn, needle in cases:
            S.socket.create_connection = conn
            try:
                asyncio.run(S.mcp.call_tool("lom_ping", {}))
                check("%s raises" % name, False, "no raise")
            except Exception as e:
                check("%s message reaches the client" % name,
                      needle in str(e), "%s: %s" % (type(e).__name__, e))
                # The message check alone passes on mcp 2.0.0 even when
                # LiveError is a plain RuntimeError: only later releases
                # mask. The cause's type is what decides it on every 2.x.
                check("%s raised as a ToolError" % name,
                      isinstance(e.__cause__, ToolError),
                      type(e.__cause__).__name__)
    finally:
        S._request = stub
        S.socket.create_connection = real


def main():
    S._request = fake_request
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("MCP layer unit tests (%d groups, no Live required)" % len(tests))
    for t in tests:
        print("- %s" % t.__name__)
        try:
            t()
        except Exception as e:
            FAILURES.append("%s raised %s: %s" % (t.__name__, type(e).__name__, e))
            print("  ERROR %s: %s" % (type(e).__name__, e))
    print()
    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
