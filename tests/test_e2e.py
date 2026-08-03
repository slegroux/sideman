#!/usr/bin/env python3
"""End-to-end: the real MCP tool functions against a real Ableton Live.

This closes the seam the other two suites leave open:

  test_mcp.py   MCP layer with a STUBBED socket   - never touches Live
  smoke.py      engine via a RAW socket           - never loads the MCP layer
  test_e2e.py   MCP layer -> socket -> Live       - the path Claude actually uses

A bug that only appears when the real layers meet - a response shape the tool
cannot serialise, an error that escapes as the wrong type - passes both other
suites. Requires Live with AbletonLOM enabled.

  ./tests/test_e2e.py        run
  ./tests/test_e2e.py -v     show each assertion
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import mcp_server.server as S  # noqa: E402

VERBOSE = "-v" in sys.argv
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        if VERBOSE:
            print("  ok   %s" % name)
    else:
        FAILURES.append("%s %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))


def jcall(fn, *a, **kw):
    """Call a tool exactly as the MCP runtime would, and parse its JSON."""
    raw = fn(*a, **kw)
    check("%s returns a str" % fn.__name__, isinstance(raw, str),
          type(raw).__name__)
    return json.loads(raw)


def test_live_is_reachable():
    r = jcall(S.lom_ping)
    check("ping via MCP layer", r.get("pong") is True, r)
    check("engine handlers loaded", r.get("handlers") is True, r)


def test_read_path():
    d = jcall(S.lom_describe, "live_set", include_values=False)
    check("describe type", d.get("type") == "Song.Song", d.get("type"))
    check("describe carries counts", "counts" in d)

    g = jcall(S.lom_get, "live_set", "tempo")
    check("get returns a numeric tempo",
          isinstance(g.get("value"), (int, float)), g.get("value"))

    c = jcall(S.lom_count, "live_set", "tracks")
    check("count returns an int", isinstance(c.get("count"), int), c)

    b = jcall(S.lom_get_batch, [{"path": "live_set", "property": "tempo"},
                                {"path": "live_set", "property": "nope_nope"}])
    check("batch isolates a bad read", b.get("ok_count") == 1, b.get("ok_count"))


def test_search_and_canonical():
    s = jcall(S.lom_search, type="Track", max_results=50)
    check("search finds tracks", s.get("count", 0) > 0, s.get("count"))
    cp = jcall(S.lom_canonical_path, "live_set view selected_track")
    check("canonical resolves an alias",
          str(cp.get("canonical_path", "")).startswith("live_set tracks "),
          cp.get("canonical_path"))


def test_large_response_serialises():
    """lom_types is the biggest payload the server produces. If any value in it
    is not JSON-serialisable, only this path catches it."""
    t = jcall(S.lom_types)
    check("types census returns types", t.get("type_count", 0) > 40,
          t.get("type_count"))
    check("types census is fully serialisable",
          isinstance(json.dumps(t), str))


def test_errors_surface_as_LiveError():
    for label, fn, args in (
        ("bad path", S.lom_get, ("live_set nonexistent 0", "name")),
        ("bad property", S.lom_get, ("live_set", "not_a_real_property")),
        ("bad op target", S.lom_call, ("live_set", "not_a_real_function")),
    ):
        try:
            fn(*args)
            check("%s raises" % label, False, "no exception")
        except S.LiveError as e:
            check("%s -> LiveError" % label, True)
            check("%s message is not empty" % label, bool(str(e).strip()))
        except Exception as e:
            check("%s -> LiveError" % label, False,
                  "got %s" % type(e).__name__)


def test_write_guard_via_mcp():
    try:
        S.lom_call("live_set", "delete_track", [0])
        check("destructive call refused through the MCP layer", False,
              "no exception")
    except S.LiveError as e:
        check("guard fires through the MCP layer", "PermissionError" in str(e),
              str(e)[:70])
    except Exception as e:
        check("guard raises LiveError", False, type(e).__name__)


def test_write_path_roundtrip():
    """The only mutation here: tempo, restored immediately."""
    before = jcall(S.lom_get, "live_set", "tempo")["value"]
    target = round(before + 1.0, 3)
    jcall(S.lom_set, "live_set", "tempo", target)
    after = jcall(S.lom_get, "live_set", "tempo")["value"]
    check("set via MCP layer took effect", abs(after - target) < 0.01,
          "%s vs %s" % (after, target))
    jcall(S.lom_set, "live_set", "tempo", before)
    restored = jcall(S.lom_get, "live_set", "tempo")["value"]
    check("tempo restored", abs(restored - before) < 0.01, restored)


def test_path_marker_resolves_to_a_live_object():
    """{"__path__": ...} args become real handles inside Live.

    Without this, every API taking an object - move_device, selected_track -
    is unreachable: a path arrives as a str and boost.python rejects the call
    with "did not match C++ signature". Selection is used as the probe because
    it is observable and costs the user nothing to restore.
    """
    def selected():
        # canonical_path, not path: `path` echoes the alias back verbatim.
        return jcall(S.lom_canonical_path,
                     "live_set view selected_track")["canonical_path"]

    before = selected()
    check("selected_track resolves to a stable path",
          isinstance(before, str) and before.startswith("live_set tracks"),
          before)

    count = jcall(S.lom_count, "live_set", "tracks")["count"]
    if count < 2:
        check("needs 2+ tracks to swap selection", False, count)
        return
    # Any track that is not the current one, so the write is observable.
    target = "live_set tracks 1" if before == "live_set tracks 0" \
        else "live_set tracks 0"

    jcall(S.lom_set, "live_set view", "selected_track", {"__path__": target})
    after = selected()
    check("marker resolved to the named object", after == target,
          "%s vs %s" % (after, target))

    jcall(S.lom_set, "live_set view", "selected_track", {"__path__": before})
    check("selection restored", selected() == before, selected())


def test_vector_window_reads_past_the_inline_cap():
    """Long vectors report only a count unless a window is asked for.

    That cap hides the lists worth reading - a plugin's get_parameter_names
    runs to thousands, and even Drift's 66 parameters clear it. Read-only, so
    it runs against whatever Set is open.
    """
    count = jcall(S.lom_count, "live_set", "tracks")["count"]
    if count < 2:
        check("needs 2+ tracks to window", False, count)
        return

    plain = jcall(S.lom_get, "live_set", "tracks")["value"]
    check("unwindowed shape unchanged",
          isinstance(plain, list) or plain.get("__vector__") is True, plain)

    w = jcall(S.lom_get, "live_set", "tracks", offset=1, limit=1)["value"]
    check("window reports the full count", w.get("count") == count, w)
    check("window returns the asked-for slice", w.get("returned") == 1, w)
    check("window carries items", len(w.get("items", [])) == 1, w)
    check("truncated flags the remainder",
          w.get("truncated") is (count > 2), w.get("truncated"))

    # offset=0 is falsy but meaningful; it must not be dropped en route.
    first = jcall(S.lom_get, "live_set", "tracks", offset=0, limit=1)["value"]
    check("offset=0 is honoured, not treated as unset",
          first.get("offset") == 0 and first.get("returned") == 1, first)

    past = jcall(S.lom_get, "live_set", "tracks", offset=count + 10, limit=5)
    check("offset past the end returns nothing, not an error",
          past["value"].get("returned") == 0, past["value"])

    big = jcall(S.lom_get, "live_set", "tracks", limit=10 ** 6)["value"]
    check("oversized limit clamps rather than refusing",
          big.get("returned") == min(count, 512), big.get("returned"))


def test_vector_window_rejects_bad_input():
    for label, kw in [
        ("window on a non-vector", {"limit": 5}),
        ("negative offset", {"offset": -1}),
        ("zero limit", {"limit": 0}),
    ]:
        path, prop = ("live_set", "tempo") if "non-vector" in label \
            else ("live_set", "tracks")
        try:
            S.lom_get(path, prop, **kw)
            check("%s rejected" % label, False, "no exception")
        except S.LiveError:
            check("%s rejected" % label, True)
        except Exception as e:
            check("%s raises LiveError" % label, False, type(e).__name__)


def test_path_marker_rejects_malformed_input():
    """A bad marker must fail loudly, not resolve to something arbitrary."""
    for label, bad in [
        ("extra keys", {"__path__": "live_set", "junk": 1}),
        ("non-string target", {"__path__": 7}),
        ("unknown root", {"__path__": "not_a_root tracks 0"}),
    ]:
        try:
            S.lom_set("live_set view", "selected_track", bad)
            check("%s rejected" % label, False, "no exception")
        except S.LiveError:
            check("%s rejected" % label, True)
        except Exception as e:
            check("%s raises LiveError" % label, False, type(e).__name__)


def test_plain_strings_are_never_treated_as_paths():
    """A string arg that reads like a path stays a string.

    Guards the reason markers are explicit: silently coercing path-shaped
    strings would corrupt free-form text (names, set_data) on a guess.
    """
    before = jcall(S.lom_get, "live_set tracks 0", "name")["value"]
    jcall(S.lom_set, "live_set tracks 0", "name", "live_set tracks 0")
    after = jcall(S.lom_get, "live_set tracks 0", "name")["value"]
    check("path-shaped name stored verbatim", after == "live_set tracks 0", after)
    jcall(S.lom_set, "live_set tracks 0", "name", before)
    check("name restored",
          jcall(S.lom_get, "live_set tracks 0", "name")["value"] == before)


def test_observer_roundtrip_via_mcp():
    jcall(S.lom_unobserve_all)
    o = jcall(S.lom_observe, "live_set", "tempo")
    check("observe via MCP layer", o.get("observing") is True, o)

    before = jcall(S.lom_get, "live_set", "tempo")["value"]
    jcall(S.lom_set, "live_set", "tempo", round(before + 1.0, 3))
    ev = jcall(S.lom_poll_events)
    check("poll returns the event", ev.get("count", 0) >= 1, ev.get("count"))
    check("poll reports a cursor", "latest_seq" in ev)
    jcall(S.lom_set, "live_set", "tempo", before)

    seq = ev.get("latest_seq", 0)
    later = jcall(S.lom_poll_events, since=seq + 100)
    check("since cursor filters", later.get("count") == 0, later.get("count"))

    lst = jcall(S.lom_observers)
    check("observers listed", lst.get("active_listeners") == 1, lst)
    cleared = jcall(S.lom_unobserve_all)
    check("no listeners leaked", cleared.get("leaked") == 0, cleared)


def test_browser_via_mcp():
    r = jcall(S.browser_list)
    check("browser roots via MCP layer", len(r.get("items", [])) > 0)
    sub = jcall(S.browser_list, "instruments")
    check("browser descends", len(sub.get("items", [])) > 0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("End-to-end: MCP layer -> Live (%d groups)" % len(tests))
    try:
        S._request("ping")
    except S.LiveError as e:
        print("SKIP: %s" % e)
        return 0
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
