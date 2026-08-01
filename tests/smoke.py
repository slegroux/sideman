#!/usr/bin/env python3
"""Regression smoke test against a running Ableton Live.

Locks the behaviour that the cleanup pass must not change. Requires Live
running with AbletonLOM enabled - there is no way to test a Remote Script
without the host, so this is an integration test by necessity.

Writes are reversible and restore the original value. It refuses to touch
anything but tempo and its own scratch state.

  ./tests/smoke.py            run
  ./tests/smoke.py -v         show each assertion
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from lomcli import request  # noqa: E402

VERBOSE = "-v" in sys.argv
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        if VERBOSE:
            print("  ok   %s" % name)
    else:
        FAILURES.append("%s %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))


def ok(resp, name):
    if not resp.get("ok"):
        FAILURES.append("%s -> %s" % (name, resp.get("error")))
        print("  FAIL %s -> %s" % (name, resp.get("error")))
        return None
    return resp["result"]


def test_connection():
    r = ok(request("ping"), "ping")
    check("handlers loaded", bool(r and r.get("handlers")))


def test_describe():
    r = ok(request("describe", {"path": "live_set", "include_values": False}),
           "describe")
    if not r:
        return
    check("describe type", r["type"] == "Song.Song", r.get("type"))
    for key in ("properties", "children", "functions", "unavailable", "counts"):
        check("describe has %s" % key, key in r)
    check("describe finds tempo", "tempo" in r["properties"])
    check("describe finds tracks child", "tracks" in r["children"])


def test_per_instance_availability():
    """The Phase 0 caveat: availability is per instance, not per type."""
    r = ok(request("describe", {"path": "live_set tracks 0",
                                "include_values": False}), "describe track")
    if not r:
        return
    check("track reports unavailable members", len(r["unavailable"]) > 0,
          "expected e.g. input_meter_left on a MIDI track")


def test_get_set_roundtrip():
    before = ok(request("get", {"path": "live_set", "property": "tempo"}),
                "get tempo")
    if not before:
        return
    original = before["value"]
    target = round(original + 1.0, 3)
    ok(request("set", {"path": "live_set", "property": "tempo",
                       "value": target}), "set tempo")
    after = ok(request("get", {"path": "live_set", "property": "tempo"}),
               "get tempo back")
    check("set then get roundtrips", after and abs(after["value"] - target) < 0.01,
          "wanted %s got %s" % (target, after and after["value"]))
    ok(request("set", {"path": "live_set", "property": "tempo",
                       "value": original}), "restore tempo")
    restored = ok(request("get", {"path": "live_set", "property": "tempo"}),
                  "verify restore")
    check("tempo restored", restored and abs(restored["value"] - original) < 0.01)


def test_get_batch_partial_failure():
    """One unavailable property must not lose the other reads."""
    r = ok(request("get_batch", {"specs": [
        {"path": "live_set", "property": "tempo"},
        {"path": "live_set", "property": "definitely_not_a_property"},
    ]}), "get_batch")
    if not r:
        return
    check("batch returns all entries", r["count"] == 2, r.get("count"))
    check("batch isolates failure", r["ok_count"] == 1, r.get("ok_count"))


def test_write_guard():
    r = request("call", {"path": "live_set", "function": "delete_track",
                         "args": [0], "confirm": False})
    check("destructive call refused without confirm", not r.get("ok"))
    check("refusal is a PermissionError",
          (r.get("error") or {}).get("type") == "PermissionError",
          (r.get("error") or {}).get("type"))

    # Regression: the guard used to be a hand-kept list that had drifted,
    # leaving these destructive Clip functions completely unguarded.
    clip = None
    for i in range(24):
        c = request("count", {"path": "live_set tracks %d" % i,
                              "child": "arrangement_clips"})
        if c.get("ok") and c["result"]["count"]:
            clip = "live_set tracks %d arrangement_clips 0" % i
            break
    if clip is None:
        print("  skip clip-guard checks (no clip in this Set)")
        return
    for fn in ("remove_notes_extended", "clear_all_envelopes",
               "clear_envelope", "crop", "remove_warp_marker"):
        g = request("call", {"path": clip, "function": fn, "args": [],
                             "confirm": False})
        check("guard covers %s" % fn,
              (g.get("error") or {}).get("type") == "PermissionError",
              (g.get("error") or {}).get("type"))

    # ...without over-reaching: listener plumbing is not an edit.
    live = request("call", {"path": "live_app",
                            "function": "get_major_version", "args": []})
    check("non-destructive call still allowed", live.get("ok"),
          (live.get("error") or {}).get("message", "")[:60])


def test_search():
    r = ok(request("search", {"type": "Track", "max_results": 100}), "search")
    if not r:
        return
    check("search finds tracks", r["count"] > 0, r.get("count"))
    check("search reports truncation flag", "truncated" in r)
    # The identity bug made this return only the first branch of the graph.
    r2 = ok(request("search", {"type": "DeviceParameter", "max_results": 500}),
            "search params")
    if r2:
        tracks = set(p["path"].split()[2] for p in r2["results"]
                     if p["path"].startswith("live_set tracks "))
        check("search spans multiple tracks (identity bug regression)",
              len(tracks) > 1, "only %d track(s)" % len(tracks))


def test_canonical_path():
    r = ok(request("canonical_path",
                   {"path": "live_set view selected_track"}), "canonical_path")
    if not r:
        return
    check("alias resolves to a concrete path",
          r["canonical_path"].startswith("live_set tracks "),
          r.get("canonical_path"))
    check("alias flagged", r["is_alias"] is True)


def test_transaction_groups():
    """Groupable ops must collapse into ONE undo step."""
    g = lambda p, a: (request("get", {"path": p, "property": a})
                      .get("result", {}) or {}).get("value")
    n0, n1 = g("live_set tracks 0", "name"), g("live_set tracks 1", "name")
    if n0 is None or n1 is None:
        check("transaction test needs 2 tracks", False)
        return
    r = ok(request("transaction", {"ops": [
        {"op": "set", "path": "live_set tracks 0", "property": "name",
         "value": "__smoke_a"},
        {"op": "set", "path": "live_set tracks 1", "property": "name",
         "value": "__smoke_b"},
    ]}), "transaction")
    if not r:
        return
    check("transaction applied both", r["applied"] == 2, r.get("applied"))
    check("transaction states the undo caveat", "undo" in r)
    request("call", {"path": "live_set", "function": "undo", "args": []})
    check("ONE undo reverted both (grouping)",
          g("live_set tracks 0", "name") == n0
          and g("live_set tracks 1", "name") == n1,
          "names now %r/%r" % (g("live_set tracks 0", "name"),
                               g("live_set tracks 1", "name")))


def test_observers():
    ok(request("observe_clear"), "observe_clear")
    ok(request("observe_add", {"path": "live_set", "property": "tempo"}),
       "observe_add")
    dup = ok(request("observe_add", {"path": "live_set", "property": "tempo"}),
             "observe_add dup")
    check("duplicate observe is idempotent", dup and dup.get("already_observing"))
    bad = request("observe_add", {"path": "live_set", "property": "file_path"})
    check("non-observable property rejected", not bad.get("ok"))

    before = ok(request("get", {"path": "live_set", "property": "tempo"}),
                "tempo")["value"]
    request("set", {"path": "live_set", "property": "tempo",
                    "value": round(before + 1.0, 3)})
    ev = ok(request("observe_poll", {}), "observe_poll")
    check("observer captured the change", ev and ev["count"] >= 1,
          ev and ev.get("count"))
    request("set", {"path": "live_set", "property": "tempo", "value": before})

    cleared = ok(request("observe_clear"), "observe_clear final")
    check("no listeners leaked", cleared and cleared.get("leaked") == 0,
          cleared and cleared.get("leaked"))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print("AbletonLOM smoke test (%d groups)" % len(tests))
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
