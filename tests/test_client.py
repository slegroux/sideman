#!/usr/bin/env python3
"""The Python client (sideman.client) against a live Ableton.

The third seam: test_mcp stubs the socket, smoke speaks raw JSON, this one
exercises the importable client - the surface docs/tutorial.ipynb is built on.

Scratch-safe, same rules as smoke.py: creates one MIDI track named
__sideman_client at the end of the Set, works only on it, deletes it by name -
and refuses to delete anything whose name changed underneath it. Never touches
existing tracks, never starts playback.

  ./tests/test_client.py
  ./tests/test_client.py -v
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman import Live, LiveError  # noqa: E402

SCRATCH_NAME = "__sideman_client"
VERBOSE = "-v" in sys.argv
FAILURES = []


def check(label, ok, detail=""):
    if VERBOSE or not ok:
        print("  %s %s %s" % ("ok " if ok else "FAIL", label, detail))
    if not ok:
        FAILURES.append(label)


def main():
    live = Live()

    r = live.ping()
    check("ping", r.get("pong") is True and r.get("handlers") is True, r)

    d = live.describe("live_set")
    check("describe has the three member kinds",
          d["counts"]["properties"] > 0 and d["counts"]["functions"] > 0
          and "tracks" in d["children"])

    tempo = live.get("live_set", "tempo")["value"]
    check("get tempo is a number", isinstance(tempo, (int, float)), tempo)

    try:
        live.get("live_set", "no_such_member")
        check("bogus member raises", False)
    except LiveError as e:
        check("bogus member raises", "no_such_member" in str(e), e)

    # The destructive guard must reach this client unchanged.
    try:
        live.call("live_set", "delete_track", [0])
        check("unconfirmed delete refused", False, "guard did not fire")
    except LiveError as e:
        check("unconfirmed delete refused", e.type == "PermissionError", e)

    # ---- scratch lifecycle: everything below happens on our own track ----
    n = live.count("live_set", "tracks")["count"]
    live.call("live_set", "create_midi_track", [-1])
    track = "live_set tracks %d" % n
    live.set(track, "name", SCRATCH_NAME)

    try:
        t = live.transaction(ops=[
            {"op": "set", "path": track, "property": "color_index", "value": 5},
            {"op": "set", "path": track, "property": "color_index", "value": 9},
        ])
        check("transaction applies", t["applied"] == 2 and not t["failed"], t)
        check("transaction states undo grouping", "undo" in t)

        s = live.search(query=SCRATCH_NAME)
        check("search finds the scratch track",
              any(r["path"] == track for r in s["results"]), s["results"])

        live.call(track + " clip_slots 0", "create_clip", [4.0])
        clip = track + " clip_slots 0 clip"
        live.notes_add(clip, [
            {"pitch": 60, "start_time": 0.0, "duration": 1.0, "velocity": 90},
            {"pitch": 64, "start_time": 1.0, "duration": 1.0, "velocity": 90},
        ])
        got = live.notes_get(clip)
        check("notes round-trip", got["count"] == 2, got)

        # A write of an IDENTICAL value does not fire the listener - Live
        # notifies on change, not on set. Change it, then restore it before
        # the by-name cleanup below looks for it.
        live.observe(track, "name")
        live.set(track, "name", SCRATCH_NAME + "_touch")
        live.set(track, "name", SCRATCH_NAME)
        ev = live.poll_events()
        check("observer sees the write",
              any(e["path"] == track for e in ev["events"]), ev["count"])
        cleared = live.unobserve_all()
        check("observers detach", cleared["leaked"] == 0, cleared)
    finally:
        # Delete BY NAME, never by pinned index: if the name changed
        # underneath us, leave the track rather than remove the wrong one.
        name = live.get(track, "name")["value"]
        if name == SCRATCH_NAME:
            live.call("live_set", "delete_track", [n], confirm=True)
        else:
            check("scratch cleanup", False,
                  "track %d is now %r; left in place" % (n, name))

    check("scratch track removed",
          live.count("live_set", "tracks")["count"] == n)

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
