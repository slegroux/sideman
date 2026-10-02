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
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from lomcli import request  # noqa: E402

VERBOSE = "-v" in sys.argv
FAILURES = []
SKIPS = []


def skip(name, reason):
    """A skipped group is not a passed group. Recording it here keeps the
    final summary from reading as full coverage when part of it never ran."""
    SKIPS.append("%s (%s)" % (name, reason))
    print("  skip %s (%s)" % (name, reason))


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
    """Availability is per instance, not per type."""
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


def test_stringified_scalars_are_coerced():
    """`value` is untyped in the tool schema, so some MCP clients send 122 as
    "122". Live's C++ setters reject that outright, which made lom_set unusable
    for every numeric property from those clients. No other suite sees it: they
    all call the tool function directly and never cross the client boundary
    where the retyping happens."""
    before = ok(request("get", {"path": "live_set", "property": "tempo"}),
                "get tempo")
    if not before:
        return
    original = before["value"]
    target = round(original + 1.0, 3)

    r = request("set", {"path": "live_set", "property": "tempo",
                        "value": str(target)})
    check("stringified float accepted for a float property", r.get("ok"),
          (r.get("error") or {}).get("type"))
    after = ok(request("get", {"path": "live_set", "property": "tempo"}),
               "get tempo back")
    check("stringified float lands as a number",
          after and abs(after["value"] - target) < 0.01,
          after and after["value"])
    request("set", {"path": "live_set", "property": "tempo",
                    "value": original})

    with Scratch() as s:
        # A str going into a str property must stay exactly as sent. Uses the
        # CLIP's name, never the track's: Scratch.__exit__ refuses to delete a
        # track whose name is not __lomtest, so renaming the track here would
        # strand it in the user's Set if the restore did not run.
        clip = s.clip()
        ok(request("set", {"path": clip, "property": "name",
                           "value": "128"}), "set numeric-looking clip name")
        nm = ok(request("get", {"path": clip, "property": "name"}), "get name")
        check("numeric-looking string stays a string on a str property",
              nm and nm["value"] == "128", nm and nm["value"])

        ci = request("set", {"path": s.track, "property": "color_index",
                             "value": "3"})
        check("stringified int accepted for an int property", ci.get("ok"),
              (ci.get("error") or {}).get("type"))

    bad = request("set", {"path": "live_set", "property": "tempo",
                          "value": "not_a_number"})
    check("non-numeric string still refused", not bad.get("ok"),
          bad.get("result"))


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
    # Makes its own clip rather than hunting for one - this check previously
    # self-skipped on a Set with no clips, and a check that skips is not a check.
    with Scratch() as s:
        clip = s.clip()
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
    def g(p, a):
        return (request("get", {"path": p, "property": a})
                .get("result", {}) or {}).get("value")
    with Scratch() as a, Scratch() as b:
        # Two scratch tracks, never the user's. This test used to rename
        # live_set tracks 0 and 1 and rely on undo to put them back; a run
        # killed between the rename and the undo left a user's tracks called
        # __smoke_a/__smoke_b, with nothing to restore them from.
        #
        # color_index rather than name, for the same reason: the scratch
        # tracks keep the name cleanup identifies them by, whatever undo does.
        # Colour groups under one undo step exactly as name does.
        c0, c1 = g(a.track, "color_index"), g(b.track, "color_index")
        if c0 is None or c1 is None:
            check("transaction test could read both colours", False)
            return
        r = ok(request("transaction", {"ops": [
            {"op": "set", "path": a.track, "property": "color_index",
             "value": (c0 + 1) % 70},
            {"op": "set", "path": b.track, "property": "color_index",
             "value": (c1 + 2) % 70},
        ]}), "transaction")
        if not r:
            return
        check("transaction applied both", r["applied"] == 2, r.get("applied"))
        check("transaction states the undo caveat", "undo" in r)
        request("call", {"path": "live_set", "function": "undo", "args": []})
        check("ONE undo reverted both (grouping)",
              g(a.track, "color_index") == c0 and g(b.track, "color_index") == c1,
              "colours now %r/%r" % (g(a.track, "color_index"),
                                     g(b.track, "color_index")))


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


def test_observer_poll_window():
    """Paging with since/limit must see every event exactly once, and consume
    must not discard what a limit held back."""
    ok(request("observe_clear"), "observe_clear")
    added = ok(request("observe_add", {"path": "live_set", "property": "tempo"}),
               "observe_add")
    base = added and added.get("latest_seq")
    check("observe_add returns latest_seq", isinstance(base, int), added)
    if not isinstance(base, int):
        return
    before = ok(request("get", {"path": "live_set", "property": "tempo"}),
                "tempo")["value"]
    try:
        for i in range(5):
            request("set", {"path": "live_set", "property": "tempo",
                            "value": round(before + 1.0 + i, 3)})
        whole = ok(request("observe_poll", {"since": base}), "poll all")
        want = [e["seq"] for e in whole["events"]]
        check("five changes recorded", len(want) >= 5, len(want))

        first = ok(request("observe_poll", {"since": base, "limit": 2}),
                   "poll page 1")
        check("page is the OLDEST events",
              [e["seq"] for e in first["events"]] == want[:2],
              [e["seq"] for e in first["events"]])
        check("page reports truncated", first.get("truncated") is True)

        seen, cursor = [], base
        for _ in range(len(want) + 2):
            page = ok(request("observe_poll", {"since": cursor, "limit": 2}),
                      "poll page")
            seen += [e["seq"] for e in page["events"]]
            cursor = page["next_since"]
            if not page["truncated"]:
                break
        check("paging by next_since sees every event once", seen == want,
              "%r vs %r" % (seen, want))

        ok(request("observe_poll", {"since": base, "limit": 2, "consume": True}),
           "consume page")
        rest = ok(request("observe_poll", {"since": base}), "poll after consume")
        check("consume keeps what the limit held back",
              [e["seq"] for e in rest["events"]] == want[2:],
              [e["seq"] for e in rest["events"]])
    finally:
        request("set", {"path": "live_set", "property": "tempo", "value": before})
        ok(request("observe_clear"), "observe_clear final")


# ---------------------------------------------------------------- scratch
# The remaining ops need a MIDI clip. A user's Set may be all-audio, so the
# suite makes its own track and removes it. It never touches existing tracks.

SCRATCH_NAME = "__lomtest"


class Scratch:
    """Creates a track at the end of the Set; deletes it on exit."""

    def __init__(self, kind="midi"):
        self.kind = kind

    def __enter__(self):
        n = request("count", {"path": "live_set", "child": "tracks"})
        self.index = n["result"]["count"] if n.get("ok") else None
        fn = "create_audio_track" if self.kind == "audio" else "create_midi_track"
        r = request("call", {"path": "live_set", "function": fn,
                             "args": [-1], "confirm": True})
        if not r.get("ok"):
            raise RuntimeError("could not create scratch track: %s" % r.get("error"))
        self.track = "live_set tracks %d" % self.index
        request("set", {"path": self.track, "property": "name",
                        "value": SCRATCH_NAME})
        return self

    def audio_clip(self, file_path, start=0.0):
        """Import an audio file into the Arrangement; return its clip path."""
        r = request("arrangement_create_clip",
                    {"path": self.track, "start_time": start, "kind": "audio",
                     "file_path": file_path})
        if not r.get("ok"):
            raise RuntimeError("could not create audio clip: %s" % r.get("error"))
        return "%s arrangement_clips 0" % self.track

    def clip(self, length=4.0):
        slot = "%s clip_slots 0" % self.track
        r = request("call", {"path": slot, "function": "create_clip",
                             "args": [length], "confirm": True})
        if not r.get("ok"):
            raise RuntimeError("could not create scratch clip: %s" % r.get("error"))
        return slot + " clip"

    def _locate(self):
        """Index of this scratch track, identified by name rather than by the
        index it was created at.

        The pinned index goes stale whenever the Set changes underneath the
        run - a killed earlier run leaving tracks behind, or the user editing
        while it runs. Refusing to delete on a stale index was safe but
        strands the track, and stranded tracks shift the index for the next
        run, so one interruption compounds into a Set full of __lomtest.
        """
        nm = request("get", {"path": self.track, "property": "name"})
        if nm.get("ok") and nm["result"]["value"] == SCRATCH_NAME:
            return self.index
        n = request("count", {"path": "live_set", "child": "tracks"})
        if not n.get("ok"):
            return None
        found = None
        for i in range(n["result"]["count"]):
            r = request("get", {"path": "live_set tracks %d" % i,
                                "property": "name"})
            if r.get("ok") and r["result"]["value"] == SCRATCH_NAME:
                found = i  # last match: ours was created at the end
        return found

    def __exit__(self, *exc):
        # Delete OUR track, never a user's. The name is the identity here, and
        # it is one no user Set has; the index is only a hint.
        index = self._locate()
        if index is None:
            FAILURES.append("scratch track %r not found; nothing deleted"
                            % SCRATCH_NAME)
            return False
        request("call", {"path": "live_set", "function": "delete_track",
                         "args": [index], "confirm": True})
        return False


def test_notes_lifecycle():
    with Scratch() as s:
        clip = s.clip()
        r = ok(request("notes_add", {"path": clip, "notes": [
            {"pitch": 60, "start_time": 0.0, "duration": 1.0, "velocity": 100},
            {"pitch": 64, "start_time": 1.0, "duration": 1.0, "velocity": 90},
        ]}), "notes_add")
        check("added 2 notes", r and r["added"] == 2, r)

        g = ok(request("notes_get", {"path": clip}), "notes_get")
        check("read back 2 notes", g and g["count"] == 2, g and g.get("count"))
        if not g or not g["notes"]:
            return
        first = g["notes"][0]
        for f in ("note_id", "pitch", "start_time", "duration", "velocity"):
            check("note carries %s" % f, f in first)

        m = ok(request("notes_modify", {"path": clip, "notes": [
            {"note_id": first["note_id"], "pitch": 72}]}), "notes_modify")
        check("modified 1 note", m and m["modified"] == 1, m)
        after = ok(request("notes_get", {"path": clip}), "notes_get 2")
        check("modification persisted",
              after and any(n["pitch"] == 72 for n in after["notes"]))

        bad = ok(request("notes_modify", {"path": clip,
                                          "notes": [{"note_id": 999999}]}),
                 "notes_modify bogus")
        check("unknown note_id reported, not raised",
              bad and bad["unmatched_note_ids"] == [999999], bad)

        rem = ok(request("notes_remove", {"path": clip, "from_pitch": 70,
                                          "pitch_span": 10}), "notes_remove")
        check("range-scoped removal took only the in-range note",
              rem and rem["removed"] == 1, rem)
        left = ok(request("notes_get", {"path": clip}), "notes_get 3")
        check("one note survives", left and left["count"] == 1,
              left and left.get("count"))


def test_envelope_ops():
    with Scratch() as s:
        clip = s.clip()
        param = "%s mixer_device volume" % s.track
        e = ok(request("envelope_get", {"path": clip, "parameter": param,
                                        "samples": 3}), "envelope_get empty")
        check("envelope_get handles absent envelope", e is not None and "exists" in e)

        w = ok(request("envelope_insert_step",
                       {"path": clip, "parameter": param, "time": 0.0,
                        "length": 2.0, "value": 0.5}), "envelope_insert_step")
        check("envelope step written", w is not None)
        e2 = ok(request("envelope_get", {"path": clip, "parameter": param,
                                         "samples": 5}), "envelope_get")
        check("envelope now exists", e2 and e2["exists"] is True)
        check("envelope reports the parameter name",
              e2 and e2.get("parameter_name"), e2 and e2.get("parameter_name"))
        inside = [p for p in (e2 or {}).get("points", []) if 0 < p["time"] < 2.0]
        check("written value reads back", any(abs(p["value"] - 0.5) < 1e-6
                                              for p in inside),
              [p["value"] for p in inside])

        c = ok(request("envelope_clear", {"path": clip}), "envelope_clear")
        check("envelope cleared", c and c.get("cleared") == "all", c)


def test_arrangement_ops():
    with Scratch() as s:
        r = ok(request("arrangement_create_clip",
                       {"path": s.track, "start_time": 0.0, "length": 8.0,
                        "kind": "midi"}), "arrangement_create_clip")
        check("arrangement clip created", r is not None)
        lst = ok(request("arrangement_list", {"path": s.track}),
                 "arrangement_list")
        check("arrangement lists the new clip", lst and lst["count"] == 1,
              lst and lst.get("count"))
        if lst and lst["clips"]:
            c = lst["clips"][0]
            check("clip reports its span",
                  c["start_time"] == 0.0 and c["end_time"] == 8.0, c)
            check("clip reported as MIDI", c["is_midi"] is True, c)


def test_browser_ops():
    r = ok(request("browser_list", {"path": ""}), "browser_list roots")
    check("browser exposes roots", r and len(r["items"]) > 0)
    roots = {i.get("root") for i in (r or {}).get("items", [])}
    for expect in ("instruments", "audio_effects", "plugins"):
        check("browser has %s" % expect, expect in roots)

    sub = ok(request("browser_list", {"path": "instruments"}),
             "browser_list instruments")
    check("instruments has children", sub and len(sub["items"]) > 0)

    bad = request("browser_list", {"path": "definitely_not_a_root"})
    check("unknown browser root rejected", not bad.get("ok"))


def test_browser_load():
    """Loads a real device. Slow-ish, but it is the only way to prove the
    browser path actually resolves to something loadable."""
    with Scratch() as s:
        before = ok(request("count", {"path": s.track, "child": "devices"}),
                    "devices before")
        check("scratch track starts empty", before and before["count"] == 0,
              before and before.get("count"))

        listing = ok(request("browser_list", {"path": "instruments"}),
                     "browser_list")
        target = next((i for i in (listing or {}).get("items", [])
                       if i.get("is_loadable")), None)
        if target is None:
            skip("browser_load", "no loadable instrument found")
            return

        r = ok(request("browser_load", {"path": "instruments/%s" % target["name"],
                                        "track_index": s.index}),
               "browser_load")
        check("browser_load reports the target track", r and r.get("track"), r)
        after = ok(request("count", {"path": s.track, "child": "devices"}),
                   "devices after")
        check("device actually loaded onto the track",
              after and after["count"] == 1, after and after.get("count"))

        nl = request("browser_load", {"path": "instruments"})
        check("loading a non-loadable folder is refused", not nl.get("ok"))


def test_arrangement_duplicate_clip():
    with Scratch() as s:
        clip = s.clip()
        r = ok(request("arrangement_duplicate_clip",
                       {"path": s.track, "clip": clip,
                        "destination_time": 4.0}),
               "arrangement_duplicate_clip")
        check("duplicate reported", r is not None)
        lst = ok(request("arrangement_list", {"path": s.track}),
                 "arrangement_list after duplicate")
        check("session clip landed in the arrangement",
              lst and lst["count"] == 1, lst and lst.get("count"))
        if lst and lst["clips"]:
            check("landed at the requested time",
                  lst["clips"][0]["start_time"] == 4.0, lst["clips"][0])


def _core_library_sample():
    """A .wav that ships with Live, so the warp test needs no fixture of ours."""
    roots = sorted(pathlib.Path("/Applications").glob(
        "Ableton Live *.app/Contents/App-Resources/Core Library/Samples/Loops"))
    for root in roots:
        for wav in sorted(root.rglob("*.wav")):
            return str(wav)
    return None


def test_warp_markers():
    """Warp markers cannot go through generic `call` - Live wants a C++
    WarpMarker that JSON cannot express - so this is the only cover for the
    typed path."""
    sample = _core_library_sample()
    if sample is None:
        # Live installed outside /Applications is a valid setup, not a failure.
        # It does mean the typed warp path went untested, hence the record.
        skip("test_warp_markers", "no Core Library .wav found")
        return

    with Scratch(kind="audio") as s:
        clip = s.audio_clip(sample)
        r = ok(request("warp_markers_set",
                       {"path": clip, "markers": [[0.0, 0.0], [4.0, 2.0]]}),
               "warp_markers_set")
        if not r:
            return
        check("warping turned on", r.get("warping") is True, r)
        check("markers added", r.get("added") == 2, r.get("added"))
        check("clip reports at least the new markers",
              r.get("marker_count", 0) >= 2, r.get("marker_count"))
        check("reports which constructor shape Live accepted",
              r.get("marker_shape") in ("kwargs", "positional"),
              r.get("marker_shape"))

        bad = request("warp_markers_set", {"path": clip,
                                           "markers": [[0.0, 0.0]]})
        check("fewer than 2 markers refused", not bad.get("ok"),
              bad.get("result"))
        dup = request("warp_markers_set",
                      {"path": clip, "markers": [[1.0, 0.5], [1.0, 0.9]]})
        check("non-increasing beat_times refused", not dup.get("ok"),
              dup.get("result"))
        neg = request("warp_markers_set",
                      {"path": clip, "markers": [[0.0, 0.0], [-1.0, 2.0]]})
        check("negative times refused", not neg.get("ok"), neg.get("result"))


def test_set_batch():
    with Scratch() as s:
        r = ok(request("set_batch", {"specs": [
            {"path": s.track, "property": "name", "value": "__lomtest"},
            {"path": s.track, "property": "color_index", "value": 5},
        ]}), "set_batch")
        check("set_batch applied both", r and r["applied"] == 2, r)
        check("set_batch states the undo caveat", r and "undo" in r)


def test_types_census():
    r = ok(request("types", {}), "types")
    if not r:
        return
    check("census has types", r["type_count"] > 40, r.get("type_count"))
    check("census counts substantive members",
          r["totals"]["substantive"] > 500, r["totals"].get("substantive"))
    check("census separates listener plumbing",
          r["totals"]["listeners"] > 0, r["totals"].get("listeners"))
    check("census includes unregistered types (Browser et al)",
          r["totals"].get("unregistered_types", 0) > 0,
          r["totals"].get("unregistered_types"))
    # Ableton renamed these private _MxDCore getters once already inside 12.x.
    # If a future Live renames them again, op_types raises rather than quietly
    # censusing a smaller Live - this asserts the resolver actually bound them.
    api = r.get("mxd_api") or {}
    check("census records which _MxDCore spelling resolved",
          all(api.get(k) for k in ("lom_types", "props_for_type")), api)


def test_observer_list_and_remove():
    ok(request("observe_clear"), "observe_clear")
    ok(request("observe_add", {"path": "live_set", "property": "tempo"}),
       "observe_add")
    lst = ok(request("observe_list"), "observe_list")
    check("observe_list reports the listener",
          lst and lst["active_listeners"] == 1, lst and lst.get("active_listeners"))
    check("observe_list reports object validity",
          lst and lst["listeners"][0]["object_valid"] is True)

    rm = ok(request("observe_remove", {"path": "live_set",
                                       "property": "tempo"}), "observe_remove")
    check("observe_remove reports outcome", rm and rm["outcome"] == "removed", rm)
    check("listener count back to zero",
          rm and rm["active_listeners"] == 0, rm)

    miss = ok(request("observe_remove", {"path": "live_set",
                                         "property": "tempo"}),
              "observe_remove twice")
    check("removing a non-observer is not an error",
          miss and miss["was_observing"] is False, miss)


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
    if SKIPS:
        print("SKIPPED (%d):" % len(SKIPS))
        for s in SKIPS:
            print("  -", s)
        print()
    if FAILURES:
        print("FAILED (%d):" % len(FAILURES))
        for f in FAILURES:
            print("  -", f)
        return 1
    print("ALL PASS" if not SKIPS else "PASS, %d SKIPPED" % len(SKIPS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
