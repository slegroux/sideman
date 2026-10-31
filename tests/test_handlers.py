#!/usr/bin/env python3
"""Unit tests for the Remote Script engine. Runs WITHOUT Ableton Live.

handlers.py imports Live, which exists only inside Ableton, so a stub module
stands in for it and plain Python objects stand in for the Song, tracks and
clips. That reaches the engine's own logic - here the undo classes every
mutating result carries - but not Live's behaviour: smoke.py covers that.

  ./tests/test_handlers.py        run
  ./tests/test_handlers.py -v     list each assertion
"""
import importlib.util
import json
import pathlib
import re
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent

_live = types.ModuleType("Live")
_live.Application = types.SimpleNamespace(
    get_application=lambda: types.SimpleNamespace(view=None))

# Loaded by file path: importing the AbletonLOM package would run its
# __init__.py, which needs Live's _Framework. The stub Live is in sys.modules
# only while handlers.py imports it (the module keeps its own reference), so
# it can't leak into other suites in the same process.
_had_live = "Live" in sys.modules
sys.modules.setdefault("Live", _live)
try:
    _spec = importlib.util.spec_from_file_location(
        "lom_handlers", ROOT / "remote_script" / "AbletonLOM" / "handlers.py")
    H = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(H)
finally:
    if not _had_live:
        del sys.modules["Live"]

VERBOSE = False
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        if VERBOSE:
            print("  ok   %s" % name)
    else:
        FAILURES.append("%s %s" % (name, detail))
        print("  FAIL %s %s" % (name, detail))


# ------------------------------------------------------------------- fakes

class Clip(object):
    def __init__(self, midi=True, notes=4):
        self.is_midi_clip = midi
        self.has_envelopes = False
        self.name = "clip"
        self.length = 4.0
        self.notes = list(range(notes))

    def get_notes_extended(self, *window):
        return list(self.notes)

    def remove_notes_extended(self, *window):
        self.notes = []

    def crop(self):
        pass


class Slot(object):
    def __init__(self, clip, track=None):
        self.clip = clip
        self.canonical_parent = track

    @property
    def has_clip(self):
        return self.clip is not None

    def fire(self):
        pass

    def delete_clip(self):
        self.clip = None


class Track(object):
    def __init__(self, name):
        self.name = name
        self.arm = False
        self.devices = []
        self.clip_slots = [Slot(Clip(midi=True), self),
                           Slot(Clip(midi=False), self), Slot(None, self)]

    def stop_all_clips(self):
        pass

    def delete_clip(self, clip):
        pass


class Looper(object):
    def double_speed(self):
        pass

    def undo(self):
        pass


class Browser(object):
    hotswap_target = None

    def load_item(self, item):
        pass


class Song(object):
    def __init__(self):
        self.tempo = 120.0
        self.record_mode = False
        self.tracks = [Track("A"), Track("B")]
        self.tracks[0].devices = [Looper()]
        self.view = types.SimpleNamespace(selected_track=self.tracks[1])
        self.scenes = [Scene()]
        self.browser = Browser()

    def undo(self):
        pass

    def begin_undo_step(self):
        pass

    def end_undo_step(self):
        pass

    def create_midi_track(self, index):
        self.tracks.append(Track("new"))

    def delete_track(self, index):
        del self.tracks[index]

    def start_playing(self):
        pass

    def get_current_beats_song_time(self):
        return "1.1.1"

    def some_future_function(self):
        pass

    def sync_parameter_changes(self):
        pass

    def continue_playing(self):
        pass

    def play_selection(self):
        pass


class Scene(object):
    def __init__(self):
        self.clip_slots = []

    def fire(self):
        pass

    def fire_as_selected(self):
        pass


class Unreadable(object):
    """A clip whose state raises when read."""
    @property
    def is_midi_clip(self):
        raise RuntimeError("gone")


class Surface(object):
    def __init__(self):
        self._song = Song()

    def song(self):
        return self._song


def run(op, surface=None, **params):
    out = H.dispatch(surface or Surface(), op, params)
    json.dumps(out)  # must stay JSON-serialisable
    return out


# ------------------------------------------------------------------- tests

def test_undo_class_set_is_parameter():
    out = run("set", path="live_set", property="tempo", value=128)
    check("set -> parameter", out.get("undo_class") == "parameter", out)


def test_undo_class_create_midi_track_is_additive():
    out = run("call", path="live_set", function="create_midi_track", args=[-1])
    check("create_midi_track -> additive",
          out.get("undo_class") == "additive", out)


def test_undo_class_notes_remove_is_destructive_light():
    out = run("notes_remove", path="live_set tracks 0 clip_slots 0 clip")
    check("notes_remove -> destructive_light",
          out.get("undo_class") == "destructive_light", out)


def test_undo_class_delete_track_is_destructive_heavy():
    out = run("call", path="live_set", function="delete_track", args=[0],
              confirm=True)
    check("delete_track -> destructive_heavy",
          out.get("undo_class") == "destructive_heavy", out)


def test_undo_class_delete_clip_depends_on_the_clip():
    midi = run("call", path="live_set tracks 0 clip_slots 0",
               function="delete_clip", confirm=True)
    audio = run("call", path="live_set tracks 0 clip_slots 1",
                function="delete_clip", confirm=True)
    check("delete_clip on a MIDI clip -> destructive_light",
          midi.get("undo_class") == "destructive_light", midi)
    check("delete_clip on an audio clip -> destructive_heavy",
          audio.get("undo_class") == "destructive_heavy", audio)


def test_undo_class_transport_is_none_and_reads_carry_none():
    out = run("call", path="live_set", function="start_playing")
    check("start_playing -> none", out.get("undo_class") == "none", out)
    got = run("get", path="live_set", property="tempo")
    check("a read carries no undo_class", "undo_class" not in got, got)
    fn = run("call", path="live_set", function="get_current_beats_song_time")
    check("a read function called through call -> none",
          fn.get("undo_class") == "none", fn)


def test_undo_class_unlisted_function_is_unknown():
    out = run("call", path="live_set", function="some_future_function")
    check("unlisted function -> unknown", out.get("undo_class") == "unknown", out)


def test_undo_class_transaction_reports_the_heaviest():
    out = run("transaction", ops=[
        {"op": "set", "path": "live_set", "property": "tempo", "value": 100},
        {"op": "call", "path": "live_set", "function": "create_midi_track",
         "args": [-1]},
        {"op": "notes_remove", "path": "live_set tracks 0 clip_slots 0 clip"},
    ])
    per_op = [r.get("undo_class") for r in out["results"]]
    check("transaction tags each op",
          per_op == ["parameter", "additive", "destructive_light"], per_op)
    check("transaction -> heaviest applied (destructive_light)",
          out.get("undo_class") == "destructive_light", out.get("undo_class"))


def test_undo_class_failed_ops_do_not_count():
    out = run("transaction", stop_on_error=False, ops=[
        {"op": "set", "path": "live_set", "property": "tempo", "value": 100},
        {"op": "call", "path": "live_set", "function": "delete_track",
         "args": [0]},  # no confirm -> refused, never applied
    ])
    check("a refused delete is not counted",
          out.get("undo_class") == "parameter", out.get("undo_class"))


def test_undo_class_midi_clip_with_envelopes_is_heavy():
    surf = Surface()
    surf.song().tracks[0].clip_slots[0].clip.has_envelopes = True
    out = run("call", surf, path="live_set tracks 0 clip_slots 0",
              function="delete_clip", confirm=True)
    check("delete_clip on a MIDI clip with envelopes -> destructive_heavy",
          out.get("undo_class") == "destructive_heavy", out)


def test_undo_class_per_instance_refinements():
    looper = run("call", path="live_set tracks 0 devices 0", function="undo")
    check("LooperDevice.undo -> destructive_heavy",
          looper.get("undo_class") == "destructive_heavy", looper)
    song = run("call", path="live_set", function="undo")
    check("Song.undo -> none", song.get("undo_class") == "none", song)

    surf = Surface()
    surf.song().record_mode = True
    rec = run("call", surf, path="live_set", function="start_playing")
    check("start_playing with arrangement record on -> destructive_heavy",
          rec.get("undo_class") == "destructive_heavy", rec)

    surf = Surface()
    surf.song().tracks[0].arm = True
    armed = run("call", surf, path="live_set tracks 0 clip_slots 2",
                function="fire")
    check("fire on an armed empty slot -> additive",
          armed.get("undo_class") == "additive", armed)
    plain = run("call", path="live_set tracks 0 clip_slots 0", function="fire")
    check("fire on a clip -> none", plain.get("undo_class") == "none", plain)


def test_undo_class_browser_load_depends_on_the_track():
    def load(surf, track):
        surf.song().view.selected_track = surf.song().tracks[track]
        return H._load_undo_class(surf, surf.song().browser)

    check("load onto an empty track -> additive",
          load(Surface(), 1) == "additive")
    check("load onto a track with devices -> destructive_heavy",
          load(Surface(), 0) == "destructive_heavy")
    surf = Surface()
    surf.song().browser.hotswap_target = object()
    check("load with a hot-swap target -> destructive_heavy",
          load(surf, 1) == "destructive_heavy")
    check("unreadable state -> destructive_heavy",
          H._load_undo_class(Surface(), None, track=object())
          == "destructive_heavy")


def test_undo_class_rank_order():
    check("rank order", H.UNDO_RANK == (
        "none", "parameter", "additive", "destructive_light", "unknown",
        "destructive_heavy"), H.UNDO_RANK)
    pairs = [(("parameter", "additive"), "additive"),
             (("additive", "destructive_light"), "destructive_light"),
             (("destructive_light", "unknown"), "unknown"),
             (("unknown", "destructive_heavy"), "destructive_heavy"),
             (("none", "parameter"), "parameter"),
             ((), "none")]
    for given, want in pairs:
        got = H._heaviest(list(given))
        check("heaviest%s -> %s" % (given, want), got == want, got)
    out = run("transaction", ops=[
        {"op": "call", "path": "live_set", "function": "some_future_function"},
        {"op": "call", "path": "live_set", "function": "delete_track",
         "args": [0], "confirm": True},
    ])
    check("an unknown call can't hide a heavy delete",
          out.get("undo_class") == "destructive_heavy", out.get("undo_class"))


def test_undo_class_rolled_back_batch_is_none():
    out = run("transaction", rollback_on_error=True, ops=[
        {"op": "set", "path": "live_set", "property": "tempo", "value": 100},
        {"op": "set", "path": "live_set", "property": "no_such", "value": 1},
    ])
    check("rolled back", out.get("rolled_back") is True, out)
    check("a reverted batch -> none", out.get("undo_class") == "none",
          out.get("undo_class"))


def test_undo_class_golden_table():
    # Written out by hand on purpose: a relabel in handlers.py (a heavy delete
    # turned light, say) must fail here, not pass because the test reads the
    # same table it checks.
    want = {}
    for cls, names in GOLDEN.items():
        for n in names.split():
            want[n] = cls
    got = dict(H._CALL_UNDO_CLASS)
    diff = sorted(set(want.items()) ^ set(got.items()))
    check("call table matches the golden list", not diff, diff)
    check("typed-op table matches", H._OP_UNDO_CLASS == GOLDEN_OPS,
          sorted(set(H._OP_UNDO_CLASS.items()) ^ set(GOLDEN_OPS.items())))


GOLDEN_OPS = {
    "set": "parameter", "notes_add": "additive", "notes_modify": "parameter",
    "notes_remove": "destructive_light", "envelope_insert_step": "parameter",
    "envelope_clear": "destructive_light",
    "arrangement_create_clip": "additive",
    "arrangement_duplicate_clip": "additive",
    "warp_markers_set": "destructive_light",
}

GOLDEN = {
    "none": """fire fire_as_selected stop stop_all_clips start_playing
        stop_playing continue_playing play_selection jump jump_by
        jump_to_next_cue jump_to_prev_cue jump_in_running_session_clip scrub
        scrub_by stop_scrub move_playing_pos set_fire_button_state play
        stop_preview focus_view show_view hide_view zoom_view scroll_view
        toggle_browse select_device select_instrument select_envelope_parameter
        show_envelope hide_envelope show_loop select_all_notes
        deselect_all_notes select_notes_by_id show_message
        show_on_the_fly_message undo redo begin_undo_step end_undo_step
        sync_parameter_changes begin_gesture end_gesture beat_to_sample_time
        sample_to_beat_time seconds_to_sample_time find_device_position
        guess_playback_length note_number_to_name iter_children
        automation_envelope str_for_value available_main_views __str__
        preview_item force_link_beat_time""",
    "parameter": """apply_note_modifications quantize quantize_pitch
        move_warp_marker move_slice move_device randomize_macros
        recall_selected_variation recall_last_used_variation
        re_enable_automation set_modulation_value set_data tap_tempo warp_as
        warp_double warp_half store_chosen_bank save_preset_to_compare_ab_slot
        store_variation double_speed half_speed reverse""",
    "additive": """create_audio_clip create_audio_track
        create_automation_envelope create_clip create_midi_clip
        create_midi_track create_return_track create_scene create_take_lane
        duplicate_clip_slot duplicate_clip_to_arrangement duplicate_device
        duplicate_loop duplicate_notes_by_id duplicate_region duplicate_scene
        duplicate_track insert_chain insert_device insert_slice add_new_notes
        add_warp_marker add_macro add_parameter_to_modulation_matrix
        capture_and_insert_scene capture_midi trigger_session_record overdub
        double_length export_to_clip_slot load_item""",
    "destructive_light": """remove_notes remove_notes_by_id
        remove_notes_extended set_notes replace_selected_notes clear_envelope
        clear_all_envelopes remove_warp_marker remove_slice clear_slices
        reset_slices set_or_delete_cue delete_selected_variation""",
    "destructive_heavy": """delete_track delete_return_track delete_scene
        delete_device delete_all_chains remove_macro copy_pad
        duplicate_clip_to replace_sample clear record half_length delete_clip
        crop""",
    "unknown": "press_current_dialog_button",
}


def test_undo_class_each_refinement():
    # One check per branch of _call_undo_class, so dropping any refinement
    # fails here (round-2 review: most survived before).
    for fn in ("start_playing", "continue_playing", "play_selection"):
        surf = Surface()
        surf.song().record_mode = True
        out = run("call", surf, path="live_set", function=fn)
        check("%s with record on -> heavy" % fn,
              out.get("undo_class") == "destructive_heavy", out)
        out = run("call", path="live_set", function=fn)
        check("%s with record off -> none" % fn,
              out.get("undo_class") == "none", out)
    for fn in ("fire", "fire_as_selected"):
        surf = Surface()
        surf.song().record_mode = True
        out = run("call", surf, path="live_set scenes 0", function=fn)
        check("Scene.%s with record on -> heavy" % fn,
              out.get("undo_class") == "destructive_heavy", out)
        surf = Surface()
        surf.song().tracks[1].arm = True
        out = run("call", surf, path="live_set scenes 0", function=fn)
        check("Scene.%s with an armed track -> additive" % fn,
              out.get("undo_class") == "additive", out)
        out = run("call", path="live_set scenes 0", function=fn)
        check("Scene.%s, nothing armed -> none" % fn,
              out.get("undo_class") == "none", out)

    surf = Surface()
    track = surf.song().tracks[0]
    midi = track.clip_slots[0].clip
    audio = track.clip_slots[1].clip
    check("Track.delete_clip(midi clip) -> light",
          H._call_undo_class(surf, track, "delete_clip", [midi])
          == "destructive_light")
    check("Track.delete_clip(audio clip) -> heavy",
          H._call_undo_class(surf, track, "delete_clip", [audio])
          == "destructive_heavy")
    out = run("call", path="live_set tracks 0 clip_slots 0 clip",
              function="crop", confirm=True)
    check("crop on a MIDI clip -> light",
          out.get("undo_class") == "destructive_light", out)
    out = run("call", path="live_set tracks 0 clip_slots 1 clip",
              function="crop", confirm=True)
    check("crop on an audio clip -> heavy",
          out.get("undo_class") == "destructive_heavy", out)
    check("crop on a clip that raises -> heavy",
          H._call_undo_class(surf, Unreadable(), "crop", [])
          == "destructive_heavy")
    no_attr = types.SimpleNamespace(is_midi_clip=True)
    check("MIDI clip with no has_envelopes -> heavy",
          H._call_undo_class(surf, no_attr, "crop", []) == "destructive_heavy")
    check("fire that raises -> heavy, not the table's none",
          H._call_undo_class(types.SimpleNamespace(song=None), object(),
                             "fire", []) == "destructive_heavy")

    surf = Surface()
    surf.song().view.selected_track = surf.song().tracks[0]
    out = run("call", surf, path="live_set browser", function="load_item",
              args=[None])
    check("call load_item consults the track (devices -> heavy)",
          out.get("undo_class") == "destructive_heavy", out)


def test_undo_class_browser_load_op():
    real = H._browser, H._browser_resolve
    item = types.SimpleNamespace(is_loadable=True)
    try:
        H._browser = lambda surface: surface.song().browser
        H._browser_resolve = lambda surface, rel: item
        empty = run("browser_load", path="drums/kit", track_index=1)
        full = run("browser_load", path="drums/kit", track_index=0)
    finally:
        H._browser, H._browser_resolve = real
    check("browser_load onto an empty track -> additive",
          empty.get("undo_class") == "additive", empty)
    check("browser_load onto a track with devices -> heavy",
          full.get("undo_class") == "destructive_heavy", full)


def test_undo_class_nested_rollback_is_not_none():
    out = run("transaction", ops=[{"op": "transaction",
                                   "rollback_on_error": True, "ops": [
        {"op": "set", "path": "live_set", "property": "tempo", "value": 100},
        {"op": "set", "path": "live_set", "property": "no_such", "value": 1},
    ]}])
    inner = out["results"][0]
    check("nested rollback is not labelled none",
          inner.get("undo_class") != "none", inner)


def test_undo_class_covers_every_typed_mutating_op():
    reads = {"describe", "search", "canonical_path", "get_batch", "get",
             "count", "types", "envelope_get", "arrangement_list", "notes_get",
             "browser_list", "observe_add", "observe_remove", "observe_clear",
             "observe_list", "observe_poll"}
    per_call = {"transaction", "set_batch", "call", "browser_load"}
    missing = sorted(set(H.OPS) - reads - per_call - set(H._OP_UNDO_CLASS))
    check("every op is read-only, batched or classified", not missing, missing)


def test_undo_class_table_agrees_with_the_confirm_gate():
    # Every function the confirm gate guards must be classed destructive.
    census = json.loads(
        (ROOT / "baseline" / "lom_census_12.4.6.json").read_text())
    names = {m for t in census["types"].values() for m in t["members"]}
    gated = sorted(n for n in names if H._is_destructive(n))
    wrong = [n for n in gated
             if not H._CALL_UNDO_CLASS.get(n, "").startswith("destructive")]
    check("gated functions are classed destructive (%d)" % len(gated),
          gated and not wrong, wrong)
    bad = sorted(c for c in set(H._CALL_UNDO_CLASS.values()) | set(
        H._OP_UNDO_CLASS.values()) if c not in H.UNDO_RANK)
    check("every class is a ranked one", not bad, bad)
    absent = sorted(n for n in H._CALL_UNDO_CLASS if n not in names)
    check("every classified function exists on Live 12.4.6", not absent, absent)


def main(argv=None):
    """argv: this suite's own flags (-v, -k PATTERN). The pytest entry passes
    [] so pytest's own -k never reaches here."""
    argv = sys.argv[1:] if argv is None else argv
    global VERBOSE
    VERBOSE = "-v" in argv
    selector = None
    if "-k" in argv:
        i = argv.index("-k")
        if i + 1 >= len(argv):
            print("-k needs a pattern")
            return 2
        selector = re.compile(argv[i + 1])
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and (selector is None or selector.search(k))]
    if not tests:
        print("no test matches -k %s" % argv[argv.index("-k") + 1])
        return 1
    del FAILURES[:]
    print("Engine unit tests (%d groups, no Live required)" % len(tests))
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
