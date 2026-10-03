#!/usr/bin/env python3
"""Build the tutorial's 4-bar loop from an empty Set. The driver behind site/demo.mp4.

Three beats, paced so a screen capture can follow them: an empty Set fills with
a loop, a filter sweep lands as clip automation on Drift's LP Freq, then a fader
moved from another process comes back through an observer.

This script RENAMES AND DELETES TRACKS, so it refuses to run anywhere but a
throwaway Set - see `precondition`. It will never open, close, save or create a
Set for you, and never sends keystrokes to Live: if the check fails, you open a
new Set (Cmd-N) yourself and run it again.

Re-running it over its own output is fine; that is what makes it re-filmable.

  ./scripts/demo.py           film-paced, with pauses between beats
  ./scripts/demo.py --fast    no pauses; for checking the ops, not for filming
"""
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lomcli import request  # noqa: E402

LOMCLI = str(pathlib.Path(__file__).resolve().parent / "lomcli.py")
FAST = "--fast" in sys.argv

# What a Set is allowed to contain for this to be safe to run in.
DEFAULT_TRACK = re.compile(r"^\d+-(MIDI|Audio)$")   # a brand new Set's tracks
SCRATCH = ("__lomtest", "__e2etest", "__sideman_client")  # the test suites' leftovers
BUILT = ("Chords", "Bass", "Drums")                 # our own previous run

TEMPO = 124.0
BEATS_PER_BAR = 4
BARS = 4
LENGTH = BARS * BEATS_PER_BAR       # 16 beats; "one step a bar" needs four bars
CHORDS, BASS, DRUMS = 0, 1, 2
COLOURS = (26, 14, 5)
SWEEP = (0.35, 0.5, 0.72, 0.95)     # LP Freq, one flat step per bar


# ----------------------------------------------------------------- plumbing

def say(msg=""):
    print(msg, flush=True)


START = time.time()


def announce(n, title):
    """The elapsed time is load-bearing: scripts/demo_cut.sh needs the real beat
    offsets to place its captions, and guessing them puts the caption on the
    wrong beat."""
    say()
    say("--- beat %d at t=%.1fs: %s" % (n, time.time() - START, title))


def pause(seconds):
    """Dead air on purpose - the camera needs a moment to land on each beat."""
    if not FAST:
        time.sleep(seconds)


def op(name, **params):
    """One wire op, or stop. A half-built Set is worse than no Set at all."""
    r = request(name, params)
    if not r.get("ok"):
        raise SystemExit("%s%s failed: %s" % (name, params.get("path", ""),
                                              r.get("error")))
    return r["result"]


def clip(track):
    return "live_set tracks %d clip_slots 0 clip" % track


def track_names():
    n = op("count", path="live_set", child="tracks")["count"]
    return [op("get", path="live_set tracks %d" % i, property="name")["value"]
            for i in range(n)]


def show(view):
    op("call", path="app_view", function="show_view", args=[view])


# --------------------------------------------------------- the one hard rule

def precondition():
    """Refuse to run outside a throwaway Set.

    Lives in the script rather than in the recording process on purpose: this is
    what makes it safe for anyone to re-run later, not just safe once.
    """
    r = request("ping")
    if not r.get("ok") or not r["result"].get("handlers"):
        raise SystemExit(
            "no engine on 127.0.0.1:9878 - is Live running with AbletonLOM "
            "selected as a Control Surface?\n  %s" % r.get("error", r))

    saved = op("get", path="live_set", property="file_path")["value"]
    names = track_names()
    strangers = [n for n in names
                 if not DEFAULT_TRACK.match(n)
                 and not n.startswith(SCRATCH)
                 and n not in BUILT]
    if saved or strangers:
        why = ("it has been saved, to %s" % saved if saved else
               "it holds tracks this script did not make: %s"
               % ", ".join(repr(n) for n in strangers))
        raise SystemExit(
            "REFUSING to run: the open Set is not a throwaway - %s.\n"
            "This script deletes and renames tracks, so it only ever runs in a\n"
            "fresh unsaved Set. Open a new Set in Live yourself (Cmd-N), then\n"
            "run this again. It will not open, close or save a Set for you."
            % why)
    say("precondition ok: Set is unsaved, %d track(s) - %s"
        % (len(names), ", ".join(names)))


# ------------------------------------------------------------------- beat 1

def chord_notes():
    """Am7, Fmaj7, Cmaj7, G - one voicing per bar, held."""
    voicings = ((57, 60, 64, 67), (53, 57, 60, 64),
                (60, 64, 67, 71), (55, 59, 62, 67))
    return [{"pitch": p, "start_time": float(bar * BEATS_PER_BAR),
             "duration": 3.5, "velocity": 85}
            for bar, chord in enumerate(voicings) for p in chord]


def bass_notes():
    """Root on the downbeat, then offbeat eighths. Roots A / F / C / G."""
    out = []
    for bar, root in enumerate((33, 29, 36, 31)):
        b = bar * BEATS_PER_BAR
        out.append({"pitch": root, "start_time": float(b),
                    "duration": 0.45, "velocity": 105})
        out += [{"pitch": root, "start_time": b + off,
                 "duration": 0.45, "velocity": 85}
                for off in (0.5, 1.5, 2.5, 3.5)]
    return out


def drum_notes():
    """909 kit: 36 kick, 39 clap, 42 closed hat. Kick on the quarters, clap on
    the backbeat, hats offbeat with alternating velocity so it breathes."""
    out = [{"pitch": 36, "start_time": float(b), "duration": 0.25,
            "velocity": 112} for b in range(LENGTH)]
    out += [{"pitch": 39, "start_time": float(b), "duration": 0.25,
             "velocity": 96} for b in range(1, LENGTH, 2)]
    out += [{"pitch": 42, "start_time": b + 0.5, "duration": 0.2,
             "velocity": 72 if b % 2 == 0 else 58} for b in range(LENGTH)]
    return out


def three_midi_tracks():
    """Leave exactly three MIDI tracks at indices 0-2.

    Creates the shortfall BEFORE deleting anything, so the Set never passes
    through zero tracks - Live is happier that way, and a crash mid-way leaves
    a usable Set rather than an empty one.
    """
    names = track_names()
    keep = [i for i, nm in enumerate(names) if re.match(r"^\d+-MIDI$", nm)][:3]
    for _ in range(3 - len(keep)):
        op("call", path="live_set", function="create_midi_track", args=[-1],
           confirm=True)
        keep.append(op("count", path="live_set", child="tracks")["count"] - 1)
    for i in range(op("count", path="live_set", child="tracks")["count"] - 1,
                   -1, -1):
        if i not in keep:
            op("call", path="live_set", function="delete_track", args=[i],
               confirm=True)


def beat1_loop():
    announce(1, "an empty Set fills with a 4-bar loop")
    show("Session")
    op("set", path="live_set", property="tempo", value=TEMPO)

    three_midi_tracks()
    pause(1)

    # Names and colours group into one Cmd-Z; see the undo caveat in TOOLS.md.
    op("transaction", ops=[
        spec for t, nm, c in zip((CHORDS, BASS, DRUMS), BUILT, COLOURS)
        for spec in ({"op": "set", "path": "live_set tracks %d" % t,
                      "property": "name", "value": nm},
                     {"op": "set", "path": "live_set tracks %d" % t,
                      "property": "color_index", "value": c})])
    say("tracks: %s at %.0f BPM" % (", ".join(track_names()), TEMPO))
    pause(1)

    for path, t in (("instruments/Drift", CHORDS),
                    ("instruments/Drift", BASS),
                    ("drums/909 Core Kit.adg", DRUMS)):
        say("  load %s -> %s" % (path, op("browser_load", path=path,
                                          track_index=t)["track"]))
        pause(0.8)

    for t in (CHORDS, BASS, DRUMS):
        op("call", path="live_set tracks %d clip_slots 0" % t,
           function="create_clip", args=[float(LENGTH)], confirm=True)
    pause(1)

    total = 0
    for t, notes in ((CHORDS, chord_notes()), (BASS, bass_notes()),
                     (DRUMS, drum_notes())):
        total += op("notes_add", path=clip(t), notes=notes)["added"]
        pause(0.8)
    say("%d notes over %d bars" % (total, BARS))

    op("call", path="live_set scenes 0", function="fire", args=[])
    op("call", path="live_set", function="start_playing", args=[])
    say("playing: %s" % op("get", path="live_set",
                           property="is_playing")["value"])


# ------------------------------------------------------------------- beat 2

def beat2_sweep():
    announce(2, "a filter sweep lands as clip automation on Drift's LP Freq")
    # Look it up rather than guess: Drift calls it "LP Freq", not "Cutoff".
    hits = op("search", type="DeviceParameter",
              root="live_set tracks %d" % BASS, max_results=300)["results"]
    lp = [h["path"] for h in hits if h.get("name") == "LP Freq"]
    if not lp:
        raise SystemExit("no LP Freq on the Bass track - did Drift load?")
    param = lp[0]
    say("LP Freq is %s (of %d parameters)" % (param, len(hits)))

    # Put the envelope on screen BEFORE writing into it, or the whole beat
    # happens off camera: setting detail_clip alone leaves Clip View on the
    # Notes tab, showing whichever clip Live last selected. Only
    # select_envelope_parameter + show_envelope open the Envelopes tab on the
    # parameter we are about to automate.
    # Put the envelope on screen BEFORE writing into it, or the whole beat
    # happens off camera. Open Clip View first, then point detail_clip at the
    # bass clip, then drive the panel through detail_clip's OWN view - going via
    # the clip path instead leaves Clip View on the Notes tab of whatever Live
    # had selected, which is how the first take lost this beat entirely.
    show("Detail/Clip")
    op("set", path="live_set view", property="detail_clip",
       value={"__path__": clip(BASS)})
    view = "live_set view detail_clip view"
    op("call", path=view, function="show_envelope", args=[])
    op("call", path=view, function="select_envelope_parameter",
       args=[{"__path__": param}])
    pause(1.5)

    for bar, value in enumerate(SWEEP):
        op("envelope_insert_step", path=clip(BASS), parameter=param,
           time=float(bar * BEATS_PER_BAR), length=float(BEATS_PER_BAR),
           value=value)
        say("  bar %d -> %.2f" % (bar + 1, value))
        pause(0.9)

    env = op("envelope_get", path=clip(BASS), parameter=param, samples=9)
    say("read back: %s, range %s-%s" % (env["parameter_name"], env["min"],
                                        env["max"]))
    for pt in env["points"]:
        say("  beat %5.1f -> %.3f" % (pt["time"], pt["value"]))


# ------------------------------------------------------------------- beat 3

def beat3_observers():
    announce(3, "a fader moved outside the agent comes back through an observer")
    show("Session")
    # Select the track whose fader is about to move, so the change is on camera.
    op("set", path="live_set view", property="selected_track",
       value={"__path__": "live_set tracks %d" % CHORDS})
    vol = "live_set tracks %d mixer_device volume" % CHORDS
    op("observe_add", path=vol, property="value")
    say("watching %s value (now %.3f)"
        % (vol, op("get", path=vol, property="value")["value"]))

    # The registry and its ring buffer live on the ControlSurface, so they are
    # shared by every client attached to Live - a poll with consume=true would
    # drain someone else's events and observe_clear would tear down their
    # listeners. Take a sequence baseline and only ever read forward from it.
    base = op("observe_poll", limit=1)["latest_seq"]
    pause(2)

    # A SEPARATE process, so the event cannot be an echo of our own socket.
    subprocess.run([sys.executable, LOMCLI, "set", vol, "value", "0.6"],
                   check=True, stdout=subprocess.DEVNULL)
    pause(1.5)

    ev = op("observe_poll", since=base)
    mine = [e for e in ev["events"] if e.get("path") == vol]
    say("%d new event(s), %d dropped; %d on the fader"
        % (ev["count"], ev["dropped_events"], len(mine)))
    for e in mine:
        say("  seq %s  %s %s = %s" % (e.get("seq"), e.get("path"),
                                      e.get("property"), e.get("value")))
    pause(1.5)

    # Remove only ours: a listener Live still holds is a leak, but someone
    # else's listener is not ours to drop.
    op("observe_remove", path=vol, property="value")
    op("call", path="live_set", function="stop_playing", args=[])


def main():
    global START
    precondition()
    START = time.time()
    beat1_loop()
    pause(3)
    beat2_sweep()
    pause(3)
    beat3_observers()
    say()
    say("done at t=%.1fs - the built Set IS the output. It is unsaved; nothing "
        "was saved, opened or closed." % (time.time() - START))
    return 0


if __name__ == "__main__":
    sys.exit(main())
