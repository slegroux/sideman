#!/usr/bin/env python3
"""Live checks for the live-listen plan (sideman/19), run from a worktree.

    .venv/bin/python scripts/live_listen_check.py --item 1     # one device, -20 LUFS tone
    .venv/bin/python scripts/live_listen_check.py --item 3     # 30-device kill test (~8 min)

Talks to Live through sideman.client (the Remote Script on 127.0.0.1:9878)
and receives the device's rows with its own stream.Receiver, so it needs
neither the MCP server nor the new tools installed there.

Safety: refuses unless the open Set is unsaved (live_set file_path empty), so
it never runs on a real project. It only creates audio tracks named __sml_*
at the end, and deletes only those (expect=<their name>), last first.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np          # noqa: E402
import soundfile as sf      # noqa: E402

from sideman import stream  # noqa: E402
from sideman.client import Live, LiveError  # noqa: E402

TRACK = "__sml_check"


def _build():
    spec = importlib.util.spec_from_file_location("build_amxd", ROOT / "scripts" / "build_amxd.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def value(res):
    return res.get("value") if isinstance(res, dict) else res


def refuse_saved_set(live: Live) -> int:
    path = value(live.get("live_set", "file_path"))
    if path:
        raise SystemExit("refusing: the open Set is saved (%s). Open a new, unsaved Set "
                         "(File > New Live Set) and run again." % path)
    return int(live.count("live_set", "tracks")["count"])


def fixture(sr: int) -> pathlib.Path:
    """The stereo 997 Hz sine at amplitude 0.1 that test_listen.py reads as
    -20.0 LUFS, 4 s, written as a WAV for Live to play."""
    out = ROOT / "build" / "m4l" / "fixture_997hz_-20lufs_%d.wav" % sr
    out.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(4 * sr)) / sr
    x = 0.1 * np.sin(2 * np.pi * 997 * t)
    sf.write(str(out), np.stack([x, x], axis=1).astype(np.float32), sr, subtype="FLOAT")
    return out


def find_device(live: Live, browser_path: str) -> bool:
    parent, name = browser_path.rsplit("/", 1)
    got = live.browser_list(parent)
    items = got.get("items", []) if isinstance(got, dict) else []
    names = [i.get("name") if isinstance(i, dict) else i for i in items]
    return name in names


def item1(args) -> int:
    b = _build()
    live = Live()
    n_tracks = refuse_saved_set(live)
    sr = 48000          # the file's rate; Live resamples it to the engine's if needed
    print("open Set is unsaved; %d tracks" % n_tracks)
    if value(live.get("live_app browser", "hotswap_target")) is not None:
        raise SystemExit("refusing: Live's browser is in hot-swap mode; leave it first")

    installed = b.install()
    bpath = b.browser_path(installed.stem)
    print("installed %s" % installed)
    seen = False
    for _ in range(20):                      # Live indexes a new file within seconds
        if find_device(live, bpath):
            seen = True
            break
        time.sleep(0.5)
    print("browser_list finds %s: %s" % (bpath, seen))
    if not seen:
        raise SystemExit("the browser does not show the device; rescan the User Library "
                         "in Live's Preferences > Library, then run again")

    was_playing = bool(value(live.get("live_set", "is_playing")))
    rec = stream.Receiver()
    print("receiver: %s on UDP %d" % (rec.start(), rec.port))
    wav = fixture(sr)
    idx = None
    try:
        live.call("live_set", "create_audio_track", [-1])
        idx = int(live.count("live_set", "tracks")["count"]) - 1
        tp = "live_set tracks %d" % idx
        live.set(tp, "name", TRACK)
        live.call(tp + " clip_slots 0", "create_audio_clip", [str(wav)])
        live.set(tp + " clip_slots 0 clip", "looping", True)
        t0 = time.monotonic()
        live.browser_load(bpath, track_index=idx)
        dev = None
        for _ in range(100):
            nd = int(live.count(tp, "devices")["count"])
            if nd:
                dev = "%s devices %d" % (tp, nd - 1)
                break
            time.sleep(0.05)
        if dev is None:
            raise SystemExit("no device appeared on the track")
        name = value(live.get(dev, "name"))
        print("device loaded in %.2f s: %s (%s)" % (time.monotonic() - t0, dev, name))
        if name != installed.stem:
            raise SystemExit("the track's last device is %r, not %r" % (name, installed.stem))
        names = []
        for i in range(int(live.count(dev, "parameters")["count"])):
            names.append(value(live.get("%s parameters %d" % (dev, i), "name")))
        print("device parameters: %s" % names)
        if "Slot" in names:
            live.set("%s parameters %d" % (dev, names.index("Slot")), "value", 1)
            print("Slot parameter written: 1")
        else:
            print("Slot parameter NOT exposed in the LOM; rows will carry slot 0")
        live.call(tp + " clip_slots 0", "fire")
        if not value(live.get("live_set", "is_playing")):
            live.set("live_set", "is_playing", True)
        time.sleep(5.0)
        meter = value(live.get(tp, "output_meter_level"))
        info = rec.slots()
        print("rows heard per slot: %s" % {s: (d["sr"], d["nonces"]) for s, d in info.items()})
        print("track output_meter_level with the device: %.3f" % meter)
        slot = 1 if 1 in info else (sorted(info)[0] if info else None)
        if slot is None:
            print("NO ROWS RECEIVED (bad packets: %d). Open the device (wrench icon) and read "
                  "the Max console: listen.js posts the ring name it bound; if it reads ---sml, "
                  "Live did not make the name unique, and if nothing is posted the js file was "
                  "not found next to the device." % rec.bad_packets)
            ok = False
        else:
            a = stream.analyze_rows(rec.rows(slot, last_s=3.0))
            print("slot %d: momentary %s LUFS, short-term %s LUFS, peak %s dB, seq gaps %d"
                  % (slot, a["momentary_lufs"], a["short_term_lufs"], a["sample_peak_db"],
                     info[slot]["seq_gaps"]))
            ok = (meter > 0 and a["momentary_lufs"] is not None
                  and abs(a["momentary_lufs"] + 20.0) <= 0.5)
        print("ITEM 1 LIVE CHECK: %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        cleanup(live, idx, was_playing)
        rec.stop()
        try:
            print("tracks now: %d (was %d)" % (int(live.count("live_set", "tracks")["count"]),
                                              n_tracks))
        except LiveError as e:
            print("could not recount tracks: %s" % e)


def cleanup(live: Live, idx: int | None, was_playing: bool) -> None:
    """Stop the clip, delete our track (only if it is still named TRACK), put
    the transport back. Each step runs even if the one before it failed."""
    if idx is not None:
        tp = "live_set tracks %d" % idx
        try:
            live.call(tp + " clip_slots 0", "stop")
        except LiveError as e:
            print("could not stop the clip: %s" % e)
        try:
            live.call("live_set", "delete_track", [idx], expect=TRACK)
        except LiveError as e:
            print("cleanup failed: %s (delete the %s track by hand)" % (e, TRACK))
    try:
        if not was_playing and value(live.get("live_set", "is_playing")):
            live.set("live_set", "is_playing", False)
    except LiveError as e:
        print("could not restore the transport: %s" % e)


# ------------------------------------------------------------------ item 3

KT = "__sml_kt_%02d"
REPORT = pathlib.Path.home() / ".omc" / "reports" / "sideman-live-listen" / "kill-test-2026-11.md"


def add_track(live: Live, name: str, wav: pathlib.Path) -> int:
    live.call("live_set", "create_audio_track", [-1])
    idx = int(live.count("live_set", "tracks")["count"]) - 1
    tp = "live_set tracks %d" % idx
    live.set(tp, "name", name)
    live.call(tp + " clip_slots 0", "create_audio_clip", [str(wav)])
    live.set(tp + " clip_slots 0 clip", "looping", True)
    return idx


def load_last(live: Live, idx: int, bpath: str, want: str) -> str:
    """Load a browser item onto track idx and return its device path, after
    checking it landed last and is the one asked for."""
    tp = "live_set tracks %d" % idx
    before = int(live.count(tp, "devices")["count"])
    live.browser_load(bpath, track_index=idx)
    for _ in range(200):
        n = int(live.count(tp, "devices")["count"])
        if n == before + 1:
            dev = "%s devices %d" % (tp, n - 1)
            name = value(live.get(dev, "name"))
            if name != want:
                raise SystemExit("track %d's last device is %r, not %r" % (idx, name, want))
            return dev
        time.sleep(0.05)
    raise SystemExit("no device appeared on track %d" % idx)


def param_index(live: Live, dev: str, pred) -> int | None:
    for i in range(int(live.count(dev, "parameters")["count"])):
        if pred(str(value(live.get("%s parameters %d" % (dev, i), "name")))):
            return i
    return None


def cpu(live: Live, seconds: float) -> float:
    """Mean of Application.average_process_usage (percent) sampled every 0.5 s."""
    vals = []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        vals.append(float(value(live.get("live_app", "average_process_usage"))))
        time.sleep(0.5)
    return sum(vals) / len(vals)


def item3(args) -> int:
    b = _build()
    live = Live()
    n_tracks = refuse_saved_set(live)
    if value(live.get("live_app browser", "hotswap_target")) is not None:
        raise SystemExit("refusing: Live's browser is in hot-swap mode; leave it first")
    was_playing = bool(value(live.get("live_set", "is_playing")))
    installed = b.install(dummy=args.dummy)
    bpath = b.browser_path(installed.stem)
    for _ in range(20):
        if find_device(live, bpath):
            break
        time.sleep(0.5)
    else:
        raise SystemExit("the browser does not show %s; rescan the User Library" % bpath)
    wav = fixture(48000)
    lines = ["# Kill test, plan sideman/19 item 3 (%s)" % time.strftime("%Y-%m-%d %H:%M"), "",
             "Device: `%s` (%d extra biquads per instance + a 48-tap FIR per channel), %d "
             "instances. Script: `scripts/live_listen_check.py --item 3`." % (
                 installed.name, args.dummy, args.n), ""]
    made: list[tuple[int, str]] = []
    rec = stream.Receiver()
    rec.start()
    verdict = []
    try:
        # 1. null test on two tracks, before anything else plays
        a = add_track(live, "__sml_null_a", wav)
        made.append((a, "__sml_null_a"))
        bb = add_track(live, "__sml_null_b", wav)
        made.append((bb, "__sml_null_b"))
        load_last(live, a, bpath, installed.stem)
        util = load_last(live, bb, "audio_effects/Utility", "Utility")
        for i in range(int(live.count(util, "parameters")["count"])):
            nm = str(value(live.get("%s parameters %d" % (util, i), "name")))
            if "Inv" in nm or "Phz" in nm:
                live.set("%s parameters %d" % (util, i), "value", 1)
        for t in (a, bb):
            live.call("live_set tracks %d clip_slots 0" % t, "fire")
        live.set("live_set", "is_playing", True)
        time.sleep(3.0)
        peak_master = 0.0
        for _ in range(10):
            peak_master = max(peak_master, float(value(
                live.get("live_set master_track", "output_meter_level"))))
            time.sleep(0.1)
        null_ok = peak_master == 0.0
        lines += ["## Null test", "", "Track with the device plus its copy phase-inverted through "
                  "Utility, no device: master `output_meter_level` max over 1 s = %.4f -> %s." % (
                      peak_master, "PASS" if null_ok else "FAIL"), ""]
        verdict.append(("null test", null_ok))
        for t in (a, bb):
            live.call("live_set tracks %d clip_slots 0" % t, "stop")

        # 2. CPU: n tracks playing, without then with the device
        idxs = []
        for k in range(args.n):
            idxs.append(add_track(live, KT % k, wav))
            made.append((idxs[-1], KT % k))
        for t in idxs:
            live.call("live_set tracks %d clip_slots 0" % t, "fire")
        time.sleep(2.0)
        base = cpu(live, args.cpu_s)
        t0 = time.monotonic()
        devs = []
        for k, t in enumerate(idxs):
            dev = load_last(live, t, bpath, installed.stem)
            si = param_index(live, dev, lambda nm: nm == "Slot")
            if si is not None:
                live.set("%s parameters %d" % (dev, si), "value", k + 1)
            devs.append(dev)
        load_s = time.monotonic() - t0
        time.sleep(2.0)
        loaded = cpu(live, args.cpu_s)
        delta = loaded - base
        cpu_ok = delta < 15.0
        lines += ["## CPU", "", "| | average_process_usage |", "|---|---|",
                  "| %d tracks playing, no device | %.1f %% |" % (args.n, base),
                  "| with %d devices | %.1f %% |" % (args.n, loaded),
                  "| delta | **%.1f points** (limit 15) -> %s |" % (delta, "PASS" if cpu_ok else "FAIL"),
                  "", "Loading %d devices took %.1f s." % (args.n, load_s), ""]
        verdict.append(("CPU", cpu_ok))

        # 3. seq gaps under UI load
        t_end = time.monotonic() + args.gap_s
        events = []
        stage = 0
        while time.monotonic() < t_end:
            left = t_end - time.monotonic()
            if stage == 0 and left < args.gap_s * 0.75:
                live.call("app_view", "show_view", ["Browser"])
                events.append("browser shown")
                stage = 1
            elif stage == 1 and left < args.gap_s * 0.5:
                extra = load_last(live, idxs[0], "audio_effects/Utility", "Utility")
                events.append("Utility loaded on %s" % (KT % 0))
                live.call("live_set tracks %d" % idxs[0], "delete_device",
                          [int(extra.rsplit(" ", 1)[1])], expect="Utility")
                events.append("Utility deleted")
                stage = 2
            elif stage == 2 and left < args.gap_s * 0.25:
                live.call("app_view", "hide_view", ["Browser"])
                events.append("browser hidden")
                stage = 3
            time.sleep(0.5)
        info = rec.slots()
        heard = sorted(s for s in info if 1 <= s <= args.n)
        gaps = {s: info[s]["seq_gaps"] for s in heard}
        gaps_ok = len(heard) == args.n and not any(gaps.values())
        lines += ["## Dropped rows", "", "%.0f s with %d devices; %s." % (
                      args.gap_s, args.n, ", ".join(events) or "no UI events"),
                  "Slots heard: %d of %d. Rows lost (seq gaps, js and UDP together): %d "
                  "(per slot: %s). Bad packets: %d. -> %s" % (
                      len(heard), args.n, sum(gaps.values()),
                      {s: g for s, g in gaps.items() if g} or "none", rec.bad_packets,
                      "PASS" if gaps_ok else "FAIL"), ""]
        verdict.append(("no dropped rows", gaps_ok))

        # 4. block grid alignment between devices (host_sample phase of block starts)
        phases = {}
        for s in heard:
            r = rec.rows(s, last_s=1.0)
            if len(r):
                hs, n = int(r.h("host_sample")[-1]), int(r.h("n")[-1])
                phases[s] = hs % n
        spread = (max(phases.values()) - min(phases.values())) if phases else None
        lines += ["## Block alignment", "", "Block-start phase (host_sample mod n) across devices: "
                  "spread %s samples. Free-running blocks need not align; item 5 aligns 25 ms "
                  "cells by host_sample. Lookahead-latency offsets need sub-block resolution and "
                  "are measured in item 5." % spread, ""]

        # 5. transport behaviour
        r0 = rec.rows(1, last_s=0.5)
        live.set("live_set", "is_playing", False)
        time.sleep(1.5)
        r1 = rec.rows(1, last_s=1.0)
        stopped = ("rows keep coming: %d in 1 s; playing=%s; beat %s; host_sample advances: %s"
                   % (len(r1), sorted(set(r1.h("playing").tolist())),
                      "constant" if len(set(r1.h("beat_x1e6").tolist())) <= 1 else "moving",
                      bool(len(r1) > 1 and np.all(np.diff(r1.h("host_sample")) > 0))))
        live.set("live_set", "loop_start", 0.0)
        live.set("live_set", "loop_length", 4.0)
        live.set("live_set", "loop", True)
        live.set("live_set", "current_song_time", 0.0)
        live.set("live_set", "is_playing", True)
        time.sleep(5.0)
        r2 = rec.rows(1, last_s=4.0)
        beats = r2.beat
        looping = "beat wraps %d times in 4 s (a 1-bar loop at the Set's tempo)" % int(
            np.sum(np.diff(beats) < 0)) if len(beats) > 1 else "no rows"
        live.set("live_set", "loop", False)
        slots_seen = sorted(set(rec.rows(1, last_s=3.0).h("playing_slot").tolist()))
        lines += ["## Transport (plugsync~)", "",
                  "- playing: rows before stop %d; playing=%s" % (
                      len(r0), sorted(set(r0.h("playing").tolist()))),
                  "- stopped: %s" % stopped,
                  "- looping: %s" % looping,
                  "- Session clip playing in slot 0: playing_slot values %s" % slots_seen, ""]
        ok = all(v for _, v in verdict)
        lines += ["## Verdict", "", "%s. %s" % (
            "PASS" if ok else "STOP", ", ".join("%s %s" % (k, "ok" if v else "FAILED")
                                                for k, v in verdict)), ""]
        if not ok:
            lines.append("Plan rule: over 15 points, any dropped row, or a failed null -> STOP and "
                         "re-plan (master-only fallback).")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(lines))
        print("\n".join(lines))
        print("report: %s" % REPORT)
        return 0 if ok else 1
    finally:
        for idx, name in sorted(made, reverse=True):
            try:
                live.call("live_set tracks %d clip_slots 0" % idx, "stop")
            except LiveError:
                pass
            try:
                live.call("live_set", "delete_track", [idx], expect=name)
            except LiveError as e:
                print("cleanup failed for %s: %s (delete it by hand)" % (name, e))
        try:
            live.set("live_set", "is_playing", was_playing)
        except LiveError:
            pass
        rec.stop()
        try:
            print("tracks now: %d (was %d)" % (int(live.count("live_set", "tracks")["count"]),
                                              n_tracks))
        except LiveError as e:
            print("could not recount tracks: %s" % e)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--item", type=int, required=True, choices=[1, 3])
    ap.add_argument("--n", type=int, default=30, help="item 3: devices")
    ap.add_argument("--dummy", type=int, default=180, help="item 3: extra biquads per device")
    ap.add_argument("--cpu-s", type=float, default=15.0, help="item 3: seconds per CPU reading")
    ap.add_argument("--gap-s", type=float, default=300.0, help="item 3: seconds of the drop test")
    a = ap.parse_args(argv)
    return {1: item1, 3: item3}[a.item](a)


if __name__ == "__main__":
    sys.exit(main())
