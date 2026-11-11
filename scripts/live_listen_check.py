#!/usr/bin/env python3
"""Live checks for the live-listen plan (sideman/19), run from a worktree.

    .venv/bin/python scripts/live_listen_check.py --item 1

Talks to Live through sideman.client (the Remote Script on 127.0.0.1:9878)
and receives the device's rows with its own stream.Receiver, so it needs
neither the MCP server nor the new tools installed there.

Safety: refuses unless the open Set is unsaved (live_set file_path empty), so
it never runs on a real project. It creates one audio track named
__sml_check at the end, and deletes only that track (expect=<its name>).
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--item", type=int, required=True, choices=[1])
    a = ap.parse_args(argv)
    return {1: item1}[a.item](a)


if __name__ == "__main__":
    sys.exit(main())
