#!/usr/bin/env python3
"""Build the Sideman Listen Max for Live device, and install it.

    python scripts/build_amxd.py              # build/m4l/Sideman Listen.amxd (+ .maxpat, listen.js)
    python scripts/build_amxd.py --install    # ... and copy .amxd + listen.js into the User Library
    python scripts/build_amxd.py --dummy 180  # kill-test variant "Sideman Listen KT" (item 3):
                                              # 180 extra biquads per instance (90 per channel)
                                              # plus a 48-tap FIR per channel

The patcher is generated here, not saved from Max, so it can be reviewed as a
diff. The audio path is plugin~ -> plugout~, untouched; gen~ only listens.

.amxd container (as Max 9 writes an unfrozen audio effect, and as the user's
own builders do, ~/Projects/creative/dreamfauna-og/m4l/build_devices.py):

    'ampf' u32(4) 'aaaa'  |  'meta' u32(4) u32(1)  |  'ptch' u32(len) <patcher JSON> NUL

listen.js is not embedded: it sits next to the .amxd and the js box names it
bare, so Max finds it in the device's own folder. The User Library comes from
Live's Library.cfg, never a hardcoded path.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
M4L = ROOT / "sideman" / "m4l"
NAME = "Sideman Listen"
RING = 32                 # rows in the device's ring buffer: 3.2 s of slack for js
HDR = 8                   # seq playing playing_slot beat hs_hi hs_lo sr n
N_VALUES = 77             # must equal sideman.stream.N_VALUES
W = HDR + N_VALUES
PORT = 9071
GEN_INS = 6               # L R beat host_sample playing playing_slot
BROWSER_DIR = ("Presets", "Audio Effects", "Max Audio Effect")


# ------------------------------------------------------------------ sources

def genexpr(dummy: int = 0) -> str:
    """The codebox code, with constants filled in. `dummy` > 0 adds the
    kill-test load (item 3): `dummy` biquad sections per instance, split
    between the channels, plus a 48-tap FIR per channel standing in for the 4x
    true-peak interpolator. Their output feeds a value column so the compiler
    cannot drop them."""
    code = (M4L / "listen.genexpr").read_text()
    decl, body = [], []
    if dummy:
        per = max(1, dummy // 2)
        for ch in "lr":
            for i in range(per):
                v = "d%s%d" % (ch, i)
                decl.append("History %sx1(0), %sx2(0), %sy1(0), %sy2(0);" % (v, v, v, v))
                g = 0.05 + 0.9 * i / max(1, per)      # a 2-pole low-pass, different per section
                b0, a1, a2 = g * g / 4, 2 * (1 - g), (1 - g) ** 2
                body.append("%sy = %.6f * (%s + 2 * %sx1 + %sx2) + %.6f * %sy1 - %.6f * %sy2;"
                            % (v, b0, ch, v, v, a1, v, a2, v))
                body.append("%sx2 = %sx1; %sx1 = %s; %sy2 = %sy1; %sy1 = %sy;"
                            % (v, v, v, ch, v, v, v, v))
                body.append("dm = dm + %sy * %sy;" % (v, v))
            # 12-sample delay line read at 4 phases: 48 multiply-adds per sample
            taps = ["f%s%d" % (ch, k) for k in range(12)]
            decl.append("History %s;" % ", ".join("%s(0)" % t for t in taps))
            terms = " + ".join("%.5f * %s" % (0.01 * (1 + (k * 7 + ph) % 11), taps[k])
                               for ph in range(4) for k in range(12))
            body.append("fir%s = %s;" % (ch, terms))
            body.append("dm = dm + fir%s * fir%s;" % (ch, ch))
            body.append("; ".join("%s = %s" % (taps[k], taps[k - 1]) for k in range(11, 0, -1))
                        + "; %s = %s;" % (taps[0], ch))
    code = code.replace("@DUMMY_DECL@", "\n".join(decl))
    code = code.replace("@DUMMY_BODY@", "\n".join(body))
    for k, v in (("W", W), ("RING", RING), ("NVALUES", N_VALUES)):
        code = code.replace("@%s@" % k, str(v))
    left = re.findall(r"@[A-Z_]+@", code)
    if left:
        raise SystemExit("unfilled placeholders in listen.genexpr: %s" % left)
    return code


def js() -> str:
    code = (M4L / "listen.js").read_text()
    return code.replace("@W@", str(W)).replace("@RING@", str(RING))


# ------------------------------------------------------------------ patcher

def _box(id_, text, ins, outs, rect, outtypes=None, **extra):
    b = {"id": id_, "maxclass": "newobj", "text": text, "numinlets": ins,
         "numoutlets": outs, "patching_rect": rect}
    if outs:
        b["outlettype"] = outtypes or [""] * outs
    b.update(extra)
    return {"box": b}


def _line(src, so, dst, di):
    return {"patchline": {"source": [src, so], "destination": [dst, di]}}


def gen_patcher(code: str) -> dict:
    boxes = [_box("g-in%d" % i, "in %d" % i, 0, 1, [20.0 + 60 * (i - 1), 20.0, 40.0, 22.0])
             for i in range(1, GEN_INS + 1)]
    boxes.append({"box": {"id": "g-code", "maxclass": "codebox", "code": code,
                          "fontface": 0, "fontname": "<Monospaced>", "fontsize": 12.0,
                          "numinlets": GEN_INS, "numoutlets": 1, "outlettype": [""],
                          "patching_rect": [20.0, 60.0, 700.0, 600.0]}})
    boxes.append(_box("g-out1", "out 1", 1, 0, [20.0, 680.0, 40.0, 22.0]))
    lines = [_line("g-in%d" % i, 0, "g-code", i - 1) for i in range(1, GEN_INS + 1)]
    lines.append(_line("g-code", 0, "g-out1", 0))
    return {"fileversion": 1,
            "appversion": {"major": 8, "minor": 6, "revision": 0, "architecture": "x64",
                           "modernui": 1},
            "classnamespace": "dsp.gen", "rect": [100.0, 100.0, 760.0, 720.0],
            "gridsize": [15.0, 15.0], "boxes": boxes, "lines": lines}


def patcher(dummy: int = 0) -> dict:
    code = genexpr(dummy)
    b = []
    L = []
    # audio: untouched pass-through; gen~'s zero output is summed in so that
    # the gen~ is part of the device's signal chain
    b.append(_box("in", "plugin~", 1, 3, [20.0, 20.0, 60.0, 22.0], ["signal", "signal", ""]))
    b.append(_box("out", "plugout~", 2, 0, [20.0, 400.0, 62.0, 22.0]))
    L += [_line("in", 0, "out", 0), _line("in", 1, "out", 1)]
    gen = _box("gen", "gen~", GEN_INS, 1, [120.0, 300.0, 300.0, 22.0], ["signal"])
    gen["box"]["patcher"] = gen_patcher(code)
    b.append(gen)
    L += [_line("in", 0, "gen", 0), _line("in", 1, "gen", 1), _line("gen", 0, "out", 0)]
    # transport: polled every 10 ms; beat, the sample count (one 64-bit
    # signal; gen~ splits it into 16-bit halves for the float32 ring) and
    # playing reach gen~ as signals
    b.append(_box("tick", "metro 10 @active 1", 2, 1, [120.0, 20.0, 110.0, 22.0], ["bang"]))
    b.append(_box("sync", "plugsync~", 1, 9, [120.0, 60.0, 200.0, 22.0],
                  ["int", "int", "int", "float", "list", "float", "float", "int", "int"]))
    L.append(_line("tick", 0, "sync", 0))
    b.append(_box("beat", "sig~", 1, 1, [260.0, 100.0, 40.0, 22.0], ["signal"]))
    b.append(_box("hs", "sig~", 1, 1, [320.0, 100.0, 40.0, 22.0], ["signal"]))
    b.append(_box("play", "sig~", 1, 1, [120.0, 100.0, 40.0, 22.0], ["signal"]))
    L += [_line("sync", 6, "beat", 0), _line("sync", 7, "hs", 0), _line("sync", 0, "play", 0),
          _line("beat", 0, "gen", 2), _line("hs", 0, "gen", 3), _line("play", 0, "gen", 4)]
    # this track's playing slot, observed through the Live API
    b.append(_box("dev", "live.thisdevice", 1, 3, [480.0, 20.0, 100.0, 22.0],
                  ["bang", "int", "int"]))
    b.append({"box": {"id": "pathmsg", "maxclass": "message",
                      "text": "path this_device canonical_parent", "numinlets": 2,
                      "numoutlets": 1, "outlettype": [""],
                      "patching_rect": [480.0, 60.0, 200.0, 22.0]}})
    b.append(_box("path", "live.path", 1, 3, [480.0, 100.0, 80.0, 22.0], ["", "", ""]))
    b.append(_box("obs", "live.observer playing_slot_index", 2, 2,
                  [480.0, 140.0, 200.0, 22.0], ["", ""]))
    b.append(_box("pslot", "sig~ -1", 1, 1, [480.0, 180.0, 50.0, 22.0], ["signal"]))
    L += [_line("dev", 0, "pathmsg", 0), _line("pathmsg", 0, "path", 0),
          _line("path", 0, "obs", 1), _line("obs", 0, "pslot", 0),
          _line("pslot", 0, "gen", 5)]
    # the per-instance ring: ---sml becomes a name unique to this device
    b.append(_box("buf", "buffer~ ---sml @samps %d" % (1 + RING * W), 1, 2,
                  [700.0, 20.0, 200.0, 22.0], ["float", "bang"]))
    b.append(_box("ringname", "loadmess ring ---sml", 1, 1, [700.0, 60.0, 140.0, 22.0]))
    b.append(_box("jsname", "loadmess bufname ---sml", 1, 1, [700.0, 100.0, 160.0, 22.0]))
    L += [_line("dev", 0, "ringname", 0), _line("ringname", 0, "gen", 0),
          _line("dev", 0, "jsname", 0)]
    # js drains the ring every 20 ms and sends the rows
    b.append(_box("drain", "metro 20 @active 1", 2, 1, [880.0, 140.0, 110.0, 22.0], ["bang"]))
    b.append(_box("js", "js listen.js", 1, 1, [700.0, 220.0, 120.0, 22.0]))
    b.append(_box("udp", "udpsend 127.0.0.1 %d" % PORT, 1, 0, [700.0, 260.0, 150.0, 22.0]))
    L += [_line("jsname", 0, "js", 0), _line("drain", 0, "js", 0), _line("js", 0, "udp", 0)]
    # Slot: written by listen_attach; tells the server which track this is
    b.append({"box": {
        "id": "slot", "maxclass": "live.numbox", "numinlets": 1, "numoutlets": 2,
        "outlettype": ["", "float"], "parameter_enable": 1, "varname": "Slot",
        "patching_rect": [880.0, 20.0, 44.0, 15.0], "presentation": 1,
        "presentation_rect": [4.0, 4.0, 44.0, 15.0],
        "saved_attribute_attributes": {"valueof": {
            "parameter_longname": "Slot", "parameter_shortname": "Slot",
            "parameter_type": 1, "parameter_mmin": 0.0, "parameter_mmax": 255.0,
            "parameter_initial_enable": 1, "parameter_initial": [0],
            "parameter_unitstyle": 0, "parameter_invisible": 0}}}})
    b.append(_box("slotmsg", "prepend slot", 1, 1, [880.0, 60.0, 80.0, 22.0]))
    L += [_line("slot", 0, "slotmsg", 0), _line("slotmsg", 0, "js", 0)]
    b.append({"box": {"id": "label", "maxclass": "comment",
                      "text": "Sideman Listen: analysis only, audio passes untouched",
                      "numinlets": 1, "numoutlets": 0, "presentation": 1,
                      "presentation_rect": [52.0, 2.0, 300.0, 20.0],
                      "patching_rect": [20.0, 460.0, 320.0, 20.0]}})
    return {"patcher": {
        "fileversion": 1,
        "appversion": {"major": 8, "minor": 6, "revision": 0, "architecture": "x64",
                       "modernui": 1},
        "classnamespace": "box", "rect": [40.0, 80.0, 1000.0, 520.0],
        "openinpresentation": 1, "gridsize": [15.0, 15.0], "devicewidth": 360.0,
        "boxes": b, "lines": L,
        "parameters": {"slot": ["Slot", "Slot", 0]},
        "dependency_cache": [{"name": "listen.js", "type": "TEXT", "implicit": 1}],
        "autosave": 0}}


def wrap_amxd(p: dict) -> bytes:
    payload = json.dumps(p).encode() + b"\x00"
    return (b"ampf" + struct.pack("<I", 4) + b"aaaa"
            + b"meta" + struct.pack("<I", 4) + struct.pack("<I", 1)
            + b"ptch" + struct.pack("<I", len(payload)) + payload)


def unwrap_amxd(blob: bytes) -> dict:
    i = blob.index(b"ptch")
    n = struct.unpack("<I", blob[i + 4:i + 8])[0]
    return json.loads(blob[i + 8:i + 8 + n].rstrip(b"\x00"))


# ------------------------------------------------------------------ install

def user_library() -> pathlib.Path:
    """The User Library folder from the newest Live's Library.cfg."""
    prefs = pathlib.Path.home() / "Library" / "Preferences" / "Ableton"
    cfgs = sorted(prefs.glob("Live */Library.cfg"),
                  key=lambda p: [int(x) for x in re.findall(r"\d+", p.parent.name)])
    if not cfgs:
        raise FileNotFoundError("no Live */Library.cfg under %s" % prefs)
    text = cfgs[-1].read_text(encoding="utf-8", errors="replace")
    sec = re.search(r"<UserLibrary>(.*?)</UserLibrary>", text, re.S)
    if not sec:
        raise FileNotFoundError("no <UserLibrary> in %s" % cfgs[-1])
    path = re.search(r'<ProjectPath Value="([^"]*)"', sec.group(1))
    name = re.search(r'<ProjectName Value="([^"]*)"', sec.group(1))
    if not path or not name:
        raise FileNotFoundError("no ProjectPath/ProjectName in %s" % cfgs[-1])
    lib = pathlib.Path(path.group(1)) / name.group(1)
    if not lib.is_dir():
        raise FileNotFoundError("User Library %s (from %s) does not exist" % (lib, cfgs[-1]))
    return lib


def browser_path(name: str = NAME) -> str:
    """The device's path for sideman's browser_list / browser_load."""
    return "user_library/" + "/".join(BROWSER_DIR) + "/%s.amxd" % name


def build(dummy: int = 0) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the .amxd, listen.maxpat (for reading or opening in Max) and the
    filled-in listen.js into build/m4l. Returns (amxd, js)."""
    name = NAME if not dummy else NAME + " KT"
    out = ROOT / "build" / "m4l"
    out.mkdir(parents=True, exist_ok=True)
    p = patcher(dummy)
    amxd = out / ("%s.amxd" % name)
    amxd.write_bytes(wrap_amxd(p))
    (out / ("%s.maxpat" % name)).write_text(json.dumps(p, indent=1))
    jsf = out / "listen.js"
    jsf.write_text(js())
    assert unwrap_amxd(amxd.read_bytes()) == p
    return amxd, jsf


def install(dummy: int = 0) -> pathlib.Path:
    """Build, then copy the .amxd and listen.js into the User Library folder
    that holds Max audio effects. Returns the installed .amxd."""
    amxd, jsf = build(dummy)
    dest = user_library().joinpath(*BROWSER_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(amxd, dest / amxd.name)
    shutil.copy2(jsf, dest / jsf.name)
    return dest / amxd.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--dummy", type=int, default=0)
    a = ap.parse_args(argv)
    if a.install:
        path = install(a.dummy)
        print("installed %s" % path)
        print("browser path: %s" % browser_path(path.stem))
    else:
        amxd, jsf = build(a.dummy)
        print("built %s and %s" % (amxd, jsf))
    return 0


if __name__ == "__main__":
    sys.exit(main())
