#!/usr/bin/env python3
"""Read a device's parameter units from Live into a JSON units table.

For every parameter of the device at DEVICE_PATH: its range, default, whether it is quantized (with its
value items), and, for a continuous one, what Live displays (`str_for_value`) at POINTS values across the
range. Nothing is written to the device: `str_for_value` only formats a value. The calls still go through
Sideman's undo wrapping, so they are batched (one undo step per CHUNK calls) and the device should be a
scratch one, not a device in the user's own tracks.

    uv run python scripts/operator_units.py "live_set tracks 53 devices 0" sideman/data/operator_units.json

The output holds `parameters` (everything read, with the display text as Live gave it) and `units`, the
table `sideman.patch.to_live` takes: {name: [[live_value, display_number, unit], ...]} for the continuous
parameters whose display parsed as a number with a known unit.
"""
import datetime
import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman.client import Live  # noqa: E402

POINTS = 49          # display values read across each continuous range (ends included)
CHUNK = 150          # str_for_value calls per transaction: one undo step each
KNOWN_UNITS = ("ms", "s", "Hz", "kHz", "dB", "%", "st")
NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)|[+-]?inf)\s*(.*?)\s*$")


def parse(text: str):
    """(number, unit) from Live's display text, or (None, None). "-inf dB" -> (-inf, "dB"); "1.50 s" -> (1.5, "s")."""
    m = NUMBER.match(text.replace("−", "-"))
    if not m:
        return None, None
    unit = m.group(2)
    return float(m.group(1)), unit


def main(device: str, out: str) -> None:
    live = Live()
    dev = live.get(device, "name")["value"]
    count = live.get(device, "parameters", limit=1)["value"]["count"]
    fields = ("name", "min", "max", "default_value", "is_quantized", "value")
    specs = [{"path": "%s parameters %d" % (device, i), "property": f} for i in range(count) for f in fields]
    flat = []
    for a in range(0, len(specs), 300):
        flat += live.get_batch(specs[a:a + 300])["results"]
    params = []
    for i in range(count):
        row = {f: flat[i * len(fields) + k].get("value") for k, f in enumerate(fields)}
        row["index"] = i
        params.append(row)
    calls = []
    for p in params:
        path = "%s parameters %d" % (device, p["index"])
        if p["is_quantized"]:
            try:
                items = live.get(path, "value_items", limit=512)["value"]
                p["value_items"] = items.get("items", items) if isinstance(items, dict) else items
            except Exception as e:                      # some quantized parameters expose no items
                p["value_items_error"] = str(e)
            continue
        lo, hi = float(p["min"]), float(p["max"])
        for k in range(POINTS):
            v = lo + (hi - lo) * k / (POINTS - 1)
            calls.append((p["index"], v, {"op": "call", "path": path, "function": "str_for_value", "args": [v]}))
    texts = {}
    for a in range(0, len(calls), CHUNK):
        part = calls[a:a + CHUNK]
        got = live.transaction([c[2] for c in part], stop_on_error=False)["results"]
        for (index, v, _), r in zip(part, got):
            res = r.get("result")
            text = res.get("result") if isinstance(res, dict) else res
            texts.setdefault(index, []).append([v, text if r.get("ok") else None])
    units, parameters = {}, {}
    for p in params:
        entry = {k: p[k] for k in ("index", "min", "max", "default_value", "is_quantized") if k in p}
        if "value_items" in p:
            entry["value_items"] = p["value_items"]
        if "value_items_error" in p:
            entry["value_items_error"] = p["value_items_error"]
        rows = texts.get(p["index"])
        if rows:
            entry["display"] = rows
            parsed = [(v, *parse(t)) for v, t in rows if isinstance(t, str)]
            good = [(v, n, u) for v, n, u in parsed if n is not None and u in KNOWN_UNITS + ("",) and not math.isnan(n)]
            if len(good) == len(rows) and len({u for _, n, u in good if math.isfinite(n)}) <= 2:
                units[p["name"]] = [[v, n if math.isfinite(n) else ("-inf" if n < 0 else "inf"), u] for v, n, u in good]
            else:
                entry["unparsed"] = [t for _, t in rows if not isinstance(t, str) or parse(t)[0] is None][:5]
        parameters[p["name"]] = entry
    doc = {"device": dev, "read": datetime.date.today().isoformat(), "points": POINTS,
           "parameters": parameters, "units": units}
    try:
        doc["live"] = live.call("live_app", "get_version_string")["result"]
    except Exception:
        pass                                            # the version is a convenience, not needed by to_live
    pathlib.Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    print("%s: %d parameters, %d continuous with a units table, %d quantized, %d str_for_value calls in %d undo steps"
          % (dev, count, len(units), sum(1 for p in params if p["is_quantized"]), len(calls),
             math.ceil(len(calls) / CHUNK)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
