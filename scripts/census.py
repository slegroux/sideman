#!/usr/bin/env python3
"""Regenerate the LOM census baseline from the running Ableton Live.

The census is the coverage harness's map of what Live exposes. It is pinned to
a Live version because the LOM changes between releases - members disappear
(Live 12 dropped follow_action_*, freeze, create_group_track among others), so
a stale census silently compares against an API that no longer exists.

Run this after upgrading Live. The version is read from Live itself rather
than hardcoded, so the filename always matches reality.

  ./scripts/census.py           write baseline/lom_census_<version>.json
  ./scripts/census.py --check   exit 1 if the census is missing or stale
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPO / "baseline"
sys.path.insert(0, str(REPO / "scripts"))
from lomcli import request  # noqa: E402


def live_version():
    parts = []
    for fn in ("get_major_version", "get_minor_version", "get_bugfix_version"):
        r = request("call", {"path": "live_app", "function": fn, "args": []})
        if not r.get("ok"):
            raise RuntimeError("could not read Live version: %s" % r.get("error"))
        parts.append(str(r["result"]["result"]))
    return ".".join(parts)


def newest_census():
    """Newest baseline/lom_census_<version>.json, or None. Sorted by version
    number: as text, 12.10 would sort before 12.9."""
    def version(p):
        return tuple(int(x) for x in p.stem[len("lom_census_"):].split("."))
    files = sorted(BASELINE.glob("lom_census_*.json"), key=version)
    return files[-1] if files else None


def summarise(data):
    t = data["totals"]
    return ("%d types (%d registered + %d unregistered), %d substantive members"
            % (data["type_count"], t.get("registered_types", 0),
               t.get("unregistered_types", 0), t["substantive"]))


def diff(old, new):
    """What actually changed - the reason to regenerate at all."""
    o = {k: set(v["members"]) for k, v in old["types"].items()}
    n = {k: set(v["members"]) for k, v in new["types"].items()}
    added_types = sorted(set(n) - set(o))
    removed_types = sorted(set(o) - set(n))
    added, removed = [], []
    for k in sorted(set(o) & set(n)):
        for m in sorted(n[k] - o[k]):
            added.append("%s.%s" % (k.split(".")[-1], m))
        for m in sorted(o[k] - n[k]):
            removed.append("%s.%s" % (k.split(".")[-1], m))
    return added_types, removed_types, added, removed


def main():
    check_only = "--check" in sys.argv
    try:
        version = live_version()
    except Exception as e:
        print("Cannot reach Live: %s" % e, file=sys.stderr)
        print("Start Ableton Live with AbletonLOM enabled as a Control Surface.",
              file=sys.stderr)
        return 2

    target = BASELINE / ("lom_census_%s.json" % version)
    existing = newest_census()

    if check_only:
        if not target.exists():
            print("STALE: no census for Live %s (found %s)"
                  % (version, existing.name if existing else "none"))
            return 1
        print("OK: census matches running Live %s" % version)
        return 0

    print("Live %s - fetching census..." % version)
    r = request("types", {}, timeout=120)
    if not r.get("ok"):
        print("census failed: %s" % r.get("error"), file=sys.stderr)
        return 1
    data = r["result"]

    old = None
    if existing and existing != target:
        old = json.loads(existing.read_text())
    elif target.exists():
        old = json.loads(target.read_text())

    BASELINE.mkdir(exist_ok=True)
    target.write_text(json.dumps(data, indent=1))
    print("wrote %s" % target.relative_to(REPO))
    print("  %s" % summarise(data))

    if old:
        at, rt, added, removed = diff(old, data)
        if not (at or rt or added or removed):
            print("  no change vs previous census")
        else:
            print("  vs previous:")
            for t in at:
                print("    + type %s" % t)
            for t in rt:
                print("    - type %s" % t)
            for m in added[:20]:
                print("    + %s" % m)
            if len(added) > 20:
                print("    + ... %d more" % (len(added) - 20))
            for m in removed[:20]:
                print("    - %s" % m)
            if len(removed) > 20:
                print("    - ... %d more" % (len(removed) - 20))
            if removed:
                print("  NOTE: removed members may break tools that call them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
