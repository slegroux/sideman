#!/usr/bin/env python3
"""Phase 7 coverage harness.

Turns "superset" into a test instead of a claim.

Method: name-matching tool lists would be circular (our tools are generic, so
every name would "match" nothing or everything). Instead this works at the
*member* level - it extracts the Live API attributes each competitor's bridge
actually touches, then checks them against our LOM census.

The output that matters is the RESIDUE: attributes a competitor uses that our
census does not contain. Those are candidate gaps, and they are triaged rather
than hidden.

Usage:  ./coverage_harness.py [--verbose]
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CENSUS = REPO / "baseline" / "lom_census_12.2.7.json"

SCRATCH = pathlib.Path(
    "/private/tmp/claude-501/-Users-slegroux/"
    "f6da4bf6-3332-49d3-9c2c-8776bc929239/scratchpad/abl"
)

# Only competitors that reach Live through a Python Remote Script are comparable
# by this method: their source contains literal Live API attribute accesses.
#
# EXCLUDED, with reason - counting them would produce a meaningless number:
#   xiaolaa2      goes through ableton-js, so its TypeScript contains JS builtins
#                 (.filter/.map/.catch), not Live attributes. Scored 10% here,
#                 which measures the extractor, not the coverage.
#   Simon-Kansara sends OSC address STRINGS ("/live/song/get/track_names") and
#                 never touches a Live object. Scored 0% for the same reason.
# Both are handled by the OSC/ableton-js note in the report instead.
COMPETITORS = {
    "jpoindexter/ableton-mcp": [
        SCRATCH / "jpoindexter_ableton-mcp/AbletonMCP_Remote_Script/__init__.py"],
    "uisato/ableton-mcp-extended": [
        pathlib.Path.home()
        / "Projects/vendor/ableton-mcp-extended/AbletonMCP_Remote_Script/__init__.py"],
    "ahujasid/ableton-mcp": [
        SCRATCH / "ahujasid_ableton-mcp/AbletonMCP_Remote_Script/__init__.py"],
}

# Members of Browser / BrowserItem. These are genuinely reachable (Phase 3's
# browser_list/browser_load use them) but do NOT appear in the census, because
# get_available_lom_types() registers 43 types and Browser is not one of them.
# A real census gap, tracked rather than hidden.
BROWSER_SURFACE = {
    "children", "is_device", "is_folder", "is_loadable", "display_name",
    "hotswap_target", "uri", "browser", "load_item", "instruments", "sounds",
    "drums", "audio_effects", "midi_effects", "plugins", "packs",
    "user_library", "current_project", "max_for_live", "source",
}

# Infrastructure of the competitor's own server, plus stdlib the attribute
# regex cannot distinguish from Live members. Not Live API.
INFRA = {
    "client_threads", "format_exc", "environ", "choice", "g", "flatten",
    "server", "server_thread", "socket", "sleep", "timeout", "running",
    "random", "randint", "pow", "isabs", "py", "x", "samples",
}

# Attributes that look like LOM members but belong to Python, the MCP/asyncio
# stack, or the competitor's own plumbing. Excluded from the residue so the
# genuine gaps are visible.
NOT_LOM = set("""
append extend insert remove pop clear copy count index sort reverse
get keys values items update setdefault popitem fromkeys
format join split strip lstrip rstrip replace startswith endswith lower upper
encode decode find rfind title capitalize splitlines zfill ljust rjust
read write close open flush seek tell readline readlines truncate fileno
send sendall recv accept bind listen connect settimeout setsockopt shutdown
put task_done join_thread empty full qsize get_nowait put_nowait
start run daemon is_alive acquire release wait notify set_result
dumps loads dump load
log_message show_message schedule_message song application
add_error done cancel result exception set_exception
now today strftime isoformat timestamp
match search sub findall finditer group groups compile
exists is_file is_dir mkdir rglob glob resolve parent name suffix stem
lstrip_prefix to_dict from_dict json dict model_dump
tool resource prompt call_tool list_tools run_stdio_async
info debug warning error critical exception setLevel addHandler
value_error type_error key_error
self cls args kwargs params kwargs_
""".split())

ATTR_RE = re.compile(r"\.([a-z_][a-z_0-9]*)\b")
LISTENER_RE = re.compile(r"^(?:add|remove)_(.+)_listener$|^(.+)_has_listener$")


def census_universe() -> tuple[set[str], dict]:
    data = json.loads(CENSUS.read_text())
    members: set[str] = set()
    for t in data["types"].values():
        members |= set(t["members"])
    return members, data


def is_listener_of(attr: str, members: set[str]) -> bool:
    m = LISTENER_RE.match(attr)
    if not m:
        return False
    base = m.group(1) or m.group(2)
    return base in members


def extract(paths: list[pathlib.Path]) -> tuple[set[str], int]:
    attrs: set[str] = set()
    seen = 0
    for p in paths:
        if not p.exists():
            continue
        seen += 1
        text = p.read_text(errors="replace")
        # Drop the competitor's own defined names - those are their plumbing,
        # not Live API surface.
        own = set(re.findall(r"^\s*(?:async\s+)?def\s+([a-z_][a-z_0-9]*)",
                             text, re.M))
        own |= set(re.findall(r"^\s*([a-z_][a-z_0-9]*)\s*[:=]", text, re.M))
        for a in ATTR_RE.findall(text):
            if a.startswith("_") or a in own or a in NOT_LOM:
                continue
            attrs.add(a)
    return attrs, seen


def main(verbose: bool = False) -> int:
    if not CENSUS.exists():
        print(f"missing census at {CENSUS}; run: lomcli.py types", file=sys.stderr)
        return 2
    members, data = census_universe()
    print("=" * 74)
    print("PHASE 7 COVERAGE HARNESS")
    print("=" * 74)
    print(f"Census: Live {CENSUS.stem.split('_')[-1]}  "
          f"{data['type_count']} types  {len(members)} substantive members")
    print()

    grand_residue: dict[str, set[str]] = {}
    rows = []
    for name, paths in COMPETITORS.items():
        attrs, nfiles = extract(paths)
        if not nfiles:
            rows.append((name, 0, 0, 0.0, "SOURCE MISSING"))
            continue
        covered = {a for a in attrs
                   if a in members or is_listener_of(a, members)}
        residue = attrs - covered
        pct = 100.0 * len(covered) / len(attrs) if attrs else 0.0
        rows.append((name, len(attrs), len(covered), pct, ""))
        grand_residue[name] = residue
        if verbose and residue:
            print(f"--- residue: {name} ({len(residue)}) ---")
            print("   " + ", ".join(sorted(residue)))
            print()

    print(f"{'competitor':40} {'attrs':>6} {'covered':>8} {'%':>7}")
    print("-" * 74)
    for name, tot, cov, pct, note in rows:
        print(f"{name:40} {tot:>6} {cov:>8} {pct:>6.1f}%  {note}")
    print()

    allres: set[str] = set()
    for r in grand_residue.values():
        allres |= r

    browser = sorted(allres & BROWSER_SURFACE)
    infra = sorted(allres & INFRA)
    unexplained = sorted(allres - BROWSER_SURFACE - INFRA)

    print(f"RESIDUE: {len(allres)} attributes competitors touch that the census lacks")
    print()
    print(f"  [A] Browser/BrowserItem surface ......... {len(browser):>3}  REAL CENSUS GAP")
    print("      Reachable today (Phase 3 browser_list/load use them) but absent")
    print("      from the census: get_available_lom_types() registers 43 types and")
    print("      Browser is not one. The census understates the reachable graph.")
    if browser:
        print("      " + ", ".join(browser))
    print()
    print(f"  [B] Competitor server infrastructure .... {len(infra):>3}  not Live API")
    if infra:
        print("      " + ", ".join(infra))
    print()
    print(f"  [C] UNEXPLAINED - needs live verification {len(unexplained):>3}")
    print("      Either genuinely absent from Live 12.2.7 (competitor targets an")
    print("      older Live and this is dead code), or a real gap. CANNOT be")
    print("      settled without Ableton running - do not assume either way.")
    for a in unexplained:
        who = [n.split("/")[0] for n, r in grand_residue.items() if a in r]
        print(f"      {a:32} used by: {', '.join(who)}")
    print()
    print("VERDICT: superset claim is CONDITIONAL pending [A] and [C].")
    return 0


if __name__ == "__main__":
    sys.exit(main("--verbose" in sys.argv))
