# ableton-mcp-lom

An Ableton Live MCP server with **complete Live Object Model coverage**.

Every other Ableton MCP server hand-writes one tool per property, so coverage is
bounded by human labour and goes stale each Live release. This one exposes the
Live Object Model itself — the same generic `path` / `get` / `set` / `call`
contract Max for Live's `live.object` uses — so any property Live has is
reachable without shipping new code.

- **[docs/TOOLS.md](docs/TOOLS.md)** — all 29 tools with signatures (generated)
- **[docs/PROTOCOL.md](docs/PROTOCOL.md)** — wire protocol, threading contract, how to add an op
- **[skills/ableton-lom/SKILL.md](skills/ableton-lom/SKILL.md)** — path grammar for the model

## Why

Measured tool counts of the field (actual wired registrations, not README claims):

| Server | Tools | Bridge |
|---|---|---|
| jpoindexter/ableton-mcp | 128 | Remote Script |
| uisato/ableton-mcp-extended | 46 | Remote Script |
| xiaolaa2/ableton-copilot-mcp | 42 | ableton-js |
| ahujasid/ableton-mcp | ~21 | Remote Script |
| Simon-Kansara/ableton-live-mcp-server | 1 | AbletonOSC |

Live 12.2.7 exposes **47 reachable types / 922 members** by measurement (see
`baseline/`). Nobody covers that by enumeration.

Coverage is verified, not claimed: `scripts/coverage_harness.py` checks every
Live API attribute the three Remote Script competitors touch against our census
and **exits nonzero if any is unaccounted for**. Currently 0 unaccounted.
Not verified for xiaolaa2 (ableton-js) or Simon-Kansara (OSC) — different
bridges, out of scope for that method.

## Architecture

```
Claude ──MCP──► mcp_server/server.py          29 tools
                    │ JSON/TCP :9878          (not 9877 — coexists with others)
                    ▼
                remote_script/AbletonLOM/
                    ├── __init__.py    thin socket shell   (restart Live to change)
                    └── handlers.py    LOM engine, 30 ops  (hot-reload, no restart)
```

**Threading contract:** accept-thread → client-thread → `schedule_message(0, task)`.
The Live API is only ever touched on Live's main thread; the client thread blocks
on a `Queue`. Violating this freezes the GUI.

**Hot reload:** `handlers.py` is re-`importlib.reload`ed by the `reload` op, so
engine changes need no Live restart. Only the socket shell does.

## Install

```bash
git clone <repo> && cd ableton-mcp-lom
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -e .     # note: explicit, or uv may target a conda env
./scripts/install.sh                       # symlinks into Ableton's User Library
```

Then in Ableton: **Preferences → Link, Tempo & MIDI → Control Surface** and pick
**AbletonLOM** in any free slot. Restart Live first if it was running when you
ran `install.sh` — Live only scans `Remote Scripts/` at startup.

Verify and capture the LOM map for your Live version:

```bash
./scripts/lomcli.py ping
./scripts/census.py           # writes baseline/lom_census_<version>.json
```

Register with Claude Code — **`--scope user`** matters, or the server is only
visible from this directory:

```bash
claude mcp add --scope user ableton-lom -- "$PWD/.venv/bin/python" -m mcp_server.server
```

Optionally install the skill, which teaches the model the path grammar:

```bash
ln -s "$PWD/skills/ableton-lom" ~/.claude/skills/ableton-lom
```

MCP servers load at client startup — **restart your session** before the tools
appear.

## Tools

29 tools. Generic first — the typed ones exist only where generic access
genuinely cannot work. Full signatures in **[docs/TOOLS.md](docs/TOOLS.md)**.

| Tool | Purpose |
|---|---|
| `lom_search` | Find paths by name or type. **Start here.** |
| `lom_describe` | Properties, children, functions, per-instance availability |
| `lom_get` / `lom_set` / `lom_call` | Read / write / invoke anything |
| `lom_get_batch` / `lom_set_batch` | Many ops, one round trip |
| `lom_transaction` | Several ops, one undo step (see caveat below) |
| `lom_canonical_path` | Resolve aliases (`view selected_track` → `tracks 3`) |
| `lom_count` / `lom_types` / `lom_ping` | Collection size / census / health |
| `clip_get_notes` / `add` / `modify` / `remove` | MIDI notes (typed) |
| `browser_list` / `browser_load` | Library, incl. VST/AU/VST3 |
| `clip_envelope_get` / `insert_step` / `clear` | Clip automation |
| `arrangement_create_clip` / `duplicate_clip` / `list_clips` | Arrangement authoring |
| `lom_observe` / `lom_poll_events` / `lom_observers` / `lom_unobserve*` | Change notification |

**Observers are unique to this server.** Nothing else in the field reports
changes *the user* makes in the GUI.

## Paths

Space-separated, zero-indexed, unquoted:

```
live_set
live_set tracks 0 mixer_device volume
live_set tracks 2 clip_slots 0 clip
live_app browser instruments
```

Roots: `live_set`, `live_app`, `app_view`.

## Three measured behaviours

**Availability is per instance, not per type.** A member can exist on a type but
raise on a given object — MIDI tracks have no `input_meter_left`, only the main
track has a `crossfader`. `lom_describe` reports these in an `unavailable` map
rather than pretending the type list is uniformly gettable.

**Undo grouping is partial.** `lom_transaction` collapses ops into one Cmd-Z —
except automatable parameters, which Live always gives their own undo step:

| Groups | Own step |
|---|---|
| track `name`, `color_index`, `signature_numerator` | `tempo`, mixer `volume`, `mute`, device parameters |

**Destructive calls are guarded.** Anything named `delete_*`, `remove_*` or
`clear_*`, plus `crop`, refuses to run unless `confirm=true`. Derived from a
pattern, not a hand-kept list — the list had drifted, guarding two names Live 12
does not have while missing eleven it does.

## Tests

```bash
./scripts/test.sh
```

| Suite | Needs Live | Seam it covers |
|---|---|---|
| `tests/test_mcp.py` | no | MCP layer with a **stubbed socket** — op mapping, optional params, safe destructive defaults, error translation |
| `tests/test_e2e.py` | yes | **MCP layer → socket → Live** — the path Claude actually takes |
| `tests/smoke.py` | yes | engine via a **raw socket** — all 30 ops (~70s; loads a real device and cycles scratch tracks) |

The three cover deliberately different seams: `test_mcp` never touches Live,
`smoke` never loads the MCP layer, and only `test_e2e` exercises both together.

The Live-dependent suites create a scratch MIDI track named `__lomtest` and
delete it, so they work on an all-audio Set. They refuse to delete a track whose
name changed underneath them, and never touch existing tracks.

Only `test_mcp.py` is CI-able — a Remote Script cannot be exercised without its
host, so the rest are integration by necessity.

## Troubleshooting

**`ConnectionRefusedError` / nothing on 9878** — Live is not running, or
`AbletonLOM` is not selected as a Control Surface. Restarting Live or switching
Sets can clear the slot. Check with:

```bash
grep "Control Surface" ~/Library/Preferences/Ableton/Live*/Log.txt | tail -7
```

**`claude mcp list` says `✔ Connected` but tools fail** — that tick only means
the stdio process launches. It says nothing about Live. `lom_ping` is the real
health check.

**Tools do not appear at all** — MCP servers load at client startup; restart the
session. If they only appear in one directory, the server was registered at
project scope: re-add with `--scope user`.

**`uv pip install` seems to do nothing** — if `CONDA_PREFIX` is set, `uv` targets
the conda env instead of `.venv`. Pass `VIRTUAL_ENV=.venv` explicitly.

**"has no member X"** — usually correct. Several properties from older Live
releases are gone in 12 (`follow_action_a`, `fade_in_start`, `freeze`,
`create_group_track`). `lom_describe` is the source of truth.

**After upgrading Live** — rerun `./scripts/census.py`. The coverage harness
validates against the census, so a stale one silently checks an API that no
longer exists.

## Development

```bash
./scripts/lomcli.py ping                       # health
./scripts/lomcli.py describe "live_set"        # explore
./scripts/lomcli.py reload                     # after editing handlers.py
./scripts/census.py                            # refresh the LOM baseline
./scripts/gen_docs.py                          # regenerate docs/TOOLS.md
./scripts/coverage_harness.py                  # verify the superset claim
```

`lomcli.py` speaks the wire protocol directly, so the engine can be exercised
without the MCP layer in the way. See **[docs/PROTOCOL.md](docs/PROTOCOL.md)**
for the protocol and how to add an op.

Deliberately **no curated per-feature tool layer**. The plan called for ~30 sugar
verbs; at 29 generic tools that would mean 59, and tool overload degrades
selection. The skill buys the same ergonomics at zero tool cost.

## Status

Phases 0–8 complete. Plan and full build log:
`~/.omc/plans/2026-08-01-ableton-mcp-superset.md`.
