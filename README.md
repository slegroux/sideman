# ableton-mcp-lom

An Ableton Live MCP server with **complete Live Object Model coverage**.

Every other Ableton MCP server hand-writes one tool per property, so coverage is
bounded by human labour and goes stale each Live release. This one exposes the
Live Object Model itself — the same generic `path` / `get` / `set` / `call`
contract Max for Live's `live.object` uses — so any property Live has is
reachable without shipping new code.

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
Claude ──MCP──► mcp_server/server.py
                    │ JSON/TCP :9878   (not 9877 — coexists with other servers)
                    ▼
                remote_script/AbletonLOM/
                    ├── __init__.py    thin socket shell   (restart Live to change)
                    └── handlers.py    LOM engine          (hot-reload, no restart)
```

**Threading contract:** accept-thread → client-thread → `schedule_message(0, task)`.
The Live API is only ever touched on Live's main thread; the client thread blocks
on a `Queue`. Violating this freezes the GUI.

**Hot reload:** `handlers.py` is re-`importlib.reload`ed by the `reload` op, so
engine changes need no Live restart. Only the socket shell does.

## Install

```bash
./scripts/install.sh          # symlinks into Ableton's User Library
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -e .
```

Then restart Live and enable **AbletonLOM** under
*Preferences → Link, Tempo & MIDI → Control Surface*.

Register with Claude Code:

```bash
claude mcp add ableton-lom -- /ABS/PATH/.venv/bin/python -m mcp_server.server
```

## Tools

29 tools. Generic first — the typed ones exist only where generic access
genuinely cannot work.

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
```

Roots: `live_set`, `live_app`, `app_view`.

## Two behaviours worth knowing

**Availability is per-instance, not per-type.** A member can exist on a type but
raise on a given object — MIDI tracks have no `input_meter_left`, only the main
track has a `crossfader`. `lom_describe` reports these in an `unavailable` map
rather than pretending the type list is uniformly gettable.

**Destructive calls are guarded.** Anything named `delete_*`, `remove_*` or
`clear_*`, plus `crop`, refuses to run unless `confirm=true`. Derived from a
pattern, not a hand-kept list — the list had drifted, guarding two names Live 12
does not have while missing eleven it does.

## Tests

```bash
./scripts/test.sh
```

| Suite | Needs Live | Covers |
|---|---|---|
| `tests/test_mcp.py` | no | all 29 MCP tools: op mapping, optional-param handling, safe destructive defaults, transport error translation |
| `tests/smoke.py` | yes | all 28 engine ops against real Ableton (~70s — it loads a real device and cycles scratch tracks) |

The integration suite creates a scratch MIDI track named `__lomtest` and deletes
it, so it works on an all-audio Set. It refuses to delete a track whose name
changed underneath it, and never touches existing tracks.

Only `test_mcp.py` is CI-able — a Remote Script cannot be exercised without its
host, so the engine tests are integration by necessity.

## Development

```bash
./scripts/lomcli.py ping                       # health
./scripts/lomcli.py describe "live_set"        # explore
./scripts/lomcli.py reload                     # after editing handlers.py
./scripts/lomcli.py types                      # full census
```

`lomcli.py` speaks the wire protocol directly, so the engine can be exercised
without the MCP layer in the way.

## Skill

`skills/ableton-lom/SKILL.md` teaches the path grammar — the one thing the tools
cannot infer. Symlink it into `~/.claude/skills/`.

Deliberately **no curated per-feature tool layer**. The plan called for ~30 sugar
verbs; at 29 generic tools that would mean 59, and tool overload degrades
selection. The skill buys the same ergonomics at zero tool cost.

## Two measured caveats

**Availability is per instance, not per type.** A MIDI track has no
`input_meter_left`; only the main track has a `crossfader`. `lom_describe`
reports these in an `unavailable` map instead of pretending the type list is
uniformly gettable.

**Undo grouping is partial.** `lom_transaction` collapses ops into one Cmd-Z —
except automatable parameters, which Live always gives their own undo step:

| Groups | Own step |
|---|---|
| track `name`, `color_index`, `signature_numerator` | `tempo`, mixer `volume`, `mute`, device parameters |

## Status

Phases 0–7 done. Plan: `~/.omc/plans/2026-08-01-ableton-mcp-superset.md`.
