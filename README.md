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

Live 12.2.7 exposes **43 LOM types**; `Song` alone has **91 members**. Nobody
covers it by enumeration.

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

| Tool | Purpose |
|---|---|
| `lom_describe` | Discover properties, children, functions at a path. **Start here.** |
| `lom_get` / `lom_set` | Read/write any property (writes are undoable) |
| `lom_call` | Call any LOM function (undoable, guarded) |
| `lom_count` | Size of a collection |
| `lom_types` | Full type census — the coverage baseline |
| `lom_ping` | Connection + engine health |

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

**Destructive calls are guarded.** `delete_track`, `delete_scene`,
`remove_all_notes` and friends refuse to run unless `confirm=true`.

## Development

```bash
./scripts/lomcli.py ping                       # health
./scripts/lomcli.py describe "live_set"        # explore
./scripts/lomcli.py reload                     # after editing handlers.py
./scripts/lomcli.py types                      # full census
```

`lomcli.py` speaks the wire protocol directly, so the engine can be exercised
without the MCP layer in the way.

## Status

Phase 1 (vertical slice) — generic engine + `describe`/`get`/`set`/`call`/`count`/`types`.
Plan and phase roadmap: `~/.omc/plans/2026-08-01-ableton-mcp-superset.md`.
