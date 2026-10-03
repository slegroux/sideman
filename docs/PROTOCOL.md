# Wire protocol

What the MCP server, the Python client (`sideman.client`) and `scripts/lomcli.py`
all speak to the Remote Script.
Read this before adding an op or writing another client.

## Transport

Newline-delimited JSON over TCP on `127.0.0.1:9878`. One request per line, one
response per line, in order. The connection may be reused.

Port 9878 is deliberate: `ahujasid`, `uisato` and `jpoindexter` all use 9877, so
this server coexists with them rather than fighting for the port.

**Request**

```json
{"id": "anything", "op": "get", "params": {"path": "live_set", "property": "tempo"}}
```

**Response — success**

```json
{"id": "anything", "ok": true, "result": {"path": "live_set", "property": "tempo", "value": 120.0}}
```

**Response — failure**

```json
{"id": "anything", "ok": false,
 "error": {"type": "AttributeError", "message": "...", "traceback": "..."}}
```

`id` is echoed untouched. `error.type` is the Python exception class name, which
is load-bearing: the MCP layer surfaces it, and callers distinguish
`PermissionError` (write guard) from `AttributeError` (member does not exist).

## Threading contract

This is the part that breaks Ableton if you get it wrong.

```
accept thread (daemon)
  └── client thread (daemon, one per connection)
        └── schedule_message(0, task)  ──►  LIVE'S MAIN THREAD
                                             │  the ONLY place the Live API
                                             │  may be touched
              ◄── queue.Queue(1) ────────────┘
```

The client thread blocks on a `Queue` with a 15s timeout while the main thread
runs the handler. **Never touch a Live object from a socket thread** — it
freezes or crashes the GUI.

Handlers therefore must not block. Anything slow (a large graph walk) needs a
node budget, which is why `search` and `canonical_path` take `max_nodes`.

## Ops served by the shell

Handled without reaching the engine, so they work even when it fails to load:

| op | purpose |
|---|---|
| `ping` | health; reports whether handlers loaded and any load error |
| `reload` | re-`importlib.reload` the engine — **no Live restart needed** |

`ping` is answered on the socket thread; `reload` is marshalled onto the main
thread like an engine op, because tearing observers down removes listeners —
a Live API call.

`reload` tears observers down first. `importlib.reload` re-executes the module,
so anything registered in module globals would be orphaned while Live still held
the callback. State that must survive reload lives on the ControlSurface
instance instead (the observer registry and the undo-depth counter both do).

## Ops served by the engine

See `docs/TOOLS.md` for parameters — every MCP tool maps to one of these.

`describe` `get` `set` `call` `count` `types` `search` `canonical_path`
`get_batch` `set_batch` `transaction` `notes_get` `notes_add` `notes_modify`
`notes_remove` `browser_list` `browser_load` `envelope_get`
`envelope_insert_step` `envelope_clear` `arrangement_list`
`arrangement_create_clip` `arrangement_duplicate_clip` `observe_add`
`observe_remove` `observe_list` `observe_clear` `observe_poll`
`warp_markers_set`

## Paths

Space-separated, zero-indexed, unquoted. Roots: `live_set` (Song), `live_app`
(Application), `app_view`.

```
live_set tracks 0 mixer_device volume
live_set tracks 2 clip_slots 0 clip
live_app browser instruments
```

Resolution is a plain `getattr` / index walk (`handlers.resolve`), deliberately
vendored rather than using `_MxDCore`'s resolver, so a Live update cannot break
path handling.

## Adding an op

1. Write `op_yourthing(surface, params)` in `handlers.py`. It runs on the main
   thread; return anything JSON-serialisable.
2. Wrap mutations in `with _undo(surface):` — reentrant, so nesting inside a
   `transaction` still produces one undo step.
3. Register it in `OPS`.
4. `./scripts/lomcli.py reload` — no Live restart.
5. Add an MCP tool in `mcp_server/server.py`, then
   `./scripts/gen_docs.py` and add it to `EXPECTED_OP` in `tests/test_mcp.py`
   (that test fails if the registry and the test list disagree, so a new tool
   cannot ship untested).

## Gotchas the hard way

**Availability is per instance.** A member on the type can still raise on a given
object — `crossfader` is main-track only, `input_meter_left` is absent on MIDI
tracks. `describe` reports these under `unavailable` rather than omitting them.

**`id()` is not a valid identity for Live objects.** Live returns a fresh Python
wrapper per `getattr`; once one is garbage-collected its address is reused, so
`id()` makes a graph walk mistake new objects for visited ones and truncate
silently. Use `_live_ptr` (see `_identity`) and hold references during a walk.

**The M4L registry is a subset of the real API.** For `Clip`,
`get_available_properties_for_type` lists 69 members while `dir()` finds 190 —
the missing ones include the entire extended-notes API and every listener.
`_member_names` returns the union.

**Undo grouping is partial.** `begin_undo_step`/`end_undo_step` groups track
name, colour and time signature, but automatable parameters (tempo, mixer
volume, mute, device parameters) always get their own step regardless.

**Some things Live's API simply cannot do.** The 12.2.7 census has no
`move_track`, so tracks cannot be reordered after creation — create them in their
final order (`duplicate_track` exists but does not reposition). There is no save,
export/render, freeze, consolidate or flatten either; the only `export` in the
object model is `LooperDevice.export_to_clip_slot`. A scripted session has to end
with a human saving and bouncing, so say so rather than assuming the set persists.
