---
name: ableton-lom
description: Control Ableton Live through the ableton-lom MCP server, which exposes the whole Live Object Model generically instead of one tool per feature. Use whenever the task involves Ableton Live - tracks, clips, MIDI notes, devices, mixing, arrangement, browser, tempo, automation - or when a lom_* / clip_* / browser_* / arrangement_* tool is available. Teaches the path grammar, which is the one thing the tools cannot infer.
---

# Ableton Live via the LOM

This server has **no tool per feature**. It exposes Ableton's Live Object Model
directly, so anything Live can do is reachable — but you must navigate to it.
That trade is why coverage is ~900 members instead of ~128.

## Paths

Space-separated, zero-indexed, unquoted.

```
live_set                                  the Song
live_set tempo                            a property
live_set tracks 0                         first track
live_set tracks 0 mixer_device volume     a DeviceParameter
live_set tracks 2 clip_slots 0 clip       a session clip
live_set tracks 0 arrangement_clips 0     an arrangement clip
live_set master_track mixer_device        the main mixer
live_app browser instruments              the browser tree
```

Roots: `live_set`, `live_app`, `app_view`.

## The loop

**Never guess member names.** Two tools exist precisely so you don't have to:

1. `lom_search(query="bass")` → `live_set tracks 3`
   `lom_search(type="DeviceParameter")` → every automatable parameter
2. `lom_describe(path)` → properties (with values), children (with counts),
   functions, and `unavailable`

Then `lom_get` / `lom_set` / `lom_call`.

Guessing costs a failed round trip; `lom_describe` costs one and is always right.

## Availability is per *instance*, not per type

`lom_describe` returns an `unavailable` map. This is normal Live behaviour, not
an error:

- `input_meter_left` — audio tracks only, absent on MIDI tracks
- `crossfader`, `cue_volume` — main track only
- `fold_state` — groupable tracks only

So a member listed for `Track` may still be unavailable on *this* track. Check
`unavailable` before reporting something as broken.

## Undo grouping is partial (measured on Live 12.2.7)

`lom_transaction` collapses ops into one Cmd-Z — **but not for automatable
parameters**, which Live always gives their own undo step:

| Groups | Own step |
|---|---|
| track `name`, `color_index`, `signature_numerator` | `tempo`, mixer `volume`, `mute`, device parameters |

Don't promise the user "one undo" when the batch touches tempo or volume.

## Recipes

**Tempo** — `lom_set("live_set", "tempo", 128)`

**Rename a track** — `lom_search(query="old")` then
`lom_set("live_set tracks N", "name", "new")`

**Arguments that are Live objects** — some APIs take an object, not a scalar. A
path sent as a bare string arrives as `str` and Live rejects the call with *"did
not match C++ signature"*. Wrap it in `{"__path__": "..."}` instead:
```
lom_call("live_set", "move_device",
         [{"__path__": "live_set tracks 5 devices 0"},
          {"__path__": "live_set return_tracks 0"}, 0])
lom_set("live_set view", "selected_track", {"__path__": "live_set tracks 3"})
```
Markers work nested inside lists/dicts. A plain string is never reinterpreted as
a path, so ordinary string arguments stay safe. `move_device` targets a **Track
or Chain**, never a rack device — an empty rack has no chain to move into.

**Create things** — creation is a *function call* on the parent:
```
lom_call("live_set", "create_midi_track", [-1])
lom_call("live_set tracks 0 clip_slots 0", "create_clip", [4.0])
lom_call("live_set tracks 0 clip_slots 0", "fire")
```

**MIDI notes** — use the typed tools, not `lom_set`. Notes cannot round-trip
through generic get/set.
```
clip_add_notes(path, [{"pitch": 60, "start_time": 0, "duration": 1, "velocity": 100}])
clip_get_notes(path)        # keep note_id
clip_modify_notes(path, [{"note_id": 3, "pitch": 62}])
```
pitch is a MIDI number (60 = C3); times are in **beats**.

**Load an instrument** —
`browser_list("instruments")` → `browser_load("instruments/Drift", track_index=0)`

**Arrangement** — `arrangement_create_clip("live_set tracks 0", start_time=0, length=8)`

**Watch for user edits** — `lom_observe` then `lom_poll_events(since=last_seq)`.
This catches changes made in the GUI, not just your own writes.

**Many reads** — `lom_get_batch([...])` is one round trip instead of N.

## Safety

- Destructive functions refuse to run without `confirm=true` — anything named
  `delete_*`, `remove_*` or `clear_*`, plus `crop`. **Ask the user first** —
  these destroy work, and a Live Set is not version-controlled.
- Prefer additive operations. Renaming beats deleting-and-recreating.
- The user's Set is live and often unsaved. Don't experiment in it; if you need
  a scratch track, create one and remove it, and say what you did.

## When it looks broken

- `lom_ping` — is Live running with `AbletonLOM` enabled as a Control Surface?
- Empty results with `truncated: true` — raise `max_results`/`max_depth`.
- "has no member X" — the member does not exist in this Live version. Several
  properties from older Live releases are gone in 12 (`follow_action_a`,
  `fade_in_start`, `freeze`, `create_group_track`). `lom_describe` is truth.
