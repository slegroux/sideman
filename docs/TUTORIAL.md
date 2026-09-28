# Tutorial: build a 4-bar house loop from an empty Set

Every command here was run against Live 12.2.7 and the output is real. By the
end you have three tracks, three instruments, 76 MIDI notes, a filter sweep,
and a running loop — none of it touched by hand.

The point is not the loop. It is that no tool in this server knows what a
"chord" or a "hi-hat" is. You navigate the Live Object Model and it does what
you say.

## Who types this?

Not you. `lom_call(...)` is what **the model** sends over MCP; you say
*"make me a four-bar house loop"* and Claude issues these calls on its own —
the installed skill teaches it the path grammar. Read the calls here the way
you would read a wire capture: to understand what is possible, what actually
happened, and what to ask for.

Where this pays off in conversation:

- *"What can you see in my Set?"* → the model runs `lom_describe` / `lom_search`.
- *"Nothing works"* → ask it to run `lom_ping`; that is the real health check.
- If it asks to confirm a delete, that is the destructive guard doing its job.

The rest of this page is the same session from the model's side.

## Before you start

Live running, `AbletonLOM` selected as a Control Surface, and an **empty Set**
(File → New). Check the connection first:

```
lom_ping()
→ {"pong": true, "port": 9878, "handlers": true, "handler_error": null}
```

`handlers: false` means the engine failed to import — read `handler_error`.

> Everything below writes to the open Set. Work in a scratch Set, not on
> anything you care about. Every step is undoable, but "undoable" is not
> "saved".

## 1. Orient yourself

Never guess member names. Ask:

```
lom_describe("live_set", include_values=false)
```

On an empty Set that returns 52 properties, 186 functions, and:

```json
"children": { "tracks": 4, "scenes": 8, "return_tracks": 2, "cue_points": 0 }
```

That is the whole vocabulary for the Song. `tracks` is navigable, so
`live_set tracks 0` is a path. Zero-indexed, space-separated, no quotes.

## 2. Make a track

A default Set gives you 2 MIDI and 2 audio tracks. We need a third MIDI track
for drums, inserted at index 2:

```
lom_call("live_set", "create_midi_track", [2])
→ {"result": {"__lom__": "Track.Track", "name": "3-MIDI"}}
```

There is no `create_track` tool. `create_midi_track` is a function Live already
has; `lom_call` invokes it. Anything in the `functions` list from step 1 works
the same way.

## 3. Name and colour them in one undo step

Six writes, one Cmd-Z:

```
lom_transaction(ops=[
  {"op":"set","path":"live_set tracks 0","property":"name","value":"Chords"},
  {"op":"set","path":"live_set tracks 0","property":"color_index","value":26},
  {"op":"set","path":"live_set tracks 1","property":"name","value":"Bass"},
  {"op":"set","path":"live_set tracks 1","property":"color_index","value":14},
  {"op":"set","path":"live_set tracks 2","property":"name","value":"Drums"},
  {"op":"set","path":"live_set tracks 2","property":"color_index","value":5}
])
→ {"count": 6, "applied": 6, "failed": false, ...}
```

**Read the `undo` field in the result.** Names and colours group into one undo
step. Automatable parameters — tempo, mixer volume, mute, device parameters —
do not: Live gives each its own step no matter what you wrap them in. This was
measured, not assumed. A transaction is "one Cmd-Z for the groupable ops, plus
one per automatable parameter touched".

It is also not a database transaction. If op 5 fails, ops 1–4 have already
applied. `stop_on_error` defaults true; `rollback_on_error` defaults **false**
on purpose, because undoing on an error path can revert work you did earlier.

## 4. Load instruments

Browse by name, then load onto a specific track:

```
browser_list("instruments")     → Analog, Collision, Drift, Operator, Meld, ...
browser_load("instruments/Drift", track_index=0)
→ {"loaded": "instruments/Drift", "track": "Chords"}
```

Presets work the same way — the drum kit is a real `.adg` in the library:

```
browser_load("instruments/Operator", track_index=1)
browser_load("drums/909 Core Kit.adg", track_index=2)
→ {"loaded": "drums/909 Core Kit.adg", "track": "Drums"}
```

`browser_load` returns the track it landed on. Check it: without
`track_index` it loads onto whatever is selected.

## 5. Create clips

16 beats = 4 bars:

```
lom_call("live_set tracks 0 clip_slots 0", "create_clip", [16])
lom_call("live_set tracks 1 clip_slots 0", "create_clip", [16])
lom_call("live_set tracks 2 clip_slots 0", "create_clip", [16])
```

The clip is then at `live_set tracks 0 clip_slots 0 clip`. A `clip_slot` is not
a clip — the slot always exists, the clip only exists once created.

## 6. Write notes

Notes are one of the few places with a typed tool, because Live hands back
`MidiNote` objects but wants `MidiNoteSpecification` on write, which generic
get/set cannot express.

Chords — Am7, Fmaj7, Cmaj7, G, one per bar. `start_time` and `duration` are in
beats, pitch is a MIDI note number (60 = C3):

```
clip_add_notes("live_set tracks 0 clip_slots 0 clip", notes=[
  {"pitch":57,"start_time":0,"duration":3.5,"velocity":85},
  {"pitch":60,"start_time":0,"duration":3.5,"velocity":85},
  {"pitch":64,"start_time":0,"duration":3.5,"velocity":85},
  {"pitch":67,"start_time":0,"duration":3.5,"velocity":85},
  ...                                    # 53/57/60/64 at beat 4
])                                       # 60/64/67/71 at beat 8
→ {"added": 16}                          # 55/59/62/67 at beat 12
```

Bass — root on the downbeat, then offbeat eighths, roots A/F/C/G:

```
clip_add_notes("live_set tracks 1 clip_slots 0 clip", notes=[
  {"pitch":33,"start_time":0,   "duration":0.45,"velocity":105},
  {"pitch":33,"start_time":0.5, "duration":0.45,"velocity":85},
  {"pitch":33,"start_time":1.5, "duration":0.45,"velocity":85},
  ...
])
→ {"added": 20}
```

Drums — a 909 rack maps 36 = kick, 39 = clap, 42 = closed hat. Kick on every
beat, clap on the backbeat, hats offbeat with alternating velocity so it
breathes:

```
clip_add_notes("live_set tracks 2 clip_slots 0 clip", notes=[
  {"pitch":36,"start_time":0, "duration":0.25,"velocity":112},   # x16
  {"pitch":39,"start_time":1, "duration":0.25,"velocity":96},    # x8
  {"pitch":42,"start_time":0.5,"duration":0.2, "velocity":72},   # x16
  ...
])
→ {"added": 40}
```

Read them back any time with `clip_get_notes`, which returns `note_id` for each
— keep those, `clip_modify_notes` needs them to edit in place.

## 7. Automate a filter

First find the parameter. You do not need to know Drift's layout:

```
lom_search(type="DeviceParameter", root="live_set tracks 0", max_results=200)
→ 73 parameters, including:
   live_set tracks 0 devices 0 parameters 1   LP Freq
```

Searching `query="Cutoff"` returns nothing — Drift calls it `LP Freq`. That is
exactly why search exists: you look, rather than guessing a name and getting an
`AttributeError`.

Device parameters are normalised. Check before writing:

```
lom_get("live_set tracks 0 devices 0 parameters 1", "min")  → 0.0
lom_get("live_set tracks 0 devices 0 parameters 1", "max")  → 1.0
```

Now a rising sweep, one flat step per bar:

```
clip_envelope_insert_step(
  path="live_set tracks 0 clip_slots 0 clip",
  parameter="live_set tracks 0 devices 0 parameters 1",
  time=0, length=4, value=0.35)
# then (4, 4, 0.5), (8, 4, 0.72), (12, 4, 0.95)
```

Two paths, not one: the envelope belongs to a *(clip, parameter)* pair. Read it
back:

```
clip_envelope_get(path=..., parameter=..., samples=9)
→ exists: true | param: LP Freq | range: 0.0 - 1.0
   beat  0.0 -> 1.000      <- see below
   beat  2.0 -> 0.350
   beat  6.0 -> 0.500
   beat 10.0 -> 0.720
   beat 14.0 -> 0.950
```

Sampling at exactly beat 0 returns 1.0, the pre-envelope value, not 0.35.
Live evaluates the boundary that way. Sample inside a step, not on its edge.

## 8. Play it

```
lom_call("live_set scenes 0", "fire")
lom_call("live_set", "start_playing")

lom_get("live_set", "is_playing")         → true
lom_get("live_set", "current_song_time")  → 10.71, then 15.22, then 19.75
```

Firing a scene launches the clips; `start_playing` guarantees the transport is
rolling.

## 9. Watch what the user does

This is the part no other Ableton MCP server does. `lom_observe` reports
changes **including ones made by hand in Live's GUI**:

```
lom_observe("live_set", "tempo")
lom_observe("live_set tracks 0 mixer_device volume", "value")
→ {"observing": true, "active_listeners": 2}
```

Then drag the tempo control in Live and poll:

```
lom_poll_events()
→ {"count": 8, "latest_seq": 8, "dropped_events": 0, "events": [
     {"seq":1, "path":"live_set", "property":"tempo", "value":123.5},
     {"seq":2, "path":"live_set", "property":"tempo", "value":122.5},
     {"seq":3, "path":"live_set", "property":"tempo", "value":123.5},
     ...
     {"seq":8, "path":"live_set tracks 0 mixer_device volume","value":0.72}
   ]}
```

Those first events are a human dragging the tempo, captured while this
tutorial was being written. MCP has no server-to-client push, so this is a
pull with a ring buffer behind it: Live accumulates events, you collect them.

- Pass `since=<previous latest_seq>` to get only new events.
- Watch `dropped_events`. The buffer holds 2000 and drops oldest first, so
  nonzero means you polled too slowly and lost changes.
- Not every property is observable. If there is no `add_<property>_listener`,
  `lom_observe` says so — `lom_describe` lists what an object really has.

Always clean up. A listener Live still holds is a listener you cannot reach:

```
lom_unobserve_all()
→ {"attempted": 2, "outcomes": {"removed": 2}, "failures": [], "leaked": 0}
```

`leaked` should always be 0.

## What you learned

| Idea | Why it matters |
|---|---|
| `lom_describe` before anything | the member list is the API; guessing produces `AttributeError` |
| `lom_search` to locate | `LP Freq`, not `Cutoff` — you cannot guess vendor naming |
| Paths are values | `live_set tracks 0 devices 0 parameters 1` can be stored, passed, reused |
| Typed tools are the exception | notes, envelopes, browser, warp markers — only where generic access genuinely cannot work |
| Undo grouping is partial | check the `undo` field rather than assuming one Cmd-Z |
| Observers are a pull | poll, and watch `dropped_events` |

## Where to go next

- `lom_canonical_path("live_set view selected_track")` → `live_set tracks 3`.
  Resolve whatever the user has selected into a stable path.
- `lom_get_batch` — read 40 properties in one round trip instead of 40.
- `arrangement_create_clip` / `arrangement_duplicate_clip` — commit the session
  loop into the Arrangement.
- `clip_set_warp_markers` — retime an audio clip. `beat_time` is beats,
  `sample_time` is **seconds**, not frames.
- `lom_types()` — the full census of every type and member Live exposes.

## If something breaks

**`Nothing listening on 127.0.0.1:9878`** — Live is not running, or the Control
Surface slot was cleared. Switching Sets can clear it.

**`no member 'X' at 'live_set tracks 0'`** — usually correct. Run
`lom_describe` on that path; several properties from older Live versions are
gone in 12.

**A property reads fine on one object and raises on another** — availability is
per *instance*, not per type. A MIDI track has no `input_meter_left`; only the
main track has a `crossfader`. `lom_describe` reports these in `unavailable`.

**Destructive call refused** — anything named `delete_*`, `remove_*`, `clear_*`
or `crop` needs `confirm=true`. That is deliberate. Ask the user first.
