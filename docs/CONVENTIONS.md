# Conventions

The rules every tool and every client method share. They are not per-feature
details; they are what the generic approach costs and what it buys. Most of
what follows was measured against a running Live, not read off a spec.

## Paths

Space-separated, zero-indexed, unquoted. The first segment is a root; every
segment after it is a member name or an index.

```
live_set tracks 0 mixer_device volume
live_set tracks 2 clip_slots 0 clip
live_app browser instruments
```

Roots: `live_set` (the Song), `live_app` (the Application), `app_view`.
Resolution is a plain attribute-and-index walk, so a path is exactly what it
reads as. There is no query language and no wildcard.

Some paths are aliases: `live_set view selected_track` is whichever track is
selected at the moment you ask. `canonical_path` resolves one to the stable
form, `live_set tracks 3`, which is what to store. A stored path is a
*position*, not an identity — move a device and `devices 1` is a different
device, so re-resolve rather than reuse.

## Objects as arguments

Some Live functions take a Live object, not a scalar. A path string reaches
them as a `str` and Live rejects the call with "did not match C++ signature".
Name the object with a marker instead:

```python
live.call("live_set", "move_device",
          [{"__path__": "live_set tracks 5 devices 0"},
           {"__path__": "live_set tracks 7"}, 0])
```

`{"__path__": "..."}` works anywhere in `args`, including nested, and in the
value of a `set`. A plain string is never reinterpreted as a path, so ordinary
string arguments stay safe.

## Describe before you guess

`lom_describe` reports the properties, children, functions and current values
at a path. A guess costs a failed round trip; `describe` costs one and is
right. This is the discovery step the whole design rests on — there is no
hand-written list of what Live has, because such a list goes stale every
release.

`lom_search` finds paths by name or type when you do not know where something
is. Both match as case-insensitive substrings, so `type="Track"` also matches
each track's `Track.View`.

## Availability is per instance

A member listed for a type can still be absent on a given object. The main
track has no `mute` and no `arm`; a MIDI track has no input meters, and no
output meters either until an instrument is on it. `describe` reports these
under `unavailable` rather than omitting them, which is the difference between
"Live does not do this" and "not on this object".

Check `unavailable` before observing or batching. An observer on a property an
object does not have is a failed round trip.

## One write, many writes, one undo

Three shapes, and the choice is about undo, not speed.

- `lom_get` / `lom_set` / `lom_call` — one property or one function. Every
  write is wrapped in a native undo step, so ⌘Z reverts it.
- `lom_get_batch` / `lom_set_batch` — many properties in one round trip. A
  batched read never fails as a whole: each result carries its own ok/error,
  so one unavailable property does not lose the other forty.
- `lom_transaction` — mixed `set` and `call` ops inside one undo step.

Undo grouping is partial, and this is Live's behaviour, not the server's.
Track name, colour and time signature group into one ⌘Z. Automatable
parameters — tempo, mixer volume, mute, device parameters — each form their
own step even inside an explicit one. Say which is which rather than promising
one undo.

A transaction is not a database transaction. Live has no intra-step rollback,
so if op 5 fails, ops 1–4 have already applied. `rollback_on_error` defaults to
false on purpose: it is a mutation on an error path, and an empty step would
revert whatever the user did beforehand.

## Long lists page

A property holding more than 64 items reports only
`{"__vector__": true, "count": N}` — which hides exactly the lists worth
reading. Pass `offset` and `limit` to page through one:

```python
live.get("live_set tracks 0 devices 0", "parameters", limit=25)
live.get("live_set tracks 0 devices 0", "parameters", offset=25, limit=25)
```

The window adds `items`, `returned` and `truncated` beside the count. 512 per
page is the maximum; a larger limit is clamped, not refused. The same
`offset`/`limit` page a long return value from `lom_call`, which is how you
read a plug-in's real parameter names.

## Observers

`lom_observe` records every change to a property, including changes the user
makes by hand in the GUI. That is the capability that makes a session a
conversation rather than a script. Collect with `lom_poll_events`.

The registry lives inside Live and is **shared across clients** — an MCP
session, a shell and a notebook all see one buffer. So:

- Start from the `latest_seq` that `lom_observe` returns, and poll with
  `since=`, advancing it to each reply's `next_since`. It is cheaper than
  re-reading, and you never claim someone else's events as your own. Use
  `next_since`, not `latest_seq`: when a page is `truncated`, `latest_seq`
  jumps past the events still waiting.
- Unobserve what you created. `lom_unobserve_all` is a blunt instrument: it
  removes observers you did not create. Its `leaked` count is listeners still
  firing inside Live and should always be 0.
- An observer reports a change, not a *source* of change. Your own writes fire
  it too, so compare against what you just did or you will answer yourself.

Meters are a firehose. `output_meter_level` on eight tracks produces a couple
of hundred events in four seconds of playback — Live coalesces them rather than
firing per audio frame, but the buffer holds 2000 and drops the oldest first.
Watch `dropped_events`: nonzero while you are looking for *gestures* means a
fader move was thrown away. Take the meters off before the conversation
continues.

## Destructive calls

Any function named `delete_*`, `remove_*` or `clear_*`, plus `crop`, refuses to
run unless you pass `confirm=true`. The refusal is a `PermissionError` raised
inside Live, so it is distinguishable from a genuine failure. Ask the user
before setting it — the Set is not in version control, and nothing the server
does is saved.

## What reads back is not always what you wrote

Live stores parameter values as 32-bit floats. Write `0.78` to a mixer volume
and it reads back as `0.7799999713897705`. Note times come back at Live's own
resolution, `0.2700000520` rather than `0.27`. Round before asserting, and
compare with a tolerance.

Device parameters are normalised, but not uniformly: Drift's parameters are all
0–1, switches included, and `Noise Gain` starts at 0, which is silence and not
a neutral middle. Read `min` and `max` before writing anything.

## Automation

An envelope belongs to a *(clip, parameter)* pair — two paths, not one — so the
same filter can sweep one way in the loop clip and another way in the build.

Live evaluates a step edge as the end of what came before. Sample at beat 0 and
you get the parameter's own pre-envelope value; sample at beat 4, the edge
between the first step and the second, and you get the *first* step's value.
Sample inside a step, not on its boundary, or correct automation reads as
broken.

## Loading from the browser

`browser_load` places a device at the **end** of the track's device chain, and
nothing in the Live Object Model chooses where it lands —
`view selected_device` is read-only. Load in the order you want, or fix the
order afterwards with `move_device`.

A sample is different: it lands in the **highlighted clip slot** of the
**selected track**, and only while Live is showing the Session view. From the
Arrangement view the browser drops it nowhere and reports success anyway.
`track_index` picks the track and decides nothing else, so set
`highlighted_clip_slot` — which, unlike `selected_device`, has a setter — and
`show_view` first.

## The arrangement

An arrangement clip cannot be lengthened in place. `end_marker` and `loop_end`
move; `end_time`, which is where the clip actually stops on the timeline, does
not move with them and has no setter. Resizing is a drag in Live's GUI, so
tiling with `arrangement_duplicate_clip` is not a workaround — it is the
operation that exists.

`song_length` measures the timeline, not the music: Live pads the arrangement
eight bars past the last clip. Measure the music from `arrangement_list_clips`.
Bar positions assume one time signature, because Live's API exposes no list of
meter changes.

## Audio clips

Everything about an audio clip is plain `get`/`set` except the warp map. A
`WarpMarker` is a C++ object with no JSON form, so reading `warp_markers` as a
property returns repr strings; read the markers individually by path instead,
and write them with `clip_set_warp_markers`, which builds them inside Live.

`sample_time` is in **seconds** from the sample start, not frames. Scaling by
the sample rate produces a map that looks plausible and plays wrong. Every clip
carries one warp marker Live refuses to delete, so `remove_failed: 1` is
expected, not a failure.

## Racks and plug-ins

Two walls, both real, and both worth stating to the user rather than working
around.

`add_macro` reveals macros in pairs — 8 → 10 → 12 — so one call adds two. And a
macro can be added but not *mapped*: nothing in the Live Object Model connects
Macro 1 to a filter's cutoff, so `macros_mapped` stays all `false` until the
user drags. A VST or AU exposes only `Device On` until the user presses
**Configure** in Live, another GUI-only step, though `get_parameter_names`
reports the full list Live already knows.

Claude can read the parameter names and say exactly what to expose and what to
map. It cannot press the button or make the drag, and reporting either as done
is a lie the user finds in ten seconds.
