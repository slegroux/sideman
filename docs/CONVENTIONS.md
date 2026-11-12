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

Undo grouping: on Live 12.4+ a transaction is one ⌘Z, automatable parameters
(tempo, mixer volume, mute, device parameters) included, because the engine
syncs Live's lazily recorded parameter changes into the step. On older Live
those parameters can need extra ⌘Z. The result's `undo` field says which
applies; pass it on rather than promising one undo.

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

## Inserting native devices

On Live 12.4+, `insert_device(name, index)` on a track or chain places a native
device exactly where you ask, which `browser_load` cannot. `index` is optional;
without it the device goes at the end. `name` is the browser name, exact and
case-sensitive. Recent releases renamed devices (measured on 12.4.6): Auto Pan
is now `Auto Pan-Tremolo`, and the old Auto Pan and Erosion survive as
`… Legacy`. So a name that worked last release can come back *not found*.

It inserts native devices only. A plug-in comes back *not found* (measured), and
per Ableton's release notes so does a Max for Live device: both still go
through `browser_load`. It takes a device name, not a preset, so a preset also
needs `browser_load`. Live's own rules are enforced with readable errors: only
audio effects on an audio track, no second instrument on a track, an index
within the chain. `duplicate_device(i)` copies device `i` to `i + 1` and returns
nothing. Duplicating the instrument in a rack chain is refused.

A Drum Rack can now be built without the browser:
1. `insert_chain()` on the rack adds a pad chain.
2. Set `in_note` to the chain's MIDI note.
3. `insert_device("Simpler")` on the chain.
4. `replace_sample(abs_path)` on the Simpler.

Two measured surprises:
- A new chain's `in_note` is 36, not the −1 ("All Notes") the release notes
  describe, and setting −1 is rejected. Give every chain its note, or they all
  stack on C1.
- `replace_sample` wants an absolute path to a real audio file. It reports
  either mistake plainly and leaves the loaded sample unchanged.

Device A/B compare is reachable too, where `can_compare_ab` is true (not on
racks). `save_preset_to_compare_ab_slot()` copies the current state into the
other slot (measured from A to B), and `is_using_compare_preset_b` switches
between the slots.

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

## Drum one-shots

`sample_analyze` and `sample_compare` measure a single drum hit in the terms
a synth is programmed in. They run on files, in the MCP server, and need the
`listen` extra. What the numbers mean:

- **Everything is relative to the hit itself.** Times run from the onset (the
  first 1 ms within 40 dB of the loudest 1 ms) and levels are against the hit's own
  peak or total. Two files at different levels, with different silence before
  the hit, or at different sample rates, measure the same.
- **Fall times start at the envelope peak**, not the onset: `attack_ms` is
  onset to peak, `fall_20_ms` is peak to the last moment within 20 dB. On a
  hit that beats, the level can dip under −20 dB and come back; the fall time
  is the last moment above.
- **The pitch is the fundamental, not the loudest partial.** `pitch_body_hz`
  is read on the body (40–150 ms, shorter on short hits). `pitch_start_hz` is
  read from zero crossings in the first milliseconds, and `pitch_start_at_ms`
  says how far in. A drop that is over in under 5 ms is a click (a stick
  transient); the same drop over tens of milliseconds is a kick's glide.
- **`pitched: false` is an answer, not a failure.** Hats, claps, shakers and
  cymbals have no fundamental; rims and bells have partials but no harmonic
  series. The profile then lists the partials, and a comparison reads the band
  balance in place of harmonics.
- **`None` means it could not be measured**, and `notes` says why: the file
  ends before the hit has fallen 40 dB, the body is shorter than four periods,
  the pitch cannot be followed back to the onset.
- **A take with several hits** reports `hits` and `onsets_ms`; `hit=` picks
  one, and the default is the first that is not cut off by the next hit or the
  end of the file. Rolls faster than about 100 ms are not split.

`readings` in `sample_compare` holds one line per difference past its
threshold, largest first. An empty list means the two hits match within the
thresholds, not that they are identical. Two drops whose starts were read at
different times are compared from the later one; two pitches an octave or a
twelfth apart on the same partials are one line, not a pitch difference.

**Limits** (known, not yet handled):

- A file's own noise floor is not measured: a hit recorded quietly (say
  −50 dBFS in 16 bits) can read a shorter `duration_ms` or more noise than
  the same hit recorded hot.
- A fundamental under about 36 Hz never gets harmonics: the 40–150 ms body
  window holds fewer than four of its periods.
- A ghost note under a hit's tail, and a roll faster than about 100 ms, are
  measured as one hit with no note; their fall times run long.
- A tone that fades in over a few milliseconds can read a pitch drop or rise
  at its start that is not there: the first zero crossings of a ramp are
  unevenly spaced.
- A partial beating within about 15 Hz of another can move the body's noise
  share by tens of dB between sample rates.
- At 22.05 kHz the bands above 11 kHz are empty, so a hat compared with a
  44.1 kHz file reads "highs low".

## Comparing a scene with a reference track

`reference_profile` and `mix_report` measure one scene of the Set against the
matching section of a reference track that comes with stems. They run on
files, in the MCP server, and need the `listen` extra. The report has three
layers:

- **The mix**: bands, width and loudness of the master capture, against the
  reference mix's section.
- **Elements against the yardstick**: each element's level as a ratio to the
  kick, ours against the reference's, and the difference. Ranked `gaps` come
  first; `width_gaps` (an element's width more than 0.1 apart) are listed
  apart from them.
- **Drum sounds**: for the kick, `toms`, `snare`, `hh`, `clap` and `rim` when
  both sides have them and sound in the paired bars, the `sample_compare`
  readings between a typical clean hit of ours and of the reference stem. A
  clean hit is complete, within 6 dB of the loudest complete hit, starts from
  quiet, and has room (to the next onset) close to the median; each side is
  cut with a beat of pre-roll, only hits that start inside the section count,
  and ours is the one at the same place as the reference's.

**The reference profile** is built once with `reference_profile(mix, stems,
tempo, t0, sections)`. `stems` maps a role to a file or a list of files that
are summed (`drums` = hh + snare + toms); `kick` is the yardstick. `t0` is
where bar 0 sits in the files, in seconds, and the sections are `{name,
start_bar, bars}`. It is written as JSON next to the mix, with each input's
path, mtime and size, and `mix_report` says which have changed since. Rebuild
it when a stem is re-exported; a profile from another version is refused.

**Where to tap.** Record each element at its last bus, Post Mixer, on a muted
audio track with monitoring Off, and the master on its own track. The master
tap sits after the master chain and the element taps before it, so the master
is never ratioed against an element.

**A ratio to the kick** cancels a gain that the stems and the mix share, so a
stem exported 3 dB hot still reads right; it does not cancel bleed. "Bass
−6 dB" means the bass is 6 dB under the kick in the same bars, ours and the
reference's. Where the kick is out on either side (silent, or 30 dB under the
loudest other element, as in a break) both sides use the loudest element
present on both as the yardstick instead, and the section says which. A kick
that is out on one side only is also listed as a difference ("kick in the
reference only, out in ours: 41 dB under"). With no shared element the table
is left out with a note.

**Present, trace, absent.** An element is present in a section when at least a
quarter of its bars (and one) are within 40 dB of the elements together, a
trace when a quarter are within 60 dB, otherwise absent. "Together" is our
element sources on one side and the profile's stems on the other, never the
master capture: its make-up gain would move the line. Absent is not reported as
−120 dB. A trace on both sides, or against an absent element, is listed and
never counted as a gap or as "in the reference only", because separation bleed
lives there; an element present on one side and a trace on the other is
compared and can be a gap. The first and last 10 ms of a section (not its
inner bar lines) are left out of an element's level, so a note that starts
just before the grid line does not read into the section before it.

**Pairing.** Sections pair by name: the manifest section "E05" with the
profile's "E05". A manifest section may carry `"reference": {"section":
"E05", "from_bar": 4}` to pair with another section or start four bars in; a
shorter section pairs with the first bars. The reference's bands, widths and
loudness are then those of the whole profile section. A section with no match,
or longer than its match, reports the mix layer only, with a note.

**The capture is the reference.** If the master capture matches the reference
mix, level aside (per-bar curve and bands within 0.3 dB, width within 0.01), the
section is flagged. The usual causes are the reference track left unmuted, so
it plays into the capture, and a solo that cuts the Set down to part of itself.
Guard against it before recording: no solos, and the reference track muted;
restore both afterwards.

**Limits.**

- Stems carry their own bleed and level. A ratio is only as good as the stem.
- The drum layer needs one isolated hit. A hat or tom roll has no clean hit, and
  the report says so instead of reading one.
- A capture shorter than the section is refused; compare whole bars.
- The numbers describe the difference. Which gaps are real is the user's ear's
  call: on one replica two stem reads were wrong, and the ear won.

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
