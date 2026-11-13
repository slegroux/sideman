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

`sample_compare` also returns `distance`: one number for how far apart the
spectra of two hits are, the score an optimiser can minimise when matching a
synth patch to a sample (`distance_terms` splits it into `convergence` and
`log_magnitude`). It is the multi-resolution STFT loss of `auraloss`
(`MultiResolutionSTFTLoss`, default settings) rewritten in numpy: at FFT sizes
512, 1024 and 2048 (windows of 240, 600 and 1200 samples), spectral
convergence (how much of the loud spectrum differs) plus the mean
log-magnitude difference (how much of the quieter spectrum differs: tail,
noise, highs), averaged over the sizes. It has not been checked against
listening, only against `sample_analyze`'s readings and role-sorted samples
(below). The default differs from `auraloss` in four ways, each measured:

- **Mel bands** (`scale="mel"`): the magnitudes are first summed into Slaney
  mel bands, `auraloss`'s own `scale="mel"`, with 128, 64 and 32 bands at the
  three sizes (its one band count for all sizes gives empty filters and NaN at
  512 with 128). A noise burst's bin magnitudes change with the noise
  realisation and with a one-sample shift; bands average that out. `scale="linear"`
  is the plain bins.
- **Symmetric**: `auraloss` divides the convergence by the target's norm
  alone; here it is the mean of both divisions.
- **A dynamic-range floor** (`floor_db`, 60): the log term clamps magnitudes
  at 60 dB under the loudest magnitude of either hit at that resolution, not
  at an absolute 1e-4 (about 120 dB down), and averages over the frames where
  either hit is above that floor, so it does not depend on how much silence the
  files carry. With the absolute floor, the log term on the Bodzin pads swung
  from 0.4 to 3.6 with how much inaudible tail one hit had against the other's
  silence: a kick read 4.8 from a snare whose 40 dB fall is 508 ms. A tail or
  noise bed that rises above the floor does count: a noise bed 50 dB under a
  hat's peak adds 1.4 (it is 41 dB under the hat's loudest millisecond, 19 dB
  above the floor), and one 45 dB under a tom adds 0.06 (a tone's loudest bin
  stands about 25 dB over its mean level, so the bed is under the floor).
  A sample with a noise bed against a clean synth render therefore reads the
  bed: a hat 60 dB over its bed reads 0.5 to 0.6 (0.16 to 0.18 on linear
  bins). Pass a `floor_db` a few dB under the sample's own noise floor
  (`floor_db=50` there leaves 0.07 and keeps a decay change at 0.69).
  `floor_db=None` is the `auraloss` definition (symmetry aside).
- **Equalised first**: DC removed; the onsets aligned (the first 1 ms within
  40 dB of the loudest, or 12 dB over a noise bed that precedes the hit: a
  clean hit read late by whole or fractional samples of silence differs by a
  sample or two, 0.002 for a tone started 0.4 sample off the grid); each hit
  placed 25 ms of silence after the start so the STFT's reflect padding never
  mirrors the attack (with 2 ms of lead a one-sample shift of a hat cost 0.15 with
  linear bins and 0.013 with mel; with 25 ms, 0.012 and 0.006, against 0.23 for
  the smallest real difference below); both
  scaled to the same loudest 1 ms (its RMS, not the sample peak, which one click
  would set; `match_level=False` in `oneshot.spectral_distance` keeps the raw
  level); both cut to the longer of the two (at most 2 s), the shorter counting
  as silent after its end, so a tail on one side only costs in proportion to
  how far it rises above the floor (one that has fallen 60 dB costs nothing).
  Hits at another sample rate are compared at the lower rate. Channels are
  compared pairwise, so swapping L and R of a stereo hit reads as a difference;
  a mono hit against stereo is compared with the stereo's mono mix, which can
  comb for a wide hit.

Measured values, on synthetic drums (a 200 Hz tom with two harmonics, a 50 Hz
kick with a pitch glide, noise hats and snares), default settings: 0 for the
same hit, also read late by whole samples of silence (under 1e-9); a decay 25 %
longer (tau 50 against 40 ms) 0.23, 2.5 times longer 0.91; one semitone on the
200 Hz tom 0.30, seven 1.05, twelve 1.39; a kick against a hat 3.9, a hat
against a snare 3.7. **Low pitches have a shelf**: on the 50 Hz kick 0.05, 0.1,
0.2, 0.3 and 0.5 semitones read 0.06, 0.11, 0.18, 0.22 and 0.25, then 0.75 to
1.5 semitones read 0.25, 0.24, 0.24, 0.25 (a shallow local minimum near 1 semitone),
2 semitones 0.27 and twelve 0.79, because partials that do not overlap are as
far apart at 0.75 semitones as at 1.5. So the distance does not guide pitch
below about 2 semitones on a low drum; the search must start from the
measured pitch (`pitch_body_hz`). There is no ceiling: a hat with a 20 ms
decay against one of 400 ms reads 5.5, and the nearest of the 108 samples below to
a crash cymbal is 3.5 away.
Noise: a noise-like hit is not the same as another take of itself: the same hat
with another noise seed reads 0.6, the same snare 0.45, about two thirds of a
hat with a 50 % longer decay (0.85). Take 0.5 as the resolution on noisy
hits: a threshold under it chases noise. Another take of a hat reads 0.6 from it
and a snare 3.7. A constant noise floor
80 dB under the peak reads 0.0004, 30 dB under 1.2; a tail replaced by
silence from where it is 70 dB down reads 0.0003, from 20 dB down 0.19.

Against role-sorted samples (108 one-shots in `_PRODUCTION/Drums`, folders as
labels; the nearest other sample in the same folder, leave-one-out): 0.77
(standard error 0.04; 0.21 by chance from the folder sizes) with mel, and
0.77 with linear bins; per folder, mel: kicks 1.00 (33 files), claps and
snares 0.84 (31), closed hats 0.50 (12), open hats 0.64 (11), percussion 0.33
(9), toms 0.78 (9), rides 0 (2); mean distance within a folder 1.5, across
folders 3.0. The misses are largely real likeness (closed and open hats of
similar length, rims filed in two folders, a clave next to a clap), but not
all: the ranking is by spectral overlap, not by pitch proximity or by role.
It says how far apart two spectra are and not which property differs: for
that read `readings`. The windows are in samples, so a score is comparable
between files at one sample rate, not between 44.1 and 96 kHz.

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
  clean hit is complete (it falls 40 dB before the next one), within 6 dB of the
  loudest such hit, starts from quiet if any does (20 ms before it 20 dB down),
  has room to the next onset of at least 90 % of the median, and, if any does,
  a peak within 3 dB of the median; the earliest of those. Each side is cut with
  a beat of pre-roll and post-roll (ending 50 ms short of the beat line, where
  early hits start; ours 30 ms longer, so the same audio read a little early
  still has its room to fall), only hits that start inside the section (its
  window starting 20 ms early, as below) count, the reference's is picked first,
  and ours is the one within 30 ms of the same place when there is one. When our
  own hit-finding offers none there, a hit that rises out of what comes before
  it within 30 ms of that place is measured, with a note; otherwise our own
  typical hit, with a note that the hits are not at the same place (where each
  is) and that the readings compare different hits. The two hits are then
  measured again on cuts made from the hits themselves, of the same length, so
  the same audio gives the same readings however it is offset against the grid.

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
that is out on one side only is also listed as a difference ("In the
reference only: kick (kick out in ours: 35 dB under the loudest other role,
+6.0 dB vs bass)"). With no shared element the table is left out with a note.

**Present, trace, absent.** An element is present in a section when at least a
quarter of its bars (and one) are within 40 dB of the elements together, a
trace when a quarter are within 60 dB, otherwise absent. "Together" is our
element sources on one side and the profile's stems on the other, never the
master capture: its make-up gain would move the line. Absent is not reported as
−120 dB. A trace on both sides, or against an absent element, is listed and
never counted as a gap or as "in the reference only", because separation bleed
lives there; an element present on one side and a trace on the other is
compared and can be a gap. Presence is judged on the core of each bar, from
20 ms after its grid line to 20 ms before the next, on both sides: whatever
sounds within 20 ms of a line (a note ending on it, a hit starting just before
it) never makes an element present in a bar or a section. Levels are read over
the whole bar, except that a bar's window starts at the onset of a hit that
rises within 20 ms before its line (real drums start a few ms early, and a
capture is never sample-aligned with the reference): that hit belongs to its
own bar and section, while a note or a tail that ends on the line stays in the
bar before.

**Timing tolerance.** A hit's earliness against the grid plus the capture's
read offset (how far `offset_s` sits from where the audio really is, either
way) must stay within 20 ms. Flare-like drums start 3-9 ms early, which leaves
11-17 ms for the offset. Past 20 ms a hit at a section line counts in the
section before it (it can make an element present there, "in ours only"), and
a drum hit's place differs by more than the 30 ms the drum layer allows when
the earliness and offset together pass 30 ms ("not at the same place").

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
- A hit that cannot fall 40 dB before the next one is never compared: a long
  kick at 124 BPM, or closed hats on 16ths, reads "no clean hit".
- The drum layer compares both hits over the shorter of the two cuts. When one
  side's hit is the last in its slice, a longer tail on the other is not read.
- With grace notes or a swell a few ms before a hit, the onset found can be any
  of the rises, and the two sides can start up to about 16 ms apart.
- Claps and other layered hits can split differently on the two sides. The
  note "measured at the reference's place" then appears, without readings,
  even on identical audio.
- A section line closer than 40 ms to the start of a file gets no early start.
- The numbers describe the difference. Which gaps are real is the user's ear's
  call: on one replica two stem reads were wrong, and the ear won.

### Drum roles

`sideman.drumroles` (needs the `listen` extra; no MCP tool yet) lists the sounds
inside one drum stem. `hit_groups` finds the hits, puts them on the 16th grid
and groups the ones that sound alike. `name_roles` then gives each group a name
and `mark_bleed` marks the groups that are another stem's sound leaking in.

**The names are rules over a group's stem, its pattern and the readings of its
clean hit. They are not recognition.** A group no rule fits keeps a neutral
name (`hh 2`: the stem and the group's number) with the reason. A name says
"this group passed this rule", never "this is a hi-hat". Listen to the clean
hit before trusting one.

A group is **named only when it is eligible**: at least 4 measured hits,
`stability` measured (it is not on a stem with no more measured hits than the
sample of 32, nor for a tentative group) and at least 0.75, and `separation`
over the grouping threshold (1.5). `separation` is measured against the nearest group
of at least 4 measured hits (`hit_groups` also keeps `separation_any` and
`nearest_any`, against any group): a one-to-three hit satellite beside a big group
says nothing about whether two sounds are apart (on Flare a 3-hit group at 1.21
had kept the 57-hit toms group from a name). A group with no such neighbour has
`separation` None, no rival, and passes that test, as a lone group does.
Otherwise it keeps the neutral name and the reason says which test it failed.
The rule is the second review's: on the Flare hats, groups at stability
0.79-0.91 were the ones the sample did not move, 0.13-0.69 the ones it did.

Pattern: the hits counted are the measured ones and the ones assigned to the
group under another hit's tail (`assigned` says how many). A 16th is *in use*
when the group plays there in at least 25 % of the bars it plays in **that it could
be seen in**. A bar does not count for a 16th when the line fell inside another
hit's tail with no hit found at it (`hidden_cells`: a closed hat under an open
hat's tail cannot be seen playing there), or held an on-grid hit that was not
measured (`unmeasured_cells`: it may have been this group's), and the group does
not play there. Without that, a hat that plays every "a" but is hidden in three
bars of four reads 25 %. A 16th observable in fewer than 4 bars is *unknown*: it
counts neither for nor against, and a name given with unknown 16ths says how many
in its `note`. The per-16th bar counts and shares (over all bars and over the
observable ones) are in each role's `steps`. "n of 6 in use" is read over the
known 16ths: the share asked for (6 of 8, 3 of 4) of the known ones, rounded up,
with at least half of them known.

| Stem | Name | Rule | Where the number comes from |
|------|------|------|-----------------------------|
| hh | `hh offbeat` | at least 90 % of the hits on steps 2, 6, 10, 14 and at least 3 of those 4 in use (of the known ones) | 90 %: chosen, a tenth of the hits may be strays or assigned to the wrong 16th; 3 of 4: one offbeat may be taken by the other hat (Flare hh group 0 plays its four offbeats in 56 % to 89 % of the stem's bars) |
| hh | `hh 16th` | at least 6 of the 8 odd 16ths in use | chosen: a 16th-note hat plays the "e" and "a" of each beat, and 6 of 8 leaves two gaps (4 of 8 let a hat that plays only the "e" of each beat, Flare hh group 2, take the name); an offbeat group has none of them |
| hh | `hh open` | neither pattern fits and the clean hit falls 20 dB in 140 ms or more | `fall_20_ms` of the 22 closed and open hats of `_PRODUCTION/Drums` that could be measured (below) |
| snare | `rim` | the clean hit falls 20 dB in under 30 ms | the library's 3 rims: 5, 19 and 28 ms; its other 30 claps and snares: 35 to 167 ms |
| snare | `snare` | not a rim, and the noise share of the body is at most -1.0 dB (a tone under the noise) | the library: 9 samples (7 snares, 2 labelled clap + snare) read -1.0 to -9.1, the other 14 claps -0.3 to 0 |
| toms | `tom` | the clean hit is pitched | `profile_array`'s `pitched`; the library's toms are all pitched (53 to 578 Hz) |
| kick | `kick` | pitched, body at most 70 Hz | the library's 33 kicks: 28 to 56 Hz |

When several roles of one stem are toms they are ranked by body pitch:
`tom low`, `tom mid`, `tom high` (two toms: low and high; more mids get a
number). The rank is relative to the other toms in that stem, not a pitch class.
Two roles that get one name are numbered (`hh offbeat`, `hh offbeat 2`).

**Rules that are not there, and why.**

- **No `hh closed`.** The library's closed hats fall 20 dB in 18 to 130 ms and
  its open hats in 15 to 269 ms (12 and 10 hats measured; four of the open ones
  are under 30 ms). Nothing under 140 ms separates them, so a short decay names
  nothing. Over 140 ms: 0 of 12 closed hats, 5 of 10 open ones.
- **No `clap`.** Nothing `profile_array` reads separates a noise clap from a
  noise snare or a stick. The best rule found (noise body and under -20 dB in
  the 60-250 Hz band) names 9 of the library's 14 claps and 5 samples that are
  not claps (CMPCT SNAD, FX SNARE, SNARE STICK, STICK, STICK 1): 64 % right. A
  noise-bodied group in a snare stem stays neutral, with its measurements.
- **No `conga`.** The library has one conga: it reads unpitched, falls 40 dB in
  264 ms, inside the toms' 116 to 476 ms, and its high toms (290 to 578 Hz) sit
  where a conga would. A pitched hit in the toms stem is a `tom`: the word means
  "pitched, in the toms stem".
- The snare rule misses 4 of 11 plain snares (noise snares: the body has no
  tone to read). They stay neutral.

**Two groups that may be one sound.** The grouping cuts a continuum where the
sample puts the cut (the threshold's own limit, above). Two groups of one name
that play at the same 16ths (cosine of the two sets of 16ths with at least one hit, at least 0.8; counts per 16th were tried first and gave 0.68 for the real Flare pair, because one of the two plays mostly on the second offbeat) and are
within 1.15 times the threshold (1.73) of each other, one the other's nearest
group, are reported as **one role with variants** (each variant has its
decay, level and stability): on Flare two offbeat hat groups 1.54 apart with
different decays are this. 1.15 is chosen: it takes in the 1.54 and 1.53 of the
real pair and stays under the 1.77-1.82 at which the three Flare snare groups
held (stability 0.95-0.98). A group of that kind that is not eligible stays
neutral, and the role it is near lists it under `possibly_also`.

**Bleed.** Separators leak: a tom fill can sit 20 dB down in the kick stem. A
group is marked bleed of a group in another stem only when **all three** hold:

1. *Coincidence well over chance.* At least 70 % of its hits fall within 25 ms
   of a hit of that group, at least 25 points more than the two patterns give by
   chance, and the chance of so many from the patterns alone is at most 1 %.
   Chance, for each hit of the group at 16th s: the share of the track's bars
   (first to last hit of any stem) in which the other group plays at s, at least
   2 %; the hits taken as independent, `expected` is their mean and the chance
   of at least as many coincidences is a Poisson-binomial tail. A group that
   plays the beats against a group that plays every beat coincides every time
   and the chance rate is 100 %: that is the base rate, and the first builder
   called such groups bleed from a 96 % coincidence. The three numbers each
   guard a case: the rate a partial overlap, the margin a long stem where a
   small excess is "significant", the p value a few hits.
2. *Louder there.* The other group's hits are louder at the coincident moments
   (`level_db`, loudest ms) by at least 6 dB in at least 75 % of the pairs. 6 dB
   is chosen (half the amplitude); on Flare the snare stem's groups are louder
   than the coincident toms by 3 to 5 dB and the kick stem's second group is
   4.7 dB under the snare's hits: neither is bleed by level.
3. *Sounds like it.* The two clean hits are at most 2.0 apart. Chosen between
   the grouping threshold (1.5: the same sound) and the nearest distinct
   sounds measured (2.5 over a bleed bed, 2.8 for closed against open hat).
   Without a clean hit on either side the test is not made and the group is
   not marked; the evidence says so.

**One source split over several groups.** A separator, or the grouping, can split one
source into several groups of its stem, so that no single group coincides enough.
A group that fails the first test against every group of another stem is also
tested against the union of that stem's groups that were not marked (at least two):
the chance rate is computed the same way from the 16ths where any of them plays.
It never marks on coincidence alone: the level is read over all coincident pairs,
each hit at its own level, and the clean hit must be within 2.0 of the clean hit
of every group a coincident pair belongs to (the furthest counts). It is reported
separately (`union` in the answer; the evidence says "the union of groups ...").

A marked group stays in the output with its `evidence` (hits that coincide,
`rate`, `expected`, `p`, `louder_db`, `distance`) and is left out of the stem's
roles. A group that is not marked says which test it failed and the numbers. A
group with one to three hits can hardly pass the first test: three hits where
the other plays in a quarter of the bars is a 1.6 % chance. A lone bled hit
(the Flare replica's 81 Hz hit in the kick stem) is not found this way; it stays
a neutral group, and its note says why it was not marked.

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
