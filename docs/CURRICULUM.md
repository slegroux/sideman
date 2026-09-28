# Curriculum: a melodic house track, start to finish

Ten lessons that build one track — 122 BPM, F minor, five and a half minutes,
intro → build → drop → breakdown → second drop → outro — and along the way
touch every one of Sideman's 30 tools where a producer would actually reach
for it. Each lesson leaves a checkable state in the Set, so you can verify
where you are with a query rather than a feeling.

Every lesson has two voices: **the ask**, what a musician types into Claude,
and **underneath**, the calls Claude makes. Read the first if you produce,
the second if you build on the server. [docs/TUTORIAL.md](TUTORIAL.md) is the
short version of lessons 1–4; start there if you have ten minutes.

Prerequisites: Sideman installed, Live open on a **fresh Set** (⌘N), Claude
Code with the `sideman` skill. Nothing in a lesson touches a Set you did not
build in the previous one.

| # | Lesson | Musical outcome | Tools introduced |
|---|--------|-----------------|------------------|
| 1 | Orientation | tempo, key, an empty Set you can read | `lom_ping` `lom_describe` `lom_search` `lom_get` `lom_set` `lom_count` `lom_types` |
| 2 | Drums | kick, hats, clap, a groove | `browser_list` `browser_load` `clip_add_notes` `clip_get_notes` `clip_modify_notes` `lom_call` |
| 3 | Bass | sub and mid bass locked to the kick | `lom_transaction` `clip_envelope_insert_step` `clip_envelope_get` |
| 4 | Harmony | a four-chord progression on a pad | `lom_set_batch` `lom_canonical_path` |
| 5 | Melody and arp | a plucked lead and an arpeggio | `clip_remove_notes` `lom_get_batch` |
| 6 | Sound design | racks, macros, sends, a plug-in | `lom_call` with `__path__`, `lom_get` paging |
| 7 | Movement | filter builds, reverb throws, risers | `clip_envelope_clear` |
| 8 | Arrangement | the full 5:30 structure with locators | `arrangement_create_clip` `arrangement_duplicate_clip` `arrangement_list_clips` |
| 9 | Audio | a vocal chop, warped and placed | `clip_set_warp_markers` |
| 10 | Mixing with a human in the room | levels, sends, and reacting to what you touch | `lom_observe` `lom_poll_events` `lom_observers` `lom_unobserve` `lom_unobserve_all` |

---

## 1. Orientation — the Set is a graph

**Outcome:** 122 BPM, F minor, and a mental model of paths.

**The ask:** *"Set the tempo to 122 and the key to F minor. Then tell me what's
in this Set."*

**Underneath:** `lom_set("live_set", "tempo", 122)`; `lom_set("live_set",
"root_note", 5)` and `scale_name` `"Minor"`; `lom_describe("live_set")` for
the children (`tracks`, `scenes`, `return_tracks`, `master_track`); `lom_count`
on each. `lom_types` once, to see the 47 types the census measured.

**Lesson inside the lesson:** paths are space-separated and zero-indexed —
`live_set tracks 0 mixer_device volume` — and `lom_describe` is how Claude
learns what exists rather than guessing. A guess costs a failed round trip;
`describe` costs one and is right.

**Checkpoint:** `lom_get("live_set", "tempo") → 122.0`,
`lom_get("live_set", "scale_name") → "Minor"`, `lom_count("live_set",
"tracks") → 4` (a new Set's defaults).

## 2. Drums — the kick is the clock

**Outcome:** a Drums track with a 909-style kit, four-on-the-floor kick,
off-beat open hats, clap on 2 and 4, 16th closed hats with alternating
velocity. 4 bars, looping, swung slightly.

**The ask:** *"Make a Drums track with a 909 kit. Kick on every beat, claps on
2 and 4, open hats off-beat, closed hats on 16ths with a little velocity
movement. Four bars, then loop it."*

**Underneath:** `lom_call("live_set", "create_midi_track", [-1])`;
`browser_list("drums")` then `browser_load("drums/<kit>", track_index=…)`;
`lom_call(clip_slot, "create_clip", [16.0])`; one `clip_add_notes` with the
whole pattern (kick 36, clap 39, closed hat 42, open hat 46); `clip_get_notes`
to read back `note_id`s; `clip_modify_notes` to shape hat velocities.

**Lesson inside the lesson:** notes cannot round-trip through generic
`get`/`set` — the typed `clip_*` tools exist for exactly that. Times are
beats, pitch is a MIDI number (C3 = 60).

**Checkpoint:** `clip_get_notes(...)` → 40 notes; `lom_get(clip, "length") →
16.0`; `lom_get(clip, "looping") → true`.

## 3. Bass — sub and mid, and the first automation

**Outcome:** two bass tracks (Sub, Mid) on Drift, root-note pattern locked
to the kick, mid bass with a filter that opens over the phrase.

**The ask:** *"Add a Sub and a Mid bass on Drift. Sub plays F on the off-beats
under the kick; Mid plays a syncopated F–Ab–C line an octave up. Name and
colour both in one undo. Open the Mid's filter over the four bars."*

**Underneath:** two `create_midi_track`, `browser_load("instruments/Drift")`
×2, `lom_transaction` for four writes (two names, two `color_index`), notes,
then `lom_search(type="DeviceParameter", root=<Mid track>)` to find **LP
Freq** — Drift does not call it Cutoff — and `clip_envelope_insert_step` ×4
(0.35 → 0.5 → 0.72 → 0.95), `clip_envelope_get` to read it back.

**Lesson inside the lesson:** device parameters are normalised 0–1; read
`min`/`max` before writing. Automation is a *(clip, parameter)* pair —
two paths, not one. And this is the capability the Max for Live object
model cannot reach; it is why Sideman is a Remote Script.

**Checkpoint:** `clip_envelope_get(...)` → 4 steps ending at 0.95;
`lom_get("live_set", "can_undo") → true` and one ⌘Z removes all four
name/colour writes together.

## 4. Harmony — the progression, in one undo

**Outcome:** a Pad track with Fm – Db – Ab – Eb (i – VI – III – VII), one
chord per bar, voiced in the 3rd–4th octave with the 7th on the i.

**The ask:** *"Add a Pad on Drift with a warm preset. Play Fm7, Db, Ab, Eb —
one bar each, close voicings around C4. Keep the velocities gentle."*

**Underneath:** `browser_list("instruments/Drift")` to pick a preset;
`clip_add_notes` with the voiced chords; `lom_set_batch` to set the pad
track's `mixer_device volume` and a send in one round trip;
`lom_canonical_path("live_set view selected_track")` to resolve whatever
the user has selected into a stable path.

**Lesson inside the lesson:** undo grouping is partial. Names, colours and
time signatures collapse into one ⌘Z through `lom_transaction`; tempo,
volume, mute and device parameters always get their own step — Live decides
that, not the server. Claude should say which is which rather than promise
"one undo".

**Checkpoint:** `clip_get_notes(pad clip)` → 15 notes across 4 bars;
`lom_get("live_set tracks N mixer_device volume", "value")` → what was set.

## 5. Melody and arp — the hook

**Outcome:** a Lead track with a plucked top line (call in bars 1–2,
response in 3–4) and an Arp track running a 16th-note arpeggio of each
chord.

**The ask:** *"Write a plucky lead melody over the chords — a two-bar call
and a two-bar answer, mostly scale tones, one blue note. Then an arp track
that runs each chord as rising 16ths. Humanise the arp velocities."*

**Underneath:** notes derived from the chord tones (F minor scale intervals
from `lom_get("live_set", "scale_intervals")`); `clip_modify_notes` for
velocity humanisation and `clip_remove_notes` to thin a phrase;
`lom_get_batch` to read every track's name, colour and clip length in one
call before deciding.

**Lesson inside the lesson:** `lom_get_batch` isolates failures — one
unavailable property does not lose the other reads.

**Checkpoint:** two new tracks; arp clip has 64 notes (16 per bar);
`clip_get_notes(lead)` shows the blue note (Cb/B, pitch 59 or 71).

## 6. Sound design — racks, macros, sends, and the plug-in wall

**Outcome:** the Lead through an Audio Effect Rack with a filter and delay on
macros; sends to the A Reverb and B Delay returns; one third-party plug-in
loaded and honestly described.

**The ask:** *"Put an effect rack on the Lead with a filter and a ping-pong
delay, expose cutoff and delay feedback as macros. Send the pad to the
reverb return. Load Pro-Q on the master and tell me what you can control."*

**Underneath:** `browser_load("audio_effects/...")`; `lom_call(rack,
"add_macro", ...)`; `lom_set` on `macros_mapped`/macro parameters;
`lom_set("... mixer_device sends 0", "value", 0.4)`; for the master there is no
track index — `browser_load` loads onto the *selected* track, so select it
first: `lom_set("live_set view", "selected_track", {"__path__": "live_set
master_track"})` then `browser_load("plugins/...")`; then `lom_get(device,
"parameters", limit=25)` and `lom_call(device, "get_parameter_names",
limit=50)`.

**Lesson inside the lesson:** a VST/AU exposes only `Device On` until the
user presses **Configure** in Live — a GUI-only step with no API. Claude can
read the plug-in's full parameter names and tell the user exactly what to
expose; it cannot press the button. Arguments that are Live objects go in
`{"__path__": "..."}` markers, never as bare strings. Long vectors page with
`offset`/`limit`.

**Checkpoint:** `lom_count(master, "devices") → 1`;
`lom_get(plugin, "parameters", limit=5)` → count 1 before Configure.

## 7. Movement — builds, throws, risers

**Outcome:** a 16-bar build clip set where the pad filter opens, the reverb
send rises, and a white-noise riser on a Noise track climbs to the drop.

**The ask:** *"Make a 16-bar build: pad filter closed to open, reverb send
from 0.2 to 0.8, and a noise riser that climbs. Clear the old sweep on the
Mid bass first."*

**Underneath:** `clip_envelope_clear` on the lesson-3 sweep;
`clip_envelope_insert_step` on three different parameters across three
clips; `lom_call(clip_slot, "duplicate_clip_to", ...)` to spin 4-bar loops
into 16-bar build variants.

**Lesson inside the lesson:** sampling an envelope exactly on a step edge
returns the pre-envelope value — Live evaluates the boundary that way.
Check inside a step.

**Checkpoint:** `clip_envelope_get` on each of the three parameters returns
a rising series; the noise clip is 64 beats long.

## 8. Arrangement — the 5:30 structure

**Outcome:** Session clips laid into the Arrangement: 32-bar intro, 16-bar
build, 32-bar drop, 16-bar breakdown, 16-bar build, 32-bar drop, 24-bar
outro — with named locators at each section.

**The ask:** *"Lay this out as a track: intro 32 bars with drums and sub
only, build 16, drop 32 with everything, breakdown 16 with pad and lead,
build 16, drop 32, outro 24. Put a locator at each section."*

**Underneath:** `arrangement_create_clip(track, start_time, length)` per
section per track; `arrangement_duplicate_clip` to tile the drops;
`arrangement_list_clips` to verify; `lom_call("live_set",
"set_or_delete_cue")` after moving the playhead (`current_song_time`) to each
section start.

**Lesson inside the lesson:** a looped arrangement clip cannot be lengthened
in place; tiling is the honest operation. Bar positions assume one time
signature — Live's API exposes no list of meter changes.

**Checkpoint:** `arrangement_list_clips` → clip count per track matches the
section plan; `lom_count("live_set", "cue_points") → 7`;
`lom_get("live_set", "song_length")` ≈ 168 bars × 4 beats.

## 9. Audio — a vocal chop, warped

**Outcome:** an Audio track with a vocal phrase (a Core Library sample or
your own), warped to 122, chopped to land on the breakdown.

**The ask:** *"Bring in a vocal sample, warp it to the tempo, and place two
chops in the breakdown on the off-beats."*

**Underneath:** `browser_list("sounds")` / `browser_load` onto an audio
track (or `arrangement_create_clip` with a sample); `lom_get(clip,
"warp_markers")` → `clip_set_warp_markers(...)` because JSON cannot carry a
WarpMarker; `lom_set(clip, "warp_mode", ...)`; `arrangement_duplicate_clip`
for the second chop.

**Lesson inside the lesson:** `clip_set_warp_markers` is one of the few
typed tools — it exists only because generic access genuinely cannot work
there. Everything else about the audio clip is plain `get`/`set`.

**Checkpoint:** `lom_get(clip, "warping") → true`; `lom_get(clip,
"warp_markers")` shows the markers set; two clips on the audio track in the
breakdown range.

## 10. Mixing with a human in the room

**Outcome:** a balanced mix — levels, pans, sends, a glue compressor and
limiter on the master — done *with* you, not for you.

**The ask:** *"Watch the mixer. I'm going to set the drums and bass by ear;
then balance everything else around them, and tell me every time I change
something."*

**Underneath:** `lom_observe` on each track's `mixer_device volume value`
and on `output_meter_level`; you move faders by hand; Claude reads
`lom_poll_events`, balances the rest with `lom_set_batch`, and reports back;
`lom_observers` to list what is watched, `lom_unobserve_all` at the end (0
leaked). `browser_load("audio_effects/Glue Compressor")` and a Limiter onto
`live_set master_track`, selected first the way lesson 6 did.

**Lesson inside the lesson:** availability is per instance — MIDI tracks
have no `input_meter_left`, only the main track has a `crossfader`;
`lom_describe` reports these under `unavailable`. Observers are a pull with
a ring buffer behind them; they catch what the *user* does in the GUI,
which nothing else in the field reports. And the registry is shared across
clients (issue #2): take a `latest_seq` baseline, never `observe_clear`
what you did not create. Destructive calls — `delete_*`, `remove_*`,
`clear_*`, `crop` — refuse to run without `confirm=true`; the Set is not in
version control. Saving is yours: ⌘S.

**Checkpoint:** `lom_observers` → 0 after cleanup; master has 2 devices;
every track's volume is what you or Claude last set; the arrangement plays
from bar 1 to the end without a gap.

---

## How the lessons are kept true

Each lesson's *Underneath* and *Checkpoint* are run against a live Set when
the lesson is written, the way [TUTORIAL.md](TUTORIAL.md) was. A lesson that
cannot be checked with a query is not finished. When Live changes a member
these lessons name, `lom_describe` is the source of truth and the lesson is
wrong — fix the lesson.
