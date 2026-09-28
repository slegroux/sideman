# Demo shooting script

The landing-page video should show cause and effect at once: Claude Code
receiving the words and calling the tools on one side, the Session View
changing on the other. Three beats, under 90 seconds cut, no narration.

`scripts/demo.py` drives the same three beats without Claude and is useful
for a dry run; this script is for the real take.

## Setup (two minutes)

1. Live: **⌘N** for a fresh Set. Leave the Session View grid and the browser
   visible; the clips land in scene 1. Set nothing else — tempo, tracks and
   instruments all arrive on camera.
2. Terminal: a window floated over Live's browser panel (left third), not
   over the grid. Big font (18 pt+); the tool calls have to be legible at
   1920 wide. In it, from anywhere: `claude`. The `sideman` server is
   registered at user scope, so the `lom_*` / `clip_*` / `browser_*` tools
   are there, and the skill teaches it the path grammar.
3. Record the region covering both windows, not the whole display — a
   3440-wide full-screen capture is silently time-compressed and films the
   desktop:
   `screencapture -v -x -R <x,y,w,h> -V 150 raw.mov` (150 s ceiling), or
   QuickTime → New Screen Recording → drag the region.
4. Start recording, wait two seconds, type beat 1.

## The three beats

Type each prompt as written — the phrasing is what a musician would say,
and it is what the tutorial does. Wait for Claude's reply before the next.

### Beat 1 — the loop lands (~35 s)

> Give me a 4-bar house loop at 124 BPM: three MIDI tracks named Chords,
> Bass and Drums, coloured differently. Drift on Chords and Bass, a 909 kit
> on Drums. Write chords, a bassline and a kick-and-hats pattern, then play
> scene 1.

On screen: `create_midi_track` ×3, `lom_transaction` for names and colours,
`browser_load` for Drift and the kit, `clip_add_notes` per track, then
`fire` and `start_playing`. In Live: the empty grid gets three coloured
tracks, three clips appear in scene 1, transport starts.

### Beat 2 — the sweep is automation, not a knob (~25 s)

> Open the bass filter over the loop — closed at the start, wide open by
> bar 4 — as clip automation, and show me the envelope.

On screen: `lom_search(type="DeviceParameter")` finding **LP Freq** (Drift
does not call it Cutoff), `clip_envelope_insert_step` ×4 (0.35 → 0.5 →
0.72 → 0.95, one per bar), then the Clip View switched to the Envelopes tab
with *Drift / LP Freq / Automation* and the staircase drawn. This is the beat
no other Ableton MCP server can do; hold on it a moment.

### Beat 3 — it saw what you did (~20 s)

While the loop plays, **drag the Chords fader by hand** in Live. Then:

> I just moved something on the mixer — what changed?

On screen: `lom_observe` was set (Claude may set it here if it hasn't), then
`lom_poll_events` returning the volume events with their values; Claude
names the track and the move. In Live: nothing changes — that is the point.
Stop the transport when Claude has answered.

## Cut

`scripts/demo_cut.sh RAW.mov TRIM DURATION T1 T2 T3 [POSTER_AT]` — read the
three beat starts off the raw footage (the moment each prompt is sent), pass
them corrected by TRIM, keep DURATION ≤ 90. Take the poster from beat 2 with
the full staircase visible. Delete the raw capture afterwards.

## What not to do

- Do not record a Set you care about; the fresh Set is the demo.
- Do not narrate over it — captions are burned in by the cut script.
- Do not tidy Claude's replies; the point is that they are real.
