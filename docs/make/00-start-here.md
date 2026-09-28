# Start here: three things to say

You need Live open on a new Set (⌘N), Claude Code with Sideman installed, and
ten minutes. Type each of these, wait for Claude to finish, and watch Live.

## 1. A loop from nothing

> Give me a 4-bar house loop at 122 BPM: three MIDI tracks named Chords, Bass
> and Drums, coloured differently. Drift on Chords and Bass, a 909 kit on
> Drums. Write chords, a bassline and a kick-and-hats pattern, then play
> scene 1.

**What happens in Live:** three coloured tracks appear, instruments load from
the browser, three clips land in scene 1, the transport starts. Claude will
tell you what it wrote — how many notes, which kit.

**Check:** press ⌘Z once. The three track names and colours go together,
because Claude set them as one undo step. Press ⌘⇧Z to get them back.

## 2. Automation, not a knob

> Open the bass filter over the loop — closed at the start, wide open by bar
> 4 — as clip automation, and show me the envelope.

**What happens in Live:** the Clip View switches to the Envelopes tab on the
bass clip, labelled *Drift / LP Freq / Automation*, and a four-step staircase
is drawn — one step per bar. The bass opens up as the loop plays.

**Check:** ask *"what's the envelope on the bass clip?"* — Claude reads it
back from Live rather than from memory.

## 3. It saw what you did

Drag the Chords fader down by hand while the loop plays. Then:

> I just moved something on the mixer — what changed?

**What happens in Live:** nothing — that's the point. Claude names the track
and the move, because it watches the Set, not only its own actions.

**Check:** move a different fader and ask again.

That is the whole idea. The ten lessons that follow build a full track the
same way: you say it, it lands in your Set, and you can always check.
