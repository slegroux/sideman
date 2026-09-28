# 5. Melody and arp: the hook

**You'll have:** a Lead track with a plucked melody — a two-bar call, a
two-bar answer, one blue note — and an Arp track running each chord as
rising 16ths.

## Say

> Write a plucky lead melody over the chords — a two-bar call and a two-bar
> answer, mostly scale tones, one blue note. Then an arp track that runs each
> chord as rising 16ths. Humanise the arp velocities.

## What Claude does

Reads the key and the pad's chords back from Live, derives the scale, writes
the melody and the arpeggio, then nudges the arp velocities so they aren't
identical. Before deciding anything it reads all the tracks in one pass.

## What you'll see in Live

Two more tracks. The arp clip is dense — sixteen notes a bar. The lead clip
has one note outside the F minor scale; hover it and you'll see B.

## Check

Ask: *"which note in the lead is the blue note?"* Then: *"take out every
other note in bar 3 of the arp"* — Claude removes them by their note ids, not
by rewriting the clip.
