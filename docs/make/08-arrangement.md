# 8. Arrangement: the whole track

**You'll have:** the Session clips laid into the Arrangement as a 5:30
track — intro 32 bars, build 16, drop 32, breakdown 16, build 16, drop 32,
outro 24 — with a named locator at every section.

## Say

> Lay this out as a track: intro 32 bars with drums and sub only, build 16,
> drop 32 with everything, breakdown 16 with pad and lead, build 16, drop 32,
> outro 24. Put a locator at each section.

## What Claude does

Places each clip at each section's start on the Arrangement timeline, tiles
the drops from the loops, and drops a locator at every section boundary.

## What you'll see in Live

Switch to Arrangement view (Tab): seven sections of clips, the drops full,
the breakdown thin, and seven locators along the top. Press play from the
start.

## Check

Ask: *"how long is the track and how many locators are there?"* — 168 bars,
7. Then: *"make the second drop 48 bars"* — Claude tiles more copies; a looped
clip can't be stretched in place, and it will tell you that's why.

::: {.callout-note}
## One time signature
Claude places things by bar and beat assuming the whole track is in one
meter — Live's API doesn't expose where a meter change happens. If your track
changes time signature, place things after the change yourself.
:::
