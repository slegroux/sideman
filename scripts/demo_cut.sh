#!/usr/bin/env bash
# Cut the raw screen capture from scripts/demo.py into site/demo.mp4 + a poster.
#
# Trims the dead air at either end, scales to 1920 wide, drops audio, and burns
# one caption per beat. The caption offsets are NOT guessable: pass the beat
# times you read off the footage (scripts/demo.py prints its own beat clock, but
# the capture starts at its own moment, so confirm against the frames), already
# corrected by the same TRIM you pass here.
#
#   scripts/demo_cut.sh RAW.mov TRIM DURATION T1 T2 T3 [POSTER_AT]
#
#     TRIM      seconds to cut from the head of the raw capture
#     DURATION  seconds to keep after TRIM (must leave demo.mp4 <= 90s)
#     T1 T2 T3  each beat's start, in FINAL-video seconds (after TRIM)
#     POSTER_AT frame to grab for the poster, in final-video seconds
#
# e.g. scripts/demo_cut.sh /tmp/raw.mov 7 61 0.5 34.3 49.3 33
#
# Captions are rendered to PNG and overlaid rather than drawn with ffmpeg's
# drawtext: Homebrew's ffmpeg is built without libfreetype, so drawtext does not
# exist in it. rsvg-convert does the text, overlay does the rest.
set -euo pipefail

RAW=${1:?raw capture}
TRIM=${2:?trim seconds}
DUR=${3:?duration seconds}
T1=${4:?beat 1 offset}
T2=${5:?beat 2 offset}
T3=${6:?beat 3 offset}
POSTER_AT=${7:-$T2}

HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT="$HERE/site/demo.mp4"
POSTER="$HERE/site/demo-poster.jpg"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

HOLD=5          # seconds each caption stays up
FS=34           # caption font size

caption() {  # caption <index> <text>
  local n=$1 text=$2
  # No auto-sizing in SVG, so the box is measured from the string. 0.52em per
  # character is close enough for Arial at this size with the padding below.
  local pad=18
  local w=$(python3 -c "print(int(len('''$text''') * $FS * 0.52) + 2 * $pad)")
  local h=$((FS + 2 * pad))
  cat > "$WORK/cap$n.svg" <<SVG
<svg xmlns="http://www.w3.org/2000/svg" width="$w" height="$h">
  <rect x="0" y="0" width="$w" height="$h" rx="6" fill="black" fill-opacity="0.66"/>
  <text x="$pad" y="$((pad + FS - 8))" font-family="Arial, Helvetica, sans-serif"
        font-size="$FS" fill="white">$text</text>
</svg>
SVG
  rsvg-convert "$WORK/cap$n.svg" -o "$WORK/cap$n.png"
}

caption 1 'you: 4-bar house loop — chords, bass, drums'
caption 2 'you: open the filter over the loop'
caption 3 'you: I moved a fader — what changed?'

echo "cutting $RAW -> $OUT (trim ${TRIM}s, keep ${DUR}s)"
ffmpeg -nostdin -y -v warning -stats \
  -ss "$TRIM" -t "$DUR" -i "$RAW" \
  -i "$WORK/cap1.png" -i "$WORK/cap2.png" -i "$WORK/cap3.png" \
  -filter_complex "\
[0:v]scale=1920:-2[v0];\
[v0][1:v]overlay=46:H-h-46:enable='between(t,$T1,$(echo "$T1 + $HOLD" | bc))'[v1];\
[v1][2:v]overlay=46:H-h-46:enable='between(t,$T2,$(echo "$T2 + $HOLD" | bc))'[v2];\
[v2][3:v]overlay=46:H-h-46:enable='between(t,$T3,$(echo "$T3 + $HOLD" | bc))'" \
  -an -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -movflags +faststart \
  "$OUT"

echo "poster at ${POSTER_AT}s -> $POSTER"
ffmpeg -nostdin -y -v error \
  -ss "$(echo "$TRIM + $POSTER_AT" | bc)" -i "$RAW" \
  -frames:v 1 -vf scale=1920:-2 -q:v 4 "$POSTER"

ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_type,codec_name,width,height \
  -of default=nw=1 "$OUT"
ls -lh "$OUT" "$POSTER"
