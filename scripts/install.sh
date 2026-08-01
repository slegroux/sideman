#!/usr/bin/env bash
# Symlink the remote script into Ableton's User Library.
# A symlink (not a copy) is deliberate: edits in the repo are live, so the
# "reload" op picks them up without reinstalling.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HOME/Music/Ableton/User Library/Remote Scripts"
SRC="$REPO/remote_script/AbletonLOM"
LINK="$DEST/AbletonLOM"

[ -d "$SRC" ] || { echo "missing $SRC" >&2; exit 1; }
mkdir -p "$DEST"

if [ -L "$LINK" ]; then
  echo "replacing existing symlink $LINK"
  rm "$LINK"
elif [ -e "$LINK" ]; then
  echo "ERROR: $LINK exists and is not a symlink. Move it aside first." >&2
  exit 1
fi

ln -s "$SRC" "$LINK"
echo "linked: $LINK -> $SRC"
echo
echo "Next:"
echo "  1. Restart Ableton Live"
echo "  2. Preferences > Link, Tempo & MIDI > Control Surface > AbletonLOM"
echo "  3. $REPO/scripts/lomcli.py ping"
