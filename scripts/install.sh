#!/usr/bin/env bash
# Symlink the remote script into Ableton's User Library.
# A symlink (not a copy) is deliberate: edits in the repo are live, so the
# "reload" op picks them up without reinstalling.
#
#   ./scripts/install.sh            install the remote script
#   ./scripts/install.sh --skills   symlink the skill into ~/.claude/skills
#                                   and ~/.codex/skills (whichever exist)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HOME/Music/Ableton/User Library/Remote Scripts"
SRC="$REPO/remote_script/AbletonLOM"
LINK="$DEST/AbletonLOM"

# --skills: teach the client the path grammar. Same symlink-not-copy reasoning
# as the remote script: the skill in the repo stays the source of truth.
link_skill() {
  local root="$1" client="$2"
  if [ ! -d "$(dirname "$root")" ]; then
    echo "$client: skipped ($(dirname "$root") does not exist)"
    return 0
  fi
  mkdir -p "$root"
  local link="$root/sideman"
  if [ -L "$link" ]; then
    rm "$link"
  elif [ -e "$link" ]; then
    echo "ERROR: $link exists and is not a symlink. Move it aside first." >&2
    return 1
  fi
  ln -s "$REPO/skills/sideman" "$link"
  echo "$client skill: $link -> $REPO/skills/sideman"
}

if [ "${1:-}" = "--skills" ]; then
  link_skill "$HOME/.claude/skills" "Claude Code"
  link_skill "$HOME/.codex/skills" "Codex"
  exit 0
fi

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
# Refresh the LOM census if Live is already up with the script enabled.
# On a first install it will not be, which is fine - this is best-effort and
# never fails the install. Re-run install.sh (or scripts/census.py) once Live
# is running to capture the census for your Live version.
echo
if "$REPO/scripts/census.py" --check >/dev/null 2>&1; then
  echo "census: already matches the running Live"
elif "$REPO/scripts/census.py"; then
  :
else
  echo "census: skipped (Live not reachable yet) - run scripts/census.py after step 3"
fi

echo
echo "Next:"
echo "  1. Restart Ableton Live"
echo "  2. Preferences > Link, Tempo & MIDI > Control Surface > AbletonLOM"
echo "  3. $REPO/scripts/lomcli.py ping"
echo "  4. $REPO/scripts/census.py      # capture the LOM map for your Live version"
