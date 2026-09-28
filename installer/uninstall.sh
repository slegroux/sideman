#!/usr/bin/env bash
# Remove a Sideman install. Shipped inside the payload, so the copy a user runs
# is at ~/Library/Application Support/Sideman/uninstall.sh
#
# Only symlinks that point into the payload are removed. A real directory at
# either location predates us and is left where it is.
set -euo pipefail

PAYLOAD="${SIDEMAN_PAYLOAD:-$HOME/Library/Application Support/Sideman}"

say() { echo "sideman: $*"; }

unlink_ours() {
  local link="$1"
  if [ -L "$link" ]; then
    case "$(readlink "$link")" in
      "$PAYLOAD"/*) rm "$link"; say "removed $link" ;;
      *) say "kept $link (does not point into $PAYLOAD)" ;;
    esac
  elif [ -e "$link" ]; then
    say "kept $link (not a symlink)"
  else
    say "absent $link"
  fi
}

unlink_ours "$HOME/Music/Ableton/User Library/Remote Scripts/AbletonLOM"
unlink_ours "$HOME/.claude/skills/sideman"

for c in claude "$HOME/.local/bin/claude" /usr/local/bin/claude \
         /opt/homebrew/bin/claude "$HOME/.npm-global/bin/claude"; do
  if command -v "$c" >/dev/null 2>&1; then
    "$c" mcp remove sideman -s user >/dev/null 2>&1 && say "unregistered from Claude Code" \
      || say "no 'sideman' MCP entry to remove"
    break
  fi
done

# A receipt that outlives the files makes the next install look like an upgrade
# of something that is no longer there. Non-fatal: a per-user install may have
# left no receipt at all.
# Skipped under the test harness (SIDEMAN_PAYLOAD set): receipts are not
# HOME-scoped, and a developer with a real install must keep theirs.
if [ -z "${SIDEMAN_PAYLOAD:-}" ]; then
  pkgutil --forget com.sideman.payload >/dev/null 2>&1 \
    && say "forgot the installer receipt" || true
fi

say "Live will still list AbletonLOM until you restart it - deselect it in"
say "Settings > Link, Tempo & MIDI if it is still in a Control Surface slot."

# Last, because this script lives inside the directory it is deleting - every
# line the user needs to read has to be printed before it goes. Guarded by the
# payload's own marker: refuse to rm -rf a directory that is not ours, however
# SIDEMAN_PAYLOAD was set.
if [ -d "$PAYLOAD" ] && grep -q '^name = "sideman"' "$PAYLOAD/pyproject.toml" 2>/dev/null; then
  rm -rf "$PAYLOAD"
  say "removed $PAYLOAD"
else
  say "kept $PAYLOAD (not a Sideman payload)"
fi
