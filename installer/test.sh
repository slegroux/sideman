#!/usr/bin/env bash
# Acceptance for the .pkg installer.
#
#   installer/test.sh
#
# Never touches the real ~/Music/Ableton, ~/.claude or the developer's install:
# postinstall runs with HOME pointed at throwaway directories and with a stub
# `claude` on PATH that only records its argv. What this cannot cover is
# Installer.app itself - whether the GUI installs without an admin password has
# to be checked once by hand in a clean user account.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$REPO/installer/.cache/build/payload"
REL="Library/Application Support/Sideman"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO/pyproject.toml" | head -1)"
PKG="$REPO/dist/Sideman-$VERSION.pkg"
# Proves the bundled mcp is API-compatible with the one the suite runs against,
# which merely importing the module would not.
TOOLS='import asyncio, mcp_server.server as S; print(len(asyncio.run(S.mcp.list_tools())))'

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fails=0

ok()   { echo "PASS  $1"; }
nope() { echo "FAIL  $1"; fails=$((fails + 1)); }
# assert <description> <command...>
assert() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$d"; else nope "$d"; fi; }
# assert_eq <description> <actual> <expected>
assert_eq() {
  if [ "$2" = "$3" ]; then ok "$1"; else nope "$1"; echo "        want: $3"; echo "        got:  $2"; fi
}
# -P and a neutral cwd both: without them `import mcp_server` finds the repo's
# own source tree, and a wheel that never installed would still look installed.
pyrun() { local pl="$1"; shift; (cd / && "$pl/python/bin/python3" -P "$@"); }

# Stand in for Installer.app delivering the payload into a home directory.
# An APFS clone, not a copy: the payload is 160 MB and gets delivered into four
# throwaway homes, which a real copy cannot afford. Clones are indistinguishable
# once written, and ditto is the fallback on a non-APFS volume.
deliver() {
  mkdir -p "$1"
  cp -Rc "$STAGE/." "$1" 2>/dev/null || ditto "$STAGE" "$1"
}

# ---------------------------------------------------------------- (a) build --
echo "== a. build =="
if "$REPO/installer/build.sh" >"$TMP/build.log" 2>&1; then
  ok "build.sh succeeded"
else
  nope "build.sh succeeded"; tail -20 "$TMP/build.log"; echo "INSTALLER FAILED (1)"; exit 1
fi
assert "$(basename "$PKG") exists" test -f "$PKG"
# A staging tree deleted mid-build (two concurrent builds) still produces a
# well-formed pkg - just an almost empty one. Name that failure mode directly.
assert "pkg is a plausible size (>50 MB)" \
  bash -c "[ \"\$(stat -f%z '$PKG')\" -gt 52428800 ]"

pkgutil --expand "$PKG" "$TMP/x" >/dev/null 2>&1
CPKG="$TMP/x/sideman-component.pkg"
assert "expanded product has the component pkg" test -d "$CPKG"
assert "component carries the postinstall script" test -f "$CPKG/Scripts/postinstall"
assert "installs relative to the home domain" \
  grep -q 'install-location="Library/Application Support/Sideman"' "$CPKG/PackageInfo"
assert "component needs no root authorisation" grep -q 'auth="none"' "$CPKG/PackageInfo"
assert "distribution enables the current user home" \
  grep -q 'enable_currentUserHome="true"' "$TMP/x/Distribution"
# productbuild reformats the distribution, so match the attribute, not a line.
assert "distribution requires macOS 11 or later" \
  grep -q 'os-version min="11.0"' "$TMP/x/Distribution"
assert "conclusion screen is bundled" test -f "$TMP/x/Resources/conclusion.html"
assert "conclusion screen covers the unsigned-pkg prompt" \
  grep -q 'Open Anyway' "$TMP/x/Resources/conclusion.html"
assert "conclusion screen explains CONFLICT.txt" \
  grep -q 'CONFLICT.txt' "$TMP/x/Resources/conclusion.html"

# lsbom reads the payload manifest without decompressing 160 MB of Payload.
lsbom -s "$CPKG/Bom" > "$TMP/bom.txt" 2>/dev/null
for entry in ./mcp_server/server.py ./remote_script/AbletonLOM/__init__.py \
             ./remote_script/AbletonLOM/handlers.py ./sideman/client.py \
             ./skills/sideman/SKILL.md ./pyproject.toml ./LICENSE ./uninstall.sh \
             ./python-aarch64-apple-darwin/bin/python3 \
             ./python-x86_64-apple-darwin/bin/python3; do
  assert "payload contains $entry" grep -qxF "$entry" "$TMP/bom.txt"
done
assert "payload contains the sideman wheel" \
  grep -q '^\./wheels/sideman-.*\.whl$' "$TMP/bom.txt"
assert "payload contains the mcp dependency wheels" \
  grep -q '^\./wheels/mcp-.*\.whl$' "$TMP/bom.txt"
# The bundled interpreter has its own stdlib __pycache__; ours must not ship.
assert "our source trees carry no __pycache__" \
  bash -c "! grep -qE '^\./(mcp_server|remote_script|skills)/.*__pycache__' '$TMP/bom.txt'"

# The bundled mcp is pinned from the dev venv, not hardcoded.
DEV_MCP="$("$REPO/.venv/bin/python" -c 'import importlib.metadata as m; print(m.version("mcp"))' 2>/dev/null)"
assert_eq "bundled mcp wheel is the dev venv's version" \
  "$(ls "$STAGE/wheels" | grep -c "^mcp-$DEV_MCP-")" "1"

rm -rf "$TMP/x"   # the expanded pkg is another copy of the payload

# ------------------------------------------------------------ shared harness --
mkdir -p "$TMP/bin"
cat > "$TMP/bin/claude" <<'STUB'
#!/bin/bash
printf '%s\n' "$*" >> "$CLAUDE_STUB_LOG"
# CLAUDE_STUB_FAIL=1 makes `mcp add` fail, the way a corrupt ~/.claude.json does.
[ "${CLAUDE_STUB_FAIL:-}" = 1 ] && [ "$1" = mcp ] && [ "$2" = add ] && exit 1
exit 0
STUB
chmod +x "$TMP/bin/claude"
STUB_LOG="$TMP/claude-argv.txt"
: > "$STUB_LOG"

# env -i reproduces Installer.app's sparse environment: no Homebrew, no profile.
# argv mirrors Installer's: $1 the package, $2 the install destination.
run_postinstall() {
  local home="$1" payload="${2:-}" arg2="${3:-}"
  local -a e=(USER="${USER:-tester}" TMPDIR="$TMP"
              PATH="$TMP/bin:/usr/bin:/bin:/usr/sbin:/sbin"
              CLAUDE_STUB_LOG="$STUB_LOG" CLAUDE_STUB_FAIL="${CLAUDE_STUB_FAIL:-}")
  e+=(HOME="$home")
  if [ -n "$payload" ]; then e+=(SIDEMAN_PAYLOAD="$payload"); fi
  env -i "${e[@]}" /bin/bash "$REPO/installer/scripts/postinstall" /dev/null "$arg2"
}

# ------------------------------------------------------- (b) fresh install --
echo
echo "== b. postinstall into a fresh HOME =="
FAKE="$TMP/home"
PAYLOAD="$FAKE/$REL"
RS_LINK="$FAKE/Music/Ableton/User Library/Remote Scripts/AbletonLOM"
SKILL_LINK="$FAKE/.claude/skills/sideman"

assert "staging tree from build.sh is present" test -d "$STAGE"
mkdir -p "$FAKE/Music/Ableton/User Library" "$FAKE/.claude"
deliver "$PAYLOAD"

if run_postinstall "$FAKE" "" "" >"$TMP/post1.log" 2>&1; then
  ok "postinstall exited 0"
else
  nope "postinstall exited 0"; cat "$TMP/post1.log"
fi
assert "remote script path is a symlink" test -L "$RS_LINK"
assert_eq "remote script symlink targets the payload" \
  "$(readlink "$RS_LINK" 2>/dev/null)" "$PAYLOAD/remote_script/AbletonLOM"
assert "remote script symlink resolves" test -f "$RS_LINK/__init__.py"
assert "skill path is a symlink" test -L "$SKILL_LINK"
assert "skill symlink resolves into the payload" test -f "$SKILL_LINK/SKILL.md"
assert "matching runtime became python/" test -x "$PAYLOAD/python/bin/python3"
assert "the other architecture was pruned" \
  bash -c "! ls -d '$PAYLOAD'/python-*-apple-darwin >/dev/null 2>&1"
assert "no CONFLICT.txt on a clean install" test ! -e "$PAYLOAD/CONFLICT.txt"
assert_eq "claude was called to register the server" \
  "$(grep -cxF "mcp add --scope user sideman -- $PAYLOAD/python/bin/python3 -m mcp_server.server" "$STUB_LOG")" "1"
assert "private python imports mcp_server.server" pyrun "$PAYLOAD" -c "import mcp_server.server"
assert "private python imports the sideman client" pyrun "$PAYLOAD" -c "import sideman.client"
DEV_TOOLS="$(cd / && "$REPO/.venv/bin/python" -P -c "$TOOLS" 2>/dev/null)"
assert_eq "bundled server exposes the same tools as the dev venv ($DEV_TOOLS)" \
  "$(pyrun "$PAYLOAD" -c "$TOOLS" 2>/dev/null)" "$DEV_TOOLS"

# Resolving for the architecture this Mac is not proves the other half of
# wheels/ is complete - otherwise an x86_64 user is the one who finds out.
OTHER_PLAT=macosx_10_12_x86_64
[ "$(uname -m)" = x86_64 ] && OTHER_PLAT=macosx_11_0_arm64
assert "wheels/ resolves offline for $OTHER_PLAT too" \
  pyrun "$PAYLOAD" -m pip install --quiet --no-index \
    --find-links "$PAYLOAD/wheels" --platform "$OTHER_PLAT" \
    --python-version 3.11 --only-binary=:all: --target "$TMP/othertarget" sideman
rm -rf "$TMP/othertarget"

# ---------------------------------------------------------- (c) idempotence --
echo
echo "== c. re-run against the same payload is idempotent =="
if run_postinstall "$FAKE" "" "" >"$TMP/post2.log" 2>&1; then
  ok "second postinstall exited 0"
else
  nope "second postinstall exited 0"; cat "$TMP/post2.log"
fi
assert_eq "remote script symlink unchanged" \
  "$(readlink "$RS_LINK" 2>/dev/null)" "$PAYLOAD/remote_script/AbletonLOM"
assert "skill symlink still resolves" test -f "$SKILL_LINK/SKILL.md"
assert "private python still imports mcp_server.server" pyrun "$PAYLOAD" -c "import mcp_server.server"
assert_eq "registration was repointed, not duplicated" \
  "$(grep -cxF "mcp add --scope user sideman -- $PAYLOAD/python/bin/python3 -m mcp_server.server" "$STUB_LOG")" "2"

# ------------------------------------------------------------- (d) reinstall --
echo
echo "== d. reinstall replaces the interpreter and the installed server =="
# Its own home: this case mutates the payload, and (g) still needs (b)'s intact.
FAKED="$TMP/homed"
PAYLOADD="$FAKED/$REL"
mkdir -p "$FAKED/Music/Ableton/User Library" "$FAKED/.claude"
deliver "$PAYLOADD"
if run_postinstall "$FAKED" "" "" >"$TMP/post_d1.log" 2>&1; then
  ok "first install into its own home exited 0"
else
  nope "first install into its own home exited 0"; cat "$TMP/post_d1.log"
fi
SITE="$(pyrun "$PAYLOADD" -c 'import mcp_server, os; print(os.path.dirname(mcp_server.__file__))' 2>/dev/null)"
assert "found the installed mcp_server in site-packages" test -d "$SITE"
touch "$SITE/STALE_MARKER"
assert "marker planted in site-packages" test -f "$SITE/STALE_MARKER"
deliver "$PAYLOADD"    # Installer re-delivers, bringing python-<arch> back
if run_postinstall "$FAKED" "" "" >"$TMP/post_re.log" 2>&1; then
  ok "reinstall postinstall exited 0"
else
  nope "reinstall postinstall exited 0"; cat "$TMP/post_re.log"
fi
assert "stale site-packages was replaced, not reused" test ! -e "$SITE/STALE_MARKER"
assert "reinstalled server still imports" pyrun "$PAYLOADD" -c "import mcp_server.server"
assert_eq "reinstalled server still exposes every tool" \
  "$(pyrun "$PAYLOADD" -c "$TOOLS" 2>/dev/null)" "$DEV_TOOLS"
rm -rf "$FAKED"

# --------------------------------------------- (e) conflict is not an abort --
echo
echo "== e. a real directory at the Remote Script path is reported, not deleted =="
FAKE2="$TMP/home2"
PAYLOAD2="$FAKE2/$REL"
REAL="$FAKE2/Music/Ableton/User Library/Remote Scripts/AbletonLOM"
SKILL_LINK2="$FAKE2/.claude/skills/sideman"
mkdir -p "$REAL" "$FAKE2/.claude"
echo keep > "$REAL/someones_script.py"
deliver "$PAYLOAD2"
if run_postinstall "$FAKE2" "" "" >"$TMP/post3.log" 2>&1; then
  ok "postinstall exited 0 despite the conflict"
else
  nope "postinstall exited 0 despite the conflict"; cat "$TMP/post3.log"
fi
assert "the real directory survived" test -f "$REAL/someones_script.py"
assert "it is still a directory, not a symlink" bash -c "[ -d '$REAL' ] && [ ! -L '$REAL' ]"
assert "CONFLICT.txt was written" test -f "$PAYLOAD2/CONFLICT.txt"
assert "CONFLICT.txt names the exact path" grep -qF "$REAL" "$PAYLOAD2/CONFLICT.txt"
assert "CONFLICT.txt says what to do" grep -q 'ln -s' "$PAYLOAD2/CONFLICT.txt"
assert "the conflict was printed too" grep -qF "CONFLICT.txt" "$TMP/post3.log"
# everything else still installed
assert "interpreter installed anyway" test -x "$PAYLOAD2/python/bin/python3"
assert "server installed anyway" pyrun "$PAYLOAD2" -c "import mcp_server.server"
assert "skill linked anyway" test -f "$SKILL_LINK2/SKILL.md"
assert_eq "registered with Claude Code anyway" \
  "$(grep -cxF "mcp add --scope user sideman -- $PAYLOAD2/python/bin/python3 -m mcp_server.server" "$STUB_LOG")" "1"

rm -rf "$FAKE2"

# ------------------------------------------------- (f) hostile environment --
echo
echo "== f. HOME=/var/root and no USER: the install destination argument wins =="
FAKE3="$TMP/home3"
PAYLOAD3="$FAKE3/$REL"
deliver "$PAYLOAD3"
if env -i USER= TMPDIR="$TMP" PATH="$TMP/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
       CLAUDE_STUB_LOG="$STUB_LOG" HOME=/var/root \
       /bin/bash "$REPO/installer/scripts/postinstall" /dev/null "$PAYLOAD3" \
       >"$TMP/post4.log" 2>&1; then
  ok "postinstall exited 0 with HOME=/var/root and USER unset"
else
  nope "postinstall exited 0 with HOME=/var/root and USER unset"; cat "$TMP/post4.log"
fi
assert "installed into the temp home, not /var/root" \
  test -L "$FAKE3/Music/Ableton/User Library/Remote Scripts/AbletonLOM"
assert_eq "remote script symlink targets the temp payload" \
  "$(readlink "$FAKE3/Music/Ableton/User Library/Remote Scripts/AbletonLOM" 2>/dev/null)" \
  "$PAYLOAD3/remote_script/AbletonLOM"
assert "server installed into the temp payload" pyrun "$PAYLOAD3" -c "import mcp_server.server"
assert "nothing was written under /var/root" test ! -e /var/root/Music

rm -rf "$FAKE3"

# ------------------------------------------------------------ (g) uninstall --
echo
echo "== g. uninstall =="
: > "$STUB_LOG"   # so the remove below is the stub call being asserted
if env -i HOME="$FAKE" USER="${USER:-tester}" TMPDIR="$TMP" \
       PATH="$TMP/bin:/usr/bin:/bin:/usr/sbin:/sbin" CLAUDE_STUB_LOG="$STUB_LOG" \
       /bin/bash "$PAYLOAD/uninstall.sh" >"$TMP/uninstall.log" 2>&1; then
  ok "uninstall.sh exited 0"
else
  nope "uninstall.sh exited 0"; cat "$TMP/uninstall.log"
fi
assert "remote script symlink removed" bash -c "[ ! -e '$RS_LINK' ] && [ ! -L '$RS_LINK' ]"
assert "skill symlink removed" bash -c "[ ! -e '$SKILL_LINK' ] && [ ! -L '$SKILL_LINK' ]"
assert "payload directory removed" bash -c "[ ! -d '$PAYLOAD' ]"
assert "claude was asked to unregister" grep -qxF "mcp remove sideman -s user" "$STUB_LOG"
assert "no installer receipt is left behind" \
  bash -c "! pkgutil --pkg-info com.sideman.payload >/dev/null 2>&1"
assert "the user's Remote Scripts directory is left in place" \
  test -d "$FAKE/Music/Ableton/User Library/Remote Scripts"
assert "the user's ~/.claude/skills is left in place" test -d "$FAKE/.claude/skills"
# Only the Sideman directory goes; Application Support is the user's.
assert "the enclosing Application Support directory is left in place" \
  test -d "$FAKE/Library/Application Support"

# ------------------------------------- (h) claude registration fails --
# A corrupt ~/.claude.json makes `claude mcp add` exit non-zero. That must
# cost the user a CONFLICT.txt line, never the Remote Script or the server.
echo
echo "== h. a failing claude mcp add does not abort the install =="
HOMEH="$TMP/homeh"
PAYLOADH="$HOMEH/$REL"
RSH="$HOMEH/Music/Ableton/User Library/Remote Scripts/AbletonLOM"
mkdir -p "$HOMEH/Music/Ableton/User Library" "$HOMEH/.claude"
deliver "$PAYLOADH"
if CLAUDE_STUB_FAIL=1 run_postinstall "$HOMEH" "$PAYLOADH" >"$TMP/post_h.log" 2>&1; then
  ok "postinstall exited 0 despite the registration failure"
else
  nope "postinstall exited 0 despite the registration failure"; cat "$TMP/post_h.log"
fi
assert "CONFLICT.txt names the failed registration" \
  grep -q "Registration with Claude Code FAILED" "$PAYLOADH/CONFLICT.txt"
assert "CONFLICT.txt carries the command to run by hand" \
  grep -q "claude mcp add --scope user sideman" "$PAYLOADH/CONFLICT.txt"
assert "remote script still linked" \
  bash -c "[ -L '$RSH' ] && [ \"\$(readlink '$RSH')\" = '$PAYLOADH/remote_script/AbletonLOM' ]"
assert "skill still linked" test -L "$HOMEH/.claude/skills/sideman"
assert "server still installed" test -x "$PAYLOADH/python/bin/python3"
assert "the failure was printed" grep -q "registration: FAILED" "$TMP/post_h.log"
rm -rf "$HOMEH"

echo
if [ "$fails" -eq 0 ]; then echo "INSTALLER PASS"; else echo "INSTALLER FAILED ($fails)"; fi
exit $((fails > 0))
