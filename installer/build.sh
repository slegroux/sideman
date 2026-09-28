#!/usr/bin/env bash
# Build dist/Sideman-<version>.pkg - a per-user macOS installer for musicians.
#
#   installer/build.sh
#
# The .pkg is unsigned. Signing needs a Developer ID Installer certificate and
# notarization needs an Apple Developer account and `notarytool`; both are human
# steps outside this script. Until then macOS Gatekeeper requires the user to
# right-click > Open the .pkg once instead of double-clicking it - and on macOS
# 15+ that is often not offered either, so the user has to open it once, then go
# to System Settings > Privacy & Security and click "Open Anyway".
#
# Everything downloaded lands in installer/.cache (gitignored). The archives are
# ~27 MB each and must never be committed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALLER="$REPO/installer"
CACHE="$INSTALLER/.cache"
STAGE="$CACHE/build/payload"     # becomes ~/Library/Application Support/Sideman
COMPONENT="$CACHE/build/sideman-component.pkg"
DIST="$REPO/dist"

IDENTIFIER="com.sideman.payload"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$REPO/pyproject.toml" | head -1)"
[ -n "$VERSION" ] || { echo "cannot read version from pyproject.toml" >&2; exit 1; }

# A relocatable CPython from python-build-standalone: musicians have no uv and
# no modern python3. Pinned by tag and checksum so a rebuild is reproducible -
# the pkg ships one frozen interpreter, not whatever is current.
PBS_TAG=20260924
PBS_VER=3.11.16
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG"
SHA_aarch64=d718e3c5c6f4b225ed25f88bf65e4c5d314e0dea0d716ea50bc9d038630c502b
SHA_x86_64=a93ee2dd8f2dbddbd85a0c9f1739f98d6ead305cbf445918a2fe28deb77f2d4b
# Lowest deployment target each PBS build accepts, so a binary wheel downloaded
# here is acceptable to pip on any Mac of that architecture.
PLAT_aarch64=macosx_11_0_arm64
PLAT_x86_64=macosx_10_12_x86_64

mkdir -p "$CACHE" "$DIST"

# Two builds share $CACHE/build, and the loser's staging tree is deleted out
# from under it mid-run - which surfaces as a pkg that builds "fine" with an
# almost empty payload. mkdir is the atomic primitive macOS gives us; flock is
# not available. Serialise rather than interleave.
LOCK="$CACHE/build.lock"
locked=
for _ in $(seq 1 300); do
  if mkdir "$LOCK" 2>/dev/null; then locked=1; break; fi
  echo "waiting for another build to release $LOCK"
  sleep 2
done
[ -n "$locked" ] || { echo "timed out waiting for $LOCK - remove it if stale" >&2; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# --- 1. fetch and verify the interpreters ----------------------------------
for arch in aarch64 x86_64; do
  tarball="$CACHE/cpython-$PBS_VER+$PBS_TAG-$arch-apple-darwin-install_only.tar.gz"
  if [ ! -f "$tarball" ]; then
    echo "downloading $(basename "$tarball")"
    curl -fsSL -o "$tarball" "$PBS_BASE/$(basename "$tarball")"
  fi
  expected="SHA_$arch"
  actual="$(shasum -a 256 "$tarball" | cut -d' ' -f1)"
  [ "$actual" = "${!expected}" ] || {
    echo "checksum mismatch for $(basename "$tarball")" >&2; exit 1; }
done

# --- 2. stage the payload tree ---------------------------------------------
rm -rf "$STAGE"
mkdir -p "$STAGE"
for src in remote_script/AbletonLOM mcp_server sideman skills/sideman; do
  mkdir -p "$STAGE/$(dirname "$src")"
  rsync -a --exclude __pycache__ "$REPO/$src/" "$STAGE/$src/"
done
cp "$REPO/pyproject.toml" "$REPO/LICENSE" "$STAGE/"
install -m 755 "$INSTALLER/uninstall.sh" "$STAGE/uninstall.sh"

# Both architectures ship; postinstall keeps the matching one and prunes the
# other. Cheaper than two .pkg downloads a musician has to choose between.
for arch in aarch64 x86_64; do
  tar -xzf "$CACHE/cpython-$PBS_VER+$PBS_TAG-$arch-apple-darwin-install_only.tar.gz" \
      -C "$CACHE/build"
  mv "$CACHE/build/python" "$STAGE/python-$arch-apple-darwin"
done

# --- 3. freeze the dependency resolution into wheels/ ----------------------
# Installing offline is the point: an installer that needs PyPI fails in the
# one place we cannot debug it. Both architectures' binary wheels go in the
# same directory - pip picks the compatible ones by tag at install time.
# Pinned, not ">=": an unpinned resolution pulls a different mcp and its ~30
# transitive wheels on every cold build, with nothing in review to diff. The
# pin is read from the dev venv rather than written here, so the mcp a musician
# gets is the one the suite was last run against - a hardcoded number drifts
# away from the tree the moment someone upgrades their venv. The wheel cache is
# keyed to the version so a bump re-resolves instead of reusing.
DEV_PY="$REPO/.venv/bin/python"
[ -x "$DEV_PY" ] || { echo "no dev venv at $DEV_PY - see README Install" >&2; exit 1; }
MCP_VER="$("$DEV_PY" -c 'import importlib.metadata as m; print(m.version("mcp"))')" || {
  echo "cannot read the mcp version from $DEV_PY" >&2; exit 1; }
[ -n "$MCP_VER" ] || { echo "empty mcp version from $DEV_PY" >&2; exit 1; }
echo "pinning mcp==$MCP_VER (from the dev venv)"
# A throwaway copy of the same interpreter does the downloading and building.
# Using a staged one leaves ~9 MB of its own .pyc in the tree we ship, so the
# two runtimes in the payload would stop being the pristine upstream builds -
# and the arm64 user would get a different one from the x86_64 user.
BUILDER_DIR="$CACHE/builder"
if [ ! -x "$BUILDER_DIR/python/bin/python3" ]; then
  rm -rf "$BUILDER_DIR"; mkdir -p "$BUILDER_DIR"
  tar -xzf "$CACHE/cpython-$PBS_VER+$PBS_TAG-aarch64-apple-darwin-install_only.tar.gz" \
      -C "$BUILDER_DIR"
fi
BUILDER="$BUILDER_DIR/python/bin/python3"
WHEELS="$CACHE/wheels-mcp-$MCP_VER"   # third-party wheels only, cached across builds
if [ ! -d "$WHEELS" ]; then
  mkdir -p "$WHEELS"
  for arch in aarch64 x86_64; do
    plat="PLAT_$arch"
    "$BUILDER" -m pip download --quiet --dest "$WHEELS" \
      --only-binary=:all: --python-version 3.11 --platform "${!plat}" \
      "mcp==$MCP_VER"
  done
fi
rsync -a "$WHEELS/" "$STAGE/wheels/"

# sideman itself, built here so install time needs no build backend. Never
# cached: a cached wheel would ship whatever the tree looked like on some
# earlier build. The packages it contains follow pyproject.toml, so a new
# top-level package is picked up without touching this script.
rm -f "$STAGE"/wheels/sideman-*.whl
"$BUILDER" -m pip wheel --quiet --no-deps --wheel-dir "$STAGE/wheels" "$REPO"

# --- 4. component then product ---------------------------------------------
# A *relative* --install-location plus the distribution's
# enable_currentUserHome is what makes this a no-admin-password install.
#
# Built on a developer Mac, half the manifest below is 0-byte AppleDouble (._*)
# entries: macOS 14+ stamps com.apple.provenance on files a GUI-launched process
# creates, xattr cannot remove it, and pkgbuild encodes every xattr. Harmless to
# install. A CI runner carries no provenance and packs a clean manifest, which
# is where release packages should come from.
rm -f "$COMPONENT"
pkgbuild --quiet \
  --root "$STAGE" \
  --scripts "$INSTALLER/scripts" \
  --identifier "$IDENTIFIER" \
  --version "$VERSION" \
  --install-location "Library/Application Support/Sideman" \
  "$COMPONENT"

# pkgbuild stamps every component auth="root", which makes Installer ask for an
# admin password even though nothing leaves the user's home. There is no
# pkgbuild flag for it, so rewrite it in the expanded component.
EXPANDED="$CACHE/build/component-expanded"
rm -rf "$EXPANDED"
pkgutil --expand "$COMPONENT" "$EXPANDED"
sed -i '' 's/ auth="root"/ auth="none"/' "$EXPANDED/PackageInfo"
rm -f "$COMPONENT"
pkgutil --flatten "$EXPANDED" "$COMPONENT"
rm -rf "$EXPANDED"          # a second copy of the whole payload

sed -e "s/@VERSION@/$VERSION/g" -e "s/@IDENTIFIER@/$IDENTIFIER/g" \
    -e "s/@COMPONENT@/$(basename "$COMPONENT")/g" \
    "$INSTALLER/distribution.xml" > "$CACHE/build/distribution.xml"

PKG="$DIST/Sideman-$VERSION.pkg"
rm -f "$PKG"
productbuild \
  --distribution "$CACHE/build/distribution.xml" \
  --resources "$INSTALLER/resources" \
  --package-path "$(dirname "$COMPONENT")" \
  "$PKG"

rm -f "$COMPONENT"          # productbuild has copied it into $PKG

echo "$PKG"
du -h "$PKG" | cut -f1
