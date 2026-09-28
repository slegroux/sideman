#!/usr/bin/env python3
"""Turn every Live Object Model member name in the rendered learn pages into a
link to its entry in Cycling '74's LOM reference.

A lesson is full of member names - `create_audio_track`, `"tempo"`, `fire` - and
a reader who does not recognise one has no way from the page to its definition.
The reference already has a stable anchor for every member, so the link can be
generated: the census knows which class owns which member, and the class page
slug follows from the class name. Paths are linked segment by segment, because
`live_set tracks 0 mixer_device volume` names three members on three classes.

Runs on the Pages runner as well as locally, so: stdlib only, plain `python3`.

  scripts/lom_links.py site/learn            rewrite the HTML in place
  scripts/lom_links.py site/learn --report   also print the member mapping
"""
import html
import json
import pathlib
import re
import sys
from html.parser import HTMLParser

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE = "https://docs.cycling74.com/apiref/lom/%s/#%s"

# The reference's own page list, read off https://docs.cycling74.com/apiref/lom/.
# Hard-coded rather than fetched: the renderer must work offline, and a class
# page appearing or vanishing is a census-sized event, not a silent one. Census
# types with no page here (Browser.*, Track.Routing*) are never linked.
SLUGS = {
    "application", "application_view", "chain", "chainmixerdevice", "clip",
    "clip_view", "clipslot", "compressordevice", "controlsurface", "cuepoint",
    "device", "device_view", "deviceio", "deviceparameter", "driftdevice",
    "drumcelldevice", "drumchain", "drumpad", "eq8device", "eq8device_view",
    "groove", "groovepool", "hybridreverbdevice", "looperdevice", "maxdevice",
    "melddevice", "mixerdevice", "plugindevice", "rackdevice",
    "rackdevice_view", "roardevice", "sample", "scene", "shifterdevice",
    "simplerdevice", "simplerdevice_view", "song", "song_view",
    "spectralresonatordevice", "takelane", "this_device", "track",
    "track_view", "tuningsystem", "wavetabledevice",
}

# Which class wins when a member exists on several. The head is editorial - the
# classes a lesson actually drives, most-used first - and the tail is
# alphabetical so the choice is at least stable.
PRIORITY = ["Song.Song", "Track.Track", "Clip.Clip", "ClipSlot.ClipSlot",
            "DeviceParameter.DeviceParameter", "Device.Device",
            "MixerDevice.MixerDevice", "Scene.Scene", "Song.CuePoint",
            "Song.View", "Application.View", "Clip.View", "Track.View",
            "RackDevice.RackDevice", "Chain.Chain", "DrumPad.DrumPad"]

# A member this widespread (name, color, canonical_parent, view) says nothing
# about which class is meant, so linking it to one of them would be a guess.
TOO_GENERIC = 4

# A path's first segment is a root object, not a member, and its remaining
# segments are members or zero-based indices.
ROOTS = {"live_set", "live_app", "app_view", "this_device"}
SEGMENT = re.compile(r"[A-Za-z_]\w*|\d+")

# Sideman's own wire envelope reuses one member name: every reply is
# {"value": ...}, so `"value"` in a lesson is almost always the envelope key and
# not DeviceParameter.value. Ambiguous for the same reason TOO_GENERIC is.
WIRE_KEYS = {"value"}

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}

# Frozen cell outputs are Live talking back, not API names.
SKIP_CLASSES = {"cell-output"}


def slug(type_name):
    """Census type name -> reference page slug, or None if it has no page."""
    module, _, cls = type_name.partition(".")
    s = module.lower() + "_view" if cls == "View" else cls.lower()
    return s if s in SLUGS else None


def census():
    paths = sorted((REPO / "baseline").glob("lom_census_*.json"),
                   key=lambda p: tuple(int(x) for x in
                                       p.stem[len("lom_census_"):].split(".")))
    if not paths:
        sys.exit("no census in baseline/; run scripts/census.py")
    return json.loads(paths[-1].read_text())["types"]


def member_links(types):
    """member -> page slug, plus the census types that have no page."""
    owners = {}
    for name in types:
        if slug(name):
            for member in types[name]["members"]:
                owners.setdefault(member, []).append(name)
    rank = {n: i for i, n in enumerate(PRIORITY)}
    links = {m: slug(min(o, key=lambda n: (rank.get(n, len(rank)), n)))
             for m, o in owners.items()
             if len(o) < TOO_GENERIC and m not in WIRE_KEYS}
    return links, sorted(n for n in types if not slug(n))


class Linker(HTMLParser):
    """Collects the anchors to splice into one page.

    Rewriting the text rather than re-serialising a parse tree keeps every byte
    we do not touch exactly as Quarto wrote it. Positions come from getpos().
    """

    def __init__(self, text, links):
        super().__init__(convert_charrefs=False)
        self.text, self.links = text, links
        self.lines = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
        self.stack = []
        self.depth_a = self.depth_skip = 0
        self.edits = []

    def _pos(self):
        line, col = self.getpos()
        return self.lines[line - 1] + col

    def _tag_end(self, start):
        return self.text.index(">", start) + 1

    def _soil(self):
        """Markup inside a candidate means its text is not a bare member."""
        for entry in reversed(self.stack):
            if entry["kind"]:
                entry["dirty"] = True
                return

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return self._soil()
        classes = set(dict(attrs).get("class", "").split())
        kind = None
        if not self.depth_a and not self.depth_skip:
            if tag == "code":
                kind = "code"
            elif tag == "span" and "st" in classes:
                kind = "string"
        self._soil()
        self.stack.append({"tag": tag, "kind": kind, "start": self._pos(),
                           "text": [], "dirty": False,
                           "a": tag == "a", "skip": bool(classes & SKIP_CLASSES)})
        self.depth_a += tag == "a"
        self.depth_skip += bool(classes & SKIP_CLASSES)

    def handle_startendtag(self, tag, attrs):
        self._soil()

    def handle_data(self, data):
        for entry in reversed(self.stack):
            if entry["kind"]:
                entry["text"].append(data)
                return

    def handle_entityref(self, name):
        for entry in reversed(self.stack):
            if entry["kind"]:
                entry["text"].append(html.unescape("&%s;" % name))
                return

    def handle_charref(self, name):
        self.handle_entityref("#" + name)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                break
        else:
            return
        closed = self.stack[i]
        for entry in self.stack[i:]:
            self.depth_a -= entry["a"]
            self.depth_skip -= entry["skip"]
        del self.stack[i:]
        if closed["kind"] and not closed["dirty"]:
            self._link(closed)

    def _link(self, entry):
        content = self._tag_end(entry["start"])
        close = self._pos()
        text = "".join(entry["text"])
        inner = text.strip()
        if entry["kind"] == "string":
            quoted = re.fullmatch(r"(['\"])(.*)\1", inner)
            if not quoted:
                return
            inner = quoted.group(2)
        if inner in self.links:
            self._anchor(entry["start"], self._tag_end(close), inner)
        elif text == self.text[content:close]:
            # Verbatim content, so a path's segments can be located by offset.
            self._path(content + text.index(inner), inner)

    def _path(self, base, body):
        """Link each member segment of a LOM path separately, in place.

        `live_set tracks 0 mixer_device volume` names three members and one
        index. Wrapping the whole token would have nowhere to point, so each
        segment gets its own anchor and the root and indices are left alone.
        """
        parts = body.split()
        if len(parts) < 2 or parts[0] not in ROOTS:
            return
        if not all(SEGMENT.fullmatch(p) for p in parts):
            return
        for match in list(re.finditer(r"\S+", body))[1:]:
            if match.group() in self.links:
                self._anchor(base + match.start(), base + match.end(),
                             match.group())

    def _anchor(self, start, end, member):
        # A close landing on the same offset as the next open must come first.
        page = self.links[member]
        self.edits.append(
            (start, 1, '<a class="lom" href="%s">' % (BASE % (page, member))))
        self.edits.append((end, 0, "</a>"))


def rewrite(path, links):
    text = path.read_text()
    linker = Linker(text, links)
    linker.feed(text)
    linker.close()
    if not linker.edits:
        return 0
    out, last = [], 0
    for at, _, markup in sorted(linker.edits, key=lambda e: e[:2]):
        out.append(text[last:at])
        out.append(markup)
        last = at
    out.append(text[last:])
    path.write_text("".join(out))
    return len(linker.edits) // 2


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 1:
        sys.exit("usage: lom_links.py <html-dir> [--report]")
    links, unmapped = member_links(census())
    if "--report" in argv:
        print("%d linkable members across %d reference pages"
              % (len(links), len(set(links.values()))))
        print("no reference page for %d census types: %s"
              % (len(unmapped), ", ".join(unmapped)))
    total = 0
    for path in sorted(pathlib.Path(args[0]).rglob("*.html")):
        n = rewrite(path, links)
        total += n
        if n:
            print("%s: %d LOM links" % (path, n))
    print("%d LOM links" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
