An Ableton Live MCP server with complete Live Object Model coverage — 47 types / 922 members on Live 12.2.7, measured against a running Live.

**Install (macOS, any Live 12 edition).** Download the `.pkg`, right-click → **Open** (it is not notarized yet), and follow the last screen. It installs into your home folder, needs no password, and brings its own Python. The one manual step: Live → Settings → Link, Tempo & MIDI → Control Surface → **AbletonLOM**. Then restart your Claude session and ask "What's in my Set?".

**What it does that other Ableton MCP servers don't**

- Clip automation envelopes — write a filter sweep as automation, not just set a knob
- Observers — it sees changes you make by hand in Live's GUI
- Every Live 12 edition — a Remote Script, no Max for Live, no Suite requirement
- Real undo steps; destructive calls guarded behind `confirm=true`

**Requirements:** macOS 12+, Ableton Live 12, Claude Code or any MCP client.

**Known limits:** unsigned package (right-click → Open once); VST/AU parameters need Live's Configure button; measured on Live 12.2.7.

Docs: https://slegroux.github.io/sideman/ · https://github.com/slegroux/sideman#readme
