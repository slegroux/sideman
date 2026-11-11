// Sideman Listen: drains the gen~ ring buffer and sends one OSC message per
// block row to [udpsend 127.0.0.1 9071]. Runs on Max's main thread, polled by a
// [metro]; rows carry their own time stamps, so sending late is harmless, and the
// ring holds @RING@ rows (3.2 s) so a UI stall shorter than that loses nothing.
//
// Messages in:
//   bufname <name>   the per-instance buffer~ (a ---name, made unique by Live)
//   slot <int>       this device's Slot parameter (written by listen_attach)
//   bang             drain every row written since the last one sent
// Outlet 0: /sideman/listen/3 slot nonce seq playing playing_slot beat
//           hs_hi hs_lo sr n <values...>   (see sideman/stream.py)

autowatch = 0;
inlets = 1;
outlets = 1;

var W = @W@;          // floats per row in the ring
var RING = @RING@;    // rows in the ring
var HDR = 8;          // seq playing playing_slot beat hs_hi hs_lo sr n
var ADDRESS = "/sideman/listen/3";

var buf = null;
var sent = -1;        // rows sent so far (the next seq to send)
var slotv = 0;
var nonce = 0;

function newNonce() {
	nonce = 1 + Math.floor(Math.random() * 2147483646);
}
newNonce();

function bufname(name) {
	buf = new Buffer(name);
	sent = -1;
	// in the Max console: if this still reads ---sml, Live did not make the
	// name unique and every instance shares one ring
	post("Sideman Listen: ring " + name + ", nonce " + nonce + "\n");
}

function slot(v) {
	slotv = Math.floor(v);
}

function first(x) {
	return (x instanceof Array) ? x[0] : x;
}

function bang() {
	if (buf === null) {
		return;
	}
	var written = first(buf.peek(1, 0, 1));
	if (sent < 0 || written < sent) {
		// first poll, or gen~ restarted its count (sample-rate change, reload):
		// a new nonce, so the server keeps the new run in a new ring
		if (sent >= 0) {
			newNonce();
		}
		sent = written;
		return;
	}
	// never the oldest slot: gen~ overwrites it next
	var from = Math.max(sent, written - RING + 1);
	for (var s = from; s < written; s++) {
		var row = buf.peek(1, 1 + (s % RING) * W, W);
		if (first(row) != s) {
			continue;      // already overwritten by a newer row
		}
		var msg = [ADDRESS, slotv, nonce, s];
		for (var k = 1; k < W; k++) {
			msg.push(row[k]);
		}
		// msg: address slot nonce seq | playing playing_slot beat hs_hi hs_lo sr n | values
		outlet(0, msg);
	}
	sent = written;
}
