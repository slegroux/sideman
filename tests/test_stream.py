"""sideman.stream: rows from the Sideman Listen device, without Live.

The device's DSP is modelled by stream.blocks_from_array; these tests check the
codec, the ring, the receiver and the analysis against listen.py on the same
synthetic signals. Needs the `listen` extra.

    .venv/bin/python -m pytest tests/test_stream.py -q -rs -x --timeout=60
"""
import importlib.util
import json
import math
import pathlib
import re
import shutil
import socket
import sys
import tempfile
import time

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sideman import listen, stream  # noqa: E402


def sine(freq, seconds, amp=1.0, sr=48000):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def stereo(left, right=None):
    return np.stack([left, left if right is None else right], axis=1)


@pytest.fixture
def sock_path():
    # macOS caps a Unix socket path at 104 bytes; pytest's tmp_path is longer
    d = tempfile.mkdtemp(prefix="sml", dir="/tmp")
    yield pathlib.Path(d) / "l.sock"
    shutil.rmtree(d, ignore_errors=True)


def _build_module():
    spec = importlib.util.spec_from_file_location("build_amxd", ROOT / "scripts" / "build_amxd.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ codec

def test_codec_round_trips_a_row():
    rows = stream.blocks_from_array(stereo(sine(997, 1, 0.1)), 48000,
                                    state=stream.ModelState(beat0=3.25, slot=7, nonce=99,
                                                            host_sample0=5_000_000_000))
    data = stream.encode_row(rows.header[3], rows.values[3])
    addr, args = stream.osc_decode(data)
    assert addr == stream.ADDRESS and len(args) == stream.N_ARGS
    h, v = stream.decode_row(args)
    assert list(h) == list(rows.header[3])           # host_sample past 2^32 survives the split
    assert h[stream.HCOL["host_sample"]] == 5_000_000_000 + 3 * 4800
    assert np.array_equal(v, rows.values[3])


def test_codec_parses_by_type_tag_when_js_sends_integral_floats_as_ints():
    # Max's js emits 0.0 as an int: a silent block arrives with `i` tags
    args = [1, 2, 3, 1, -1, 0, 0, 0, 48000, 4800] + [0] * stream.N_VALUES
    addr, got = stream.osc_decode(stream.osc_encode(stream.ADDRESS, args))
    assert got == args
    h, v = stream.decode_row(got)
    assert h[stream.HCOL["playing_slot"]] == -1 and v.dtype == np.float32 and not v.any()


def test_codec_refuses_a_short_row():
    with pytest.raises(ValueError):
        stream.decode_row([1, 2, 3])


def test_receiver_counts_bad_packets_instead_of_raising(sock_path):
    r = stream.Receiver(port=0, sock_path=sock_path)
    r.feed(b"garbage")
    r.feed(stream.osc_encode("/other", [1]))
    assert r.bad_packets == 2 and r.rings == {}


# ------------------------------------------------------------------ ring

def test_ring_keeps_the_newest_rows_in_order_and_counts_seq_gaps():
    rows = stream.blocks_from_array(stereo(sine(997, 2, 0.1)), 48000)   # 20 rows
    ring = stream.Ring(capacity=8)
    for i in list(range(10)) + list(range(13, 20)):                  # 3 never arrive
        ring.push(rows.header[i], rows.values[i])
    got = ring.rows()
    assert list(got.h("seq")) == [9, 13, 14, 15, 16, 17, 18, 19]
    assert ring.seq_gaps == 3
    assert list(ring.rows(3).h("seq")) == [17, 18, 19]


def test_ring_header_is_int64_so_the_sample_count_stays_exact():
    ring = stream.Ring(capacity=4)
    h = np.zeros(len(stream.HEADER), np.int64)
    h[stream.HCOL["host_sample"]] = 2 ** 40 + 1
    ring.push(h, np.zeros(stream.N_VALUES, np.float32))
    assert ring.rows().h("host_sample")[0] == 2 ** 40 + 1


# ------------------------------------------------------------------ analysis

@pytest.mark.parametrize("sr", [44100, 48000])
def test_snapshot_of_the_minus_20_lufs_fixture(sr):
    # test_listen.py reads this stereo 997 Hz sine at amplitude 0.1 as -20.0 LUFS
    x = stereo(sine(997, 10, 0.1, sr=sr))
    rows = stream.blocks_from_array(x, sr)
    a = stream.analyze_rows(rows[-30:])
    assert a["momentary_lufs"] == pytest.approx(-20.0, abs=0.1)
    assert a["short_term_lufs"] == pytest.approx(-20.0, abs=0.1)
    f = listen.analyze_array(x, sr)
    whole = stream.analyze_rows(rows)
    assert whole["integrated_lufs"] == pytest.approx(f["integrated_lufs"], abs=0.05)
    assert whole["sample_peak_db"] == pytest.approx(f["sample_peak_db"], abs=0.05)
    assert whole["rms_db"] == pytest.approx(f["rms_db"], abs=0.05)
    assert whole["crest_db"] == pytest.approx(f["crest_db"], abs=0.1)
    assert whole["channels"] == 2


def test_snapshot_rows_are_the_same_sub_blocks_listen_gates():
    x = stereo(sine(997, 6, 0.1), sine(440, 6, 0.05))
    rows = stream.blocks_from_array(x, 48000)
    sub_file = listen._subblock_powers(x, 48000)
    sub_rows = rows.v("k_pow_l").astype(np.float64) + rows.v("k_pow_r")
    assert len(sub_rows) == len(sub_file)
    assert np.allclose(sub_rows, sub_file, rtol=1e-4)


def test_snapshot_of_silence_is_unmeasurable_not_a_number():
    rows = stream.blocks_from_array(np.zeros((48000, 2)), 48000)
    a = stream.analyze_rows(rows)
    assert a["momentary_lufs"] is None and a["integrated_lufs"] is None
    assert a["crest_db"] is None


def test_snapshot_of_no_rows():
    a = stream.analyze_rows(stream.empty_rows())
    assert a["blocks"] == 0 and a["momentary_lufs"] is None


# ------------------------------------------------------------------ receiver over UDP

def _send_rows(port, rows, step=None):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for i in range(len(rows)):
            s.sendto(stream.encode_row(rows.header[i], rows.values[i]), ("127.0.0.1", port))
    finally:
        s.close()


def _wait(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_snapshot_through_a_real_udp_receiver_on_loopback(sock_path):
    r = stream.Receiver(port=0, sock_path=sock_path)
    assert r.start() == "owner"
    try:
        assert r.address()[0] == "127.0.0.1"            # never bound beyond loopback
        rows = stream.blocks_from_array(stereo(sine(997, 4, 0.1)), 48000,
                                        state=stream.ModelState(slot=5))
        _send_rows(r.address()[1], rows)
        assert _wait(lambda: len(r.rows(5)) == len(rows))
        info = r.slots()[5]
        assert info["sr"] == 48000 and len(info["nonces"]) == 1 and info["seq_gaps"] == 0
        a = stream.analyze_rows(r.rows(5, last_s=3.0))
        assert a["blocks"] == 30
        assert a["short_term_lufs"] == pytest.approx(-20.0, abs=0.1)
    finally:
        r.stop()


def test_two_devices_on_one_slot_show_two_nonces(sock_path):
    r = stream.Receiver(port=0, sock_path=sock_path)
    rows_a = stream.blocks_from_array(stereo(sine(997, 1, 0.1)), 48000,
                                      state=stream.ModelState(slot=3, nonce=1))
    rows_b = stream.blocks_from_array(stereo(sine(997, 1, 0.1)), 48000,
                                      state=stream.ModelState(slot=3, nonce=2))
    for rows in (rows_a, rows_b):
        for i in range(len(rows)):
            r.feed(stream.encode_row(rows.header[i], rows.values[i]))
    assert sorted(r.slots()[3]["nonces"]) == [1, 2]


# ------------------------------------------------------------------ share socket (item 2)

def test_share_a_second_receiver_reads_the_owners_rows(sock_path):
    owner = stream.Receiver(port=0, sock_path=sock_path)
    assert owner.start() == "owner"
    client = stream.Receiver(port=owner.port, sock_path=sock_path)
    try:
        assert client.start() == "client"
        rows = stream.blocks_from_array(stereo(sine(997, 2, 0.1)), 48000,
                                        state=stream.ModelState(slot=9, host_sample0=2 ** 33))
        _send_rows(owner.port, rows)
        assert _wait(lambda: len(owner.rows(9)) == len(rows))
        got, mine = client.rows(9), owner.rows(9)
        assert np.array_equal(got.header, mine.header)
        assert np.array_equal(got.values, mine.values)
        assert np.array_equal(client.rows(9, last_s=0.5).header, owner.rows(9, last_s=0.5).header)
        def without_age(d):
            return {k: {kk: vv for kk, vv in v.items() if kk != "last_seen_s"} for k, v in d.items()}
        assert without_age(client.slots()) == without_age(owner.slots())
        assert set(client.slots()[9]) == {"nonces", "last_seen_s", "sr", "seq_gaps", "seq_rewinds"}
        assert oct(sock_path.stat().st_mode & 0o777) == "0o600"
    finally:
        client.stop()
        owner.stop()
    assert not sock_path.exists()                          # the owner removes its socket


def test_share_a_leftover_socket_file_is_replaced_by_the_next_owner(sock_path):
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(sock_path))                               # a crashed owner's socket
    dead.close()
    assert sock_path.exists()
    owner = stream.Receiver(port=0, sock_path=sock_path)
    try:
        assert owner.start() == "owner"
        client = stream.Receiver(port=owner.port, sock_path=sock_path)
        assert client.start() == "client"
        client.stop()
    finally:
        owner.stop()


def test_share_refuses_a_protocol_mismatch(sock_path):
    owner = stream.Receiver(port=0, sock_path=sock_path)
    owner.protocol = 99
    owner.start()
    try:
        with pytest.raises(stream.PortInUse, match="protocol"):
            stream.Receiver(port=owner.port, sock_path=sock_path).start()
    finally:
        owner.stop()


def test_share_port_held_by_a_plain_socket_is_not_a_sideman_listener(sock_path, monkeypatch):
    monkeypatch.setattr(stream, "CONNECT_RETRY_S", 0.2)
    plain = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    plain.bind(("127.0.0.1", 0))
    try:
        with pytest.raises(stream.PortInUse, match="not a Sideman listener"):
            stream.Receiver(port=plain.getsockname()[1], sock_path=sock_path).start()
    finally:
        plain.close()


def test_share_client_takes_over_when_the_owner_exits(sock_path):
    owner = stream.Receiver(port=0, sock_path=sock_path)
    owner.start()
    client = stream.Receiver(port=owner.port, sock_path=sock_path)
    try:
        assert client.start() == "client"
        owner.stop()
        assert client.rows(1).header.shape[0] == 0          # history is gone ...
        assert client.role == "owner"                       # ... and this process owns it now
        rows = stream.blocks_from_array(stereo(sine(997, 1, 0.1)), 48000,
                                        state=stream.ModelState(slot=1))
        _send_rows(client.port, rows)
        assert _wait(lambda: len(client.rows(1)) == len(rows))
    finally:
        client.stop()
        owner.stop()


# ------------------------------------------------------------------ the device build

def test_device_builds_and_matches_the_stream_layout(tmp_path, monkeypatch):
    b = _build_module()
    assert b.N_VALUES == stream.N_VALUES
    assert b.W == 8 + stream.N_VALUES and b.PORT == stream.DEFAULT_PORT
    code = b.genexpr()
    assert "@" not in code.replace("@ ", "")          # every placeholder filled
    js = b.js()
    assert "var W = %d;" % b.W in js and "var RING = %d;" % b.RING in js
    assert stream.ADDRESS in js
    monkeypatch.setattr(b, "ROOT", tmp_path)
    amxd, jsf = b.build()
    blob = amxd.read_bytes()
    assert blob[:4] == b"ampf" and blob[8:12] == b"aaaa"
    p = b.unwrap_amxd(blob)["patcher"]
    texts = [x["box"].get("text", "") for x in p["boxes"]]
    assert "plugin~" in texts and "plugout~" in texts           # audio passes through
    assert "js listen.js" in texts                               # bare name, found next to the device
    assert "udpsend 127.0.0.1 %d" % stream.DEFAULT_PORT in texts
    assert any(t.startswith("buffer~ ---sml @samps %d" % (1 + b.RING * b.W)) for t in texts)
    gen = next(x["box"] for x in p["boxes"] if x["box"].get("text") == "gen~")
    assert gen["patcher"]["classnamespace"] == "dsp.gen"
    lines = {(ln["patchline"]["source"][0], ln["patchline"]["destination"][0])
             for ln in p["lines"]}
    assert ("in", "out") in lines and ("gen", "out") in lines
    assert jsf.read_text() == js


def test_kill_test_variant_adds_the_dummy_load():
    b = _build_module()
    assert b.genexpr(0).count("History dl") == 0 and "firl =" not in b.genexpr(0)
    kt = b.genexpr(180)                          # 180 biquads per instance, 90 per channel
    assert kt.count("History dl") == 90 and kt.count("History dr") == 90
    assert kt.count(" * fl") == 48 and kt.count(" * fr") == 48     # a 48-tap FIR per channel
    assert "- -" not in kt


def test_user_library_is_read_from_library_cfg(tmp_path, monkeypatch):
    b = _build_module()
    lib = tmp_path / "Music" / "User Library"
    lib.mkdir(parents=True)
    cfg = tmp_path / "Library" / "Preferences" / "Ableton" / "Live 12.4.6"
    cfg.mkdir(parents=True)
    (cfg / "Library.cfg").write_text(
        '<Ableton><UserLibrary><LibraryProject Id="0"><ProjectLocation />'
        '<ProjectName Value="User Library" /><ProjectPath Value="%s" />'
        '</LibraryProject></UserLibrary></Ableton>' % (tmp_path / "Music"))
    monkeypatch.setattr(b.pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    assert b.user_library() == lib
    assert b.browser_path() == ("user_library/Presets/Audio Effects/Max Audio Effect/"
                                "Sideman Listen.amxd")


def test_wire_header_count_matches_js():
    # js sends: address slot nonce seq + row[1..W-1]; stream expects N_ARGS
    b = _build_module()
    assert 3 + (b.W - 1) == stream.N_ARGS


@pytest.mark.parametrize("sr", [44100, 48000, 96000])
def test_device_k_weighting_coefficients_equal_listen_k_weighting(sr):
    """Run the gen~ code's sample-rate block as Python and compare."""
    code = _build_module().genexpr()
    block = code[code.index("if (cur_sr != samplerate) {"):code.index("nblk = round")]
    env = {"samplerate": float(sr), "tan": math.tan, "pow": math.pow, "pi": math.pi}
    for line in block.splitlines()[1:]:
        line = line.split("//")[0].strip().rstrip(";")
        if line:
            exec(line, env)
    (b1, a1), (_, a2) = listen.k_weighting(sr)
    assert np.allclose([env["s1b0"], env["s1b1"], env["s1b2"]], b1, rtol=0, atol=1e-12)
    assert np.allclose([env["s1a1"], env["s1a2"]], a1[1:], rtol=0, atol=1e-12)
    assert np.allclose([env["s2a1"], env["s2a2"]], a2[1:], rtol=0, atol=1e-12)


def test_device_row_offsets_follow_the_stream_layout():
    """Every poke(ring, <value>, base + N) in the gen~ code, in N order, is the
    column stream.py decodes at that position."""
    code = _build_module().genexpr()
    pokes = dict((int(n), expr.strip()) for expr, n in
                 re.findall(r"poke\(ring, (.+?), base \+ (\d+)\);", code))
    assert sorted(pokes) == list(range(len(pokes)))
    expected = ["seq", "t_play", "t_slot", "t_beat", "hi", "t_hs - hi * 65536", "samplerate", "c",
                "s_ll / c", "s_rr / c", "s_lr / c", "s_kl / c", "s_kr / c", "s_pk", "0", "s_dm / c"]
    assert [pokes[i] for i in range(len(expected))] == expected
    # wire header order (after slot, nonce): seq playing playing_slot beat hs_hi hs_lo sr n
    assert stream.HEADER[2:5] == ("seq", "playing", "playing_slot")
    names = [n for n, _ in stream.VALUES]
    assert names[:7] == ["pow_l", "pow_r", "pow_lr", "k_pow_l", "k_pow_r", "peak", "true_peak"]
    assert stream.COL["third_oct"].start == 15 - 8      # the dummy load's column


def test_listen_snapshot_tool_reports_stale_ambiguous_and_unheard(sock_path, monkeypatch):
    import mcp_server.server as S
    r = stream.Receiver(port=0, sock_path=sock_path)
    monkeypatch.setattr(S, "_receiver", r)
    now = time.monotonic()
    x = stereo(sine(997, 4, 0.1))
    for slot, nonce, t in ((1, 10, now), (1, 11, now), (2, 20, now - 5.0)):
        rows = stream.blocks_from_array(x, 48000, state=stream.ModelState(slot=slot, nonce=nonce))
        for i in range(len(rows)):
            r.feed(stream.encode_row(rows.header[i], rows.values[i]), now=t)
    out = json.loads(S.listen_snapshot(seconds=3.0, slots=[1, 2, 3]))
    assert out["ambiguous"] == [1] and out["stale"] == [2] and out["unheard"] == [3]
    assert out["slots"]["1"]["blocks"] == 30
    assert out["slots"]["1"]["short_term_lufs"] == pytest.approx(-20.0, abs=0.1)


def test_feed_survives_rows_that_overflow():
    r = stream.Receiver(port=0)
    args = [1, 2, 3, 1, -1, float("inf"), 0, 0, 48000, 4800] + [0.0] * stream.N_VALUES
    r.feed(stream.osc_encode(stream.ADDRESS, args))
    assert r.bad_packets == 1 and r.rings == {}


def test_rings_are_evicted_per_slot_and_when_idle():
    r = stream.Receiver(port=0)
    rows = stream.blocks_from_array(stereo(sine(997, 0.2, 0.1)), 48000)
    def put(slot, nonce, t):
        r.feed(stream.encode_row(np.r_[[slot, nonce], rows.header[0][2:]].astype(np.int64),
                                 rows.values[0]), now=t)
    for nonce in range(1, 6):                       # five reloads of one device
        put(4, nonce, 100.0 + nonce)
    assert sorted(k[1] for k in r.rings if k[0] == 4) == [3, 4, 5]
    put(5, 1, 100.0 + stream.RING_IDLE_S + 50)      # much later: slot 4 is idle
    assert all(k[0] == 5 for k in r.rings)


def test_a_seq_rewind_under_one_nonce_is_counted_not_hidden():
    ring = stream.Ring(capacity=8)
    rows = stream.blocks_from_array(stereo(sine(997, 0.5, 0.1)), 48000)
    for i in (0, 1, 2, 0, 1):
        ring.push(rows.header[i], rows.values[i])
    assert ring.seq_rewinds == 1 and ring.seq_gaps == 0


def test_stop_leaves_a_successors_socket_file_alone(sock_path):
    owner = stream.Receiver(port=0, sock_path=sock_path)
    owner.start()
    successor_ino = None
    try:
        # simulate a successor binding a new socket file at the same path
        sock_path.unlink()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(sock_path))
        successor_ino = sock_path.stat().st_ino
        owner.stop()
        assert sock_path.exists() and sock_path.stat().st_ino == successor_ino
    finally:
        owner.stop()
        s.close()
