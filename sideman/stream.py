"""Real-time listening: rows from the Sideman Listen Max for Live device.

One "Sideman Listen" device sits last on a track. Every 100 ms block it sends
one OSC message over UDP to 127.0.0.1 with the block's sums (powers, peaks,
band energies), stamped with what Live's transport said at the block's start.
This module receives those rows, keeps the last 15 minutes per device, and turns
a span of them into the measurements `listen.analyze_array` gives for a file.

Two halves that must agree:
- the device (`sideman/m4l/listen.genexpr`, built by `scripts/build_amxd.py`)
- `blocks_from_array`, a numpy model of the same DSP that the tests use as the
  oracle: feed it a file and you get the rows the device would send.

Needs the `listen` extra (numpy). Local only: nothing here opens a socket to
anything but 127.0.0.1.
"""
from __future__ import annotations

import atexit
import errno
import hashlib
import json
import math
import os
import pathlib
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

from sideman import listen

ADDRESS = "/sideman/listen/3"
PROTOCOL = 3
DEFAULT_PORT = 9071
HOST = "127.0.0.1"
RING_ROWS = 9000                     # 15 minutes of 100 ms blocks
BLOCK_S = listen.SUBBLOCK_S          # one row = one BS.1770 sub-block
STALE_S = 1.0                        # a device silent this long is stale

# Header: integers, kept in int64. host_sample is sent as hi/lo halves of 16
# bits each so that no value leaves the exact range of the device's float32
# ring buffer; the codec joins them.
HEADER = ("slot", "nonce", "seq", "playing", "playing_slot", "beat_x1e6",
          "host_sample", "sr", "n")
# Values: float32, in packet order.
VALUES = (("pow_l", 1), ("pow_r", 1), ("pow_lr", 1), ("k_pow_l", 1), ("k_pow_r", 1),
          ("peak", 1), ("true_peak", 1), ("third_oct", 29), ("band6", 18), ("lo", 3),
          ("lo_sub", 20))
N_VALUES = sum(w for _, w in VALUES)  # 77
# The wire header: slot nonce seq playing playing_slot beat hs_hi hs_lo sr n
N_WIRE_HEADER = 10
N_ARGS = N_WIRE_HEADER + N_VALUES    # 87


def _value_slices() -> dict[str, slice]:
    out, i = {}, 0
    for name, w in VALUES:
        out[name] = slice(i, i + w)
        i += w
    return out


COL = _value_slices()
HCOL = {name: i for i, name in enumerate(HEADER)}


# ------------------------------------------------------------------ OSC codec

def _pad4(n: int) -> int:
    return (n + 4) & ~3


def _read_str(data: bytes, i: int) -> tuple[str, int]:
    end = data.index(b"\x00", i)
    return data[i:end].decode("ascii"), i + _pad4(end - i)


def osc_decode(data: bytes) -> tuple[str, list]:
    """Address and arguments of one OSC message. Types are taken from the type
    tag string, never assumed: Max's js emits a float with no fractional part as
    an int, so a silent block's 0.0 may arrive as `i`."""
    addr, i = _read_str(data, 0)
    tags, i = _read_str(data, i)
    if not tags.startswith(","):
        raise ValueError("no type tag string")
    args: list = []
    for t in tags[1:]:
        if t == "i":
            args.append(struct.unpack(">i", data[i:i + 4])[0])
            i += 4
        elif t == "f":
            args.append(struct.unpack(">f", data[i:i + 4])[0])
            i += 4
        elif t == "h":
            args.append(struct.unpack(">q", data[i:i + 8])[0])
            i += 8
        elif t == "d":
            args.append(struct.unpack(">d", data[i:i + 8])[0])
            i += 8
        elif t == "s":
            s, i = _read_str(data, i)
            args.append(s)
        else:
            raise ValueError("unsupported OSC type %r" % t)
    return addr, args


def osc_encode(addr: str, args: list) -> bytes:
    """The inverse, for tests and for the fake device. Python ints become `i`,
    floats `f`, as Max's udpsend does."""
    def s(x: str) -> bytes:
        b = x.encode("ascii")
        return b + b"\x00" * (_pad4(len(b)) - len(b))
    tags, body = ",", b""
    for a in args:
        if isinstance(a, (bool, np.bool_)):
            a = int(a)
        if isinstance(a, (int, np.integer)):
            tags += "i"
            body += struct.pack(">i", int(a))
        else:
            tags += "f"
            body += struct.pack(">f", float(a))
    return s(addr) + s(tags) + body


def decode_row(args: list) -> tuple[np.ndarray, np.ndarray]:
    """(header int64 [9], values float32 [77]) from one message's arguments."""
    if len(args) != N_ARGS:
        raise ValueError("expected %d arguments, got %d" % (N_ARGS, len(args)))
    slot, nonce, seq, playing, pslot, beat, hs_hi, hs_lo, sr, n = args[:N_WIRE_HEADER]
    h = np.array([int(slot), int(nonce), int(seq), int(playing), int(pslot),
                  int(round(float(beat) * 1e6)), int(hs_hi) * 65536 + int(hs_lo),
                  int(sr), int(n)], dtype=np.int64)
    v = np.asarray(args[N_WIRE_HEADER:], dtype=np.float32)
    return h, v


def encode_row(h: np.ndarray, v: np.ndarray) -> bytes:
    """One row as the device's message (the fake device and the tests)."""
    hs = int(h[HCOL["host_sample"]])
    args = [int(h[HCOL["slot"]]), int(h[HCOL["nonce"]]), int(h[HCOL["seq"]]),
            int(h[HCOL["playing"]]), int(h[HCOL["playing_slot"]]),
            float(h[HCOL["beat_x1e6"]]) / 1e6, hs // 65536, hs % 65536,
            int(h[HCOL["sr"]]), int(h[HCOL["n"]])]
    args += [float(x) for x in v]
    return osc_encode(ADDRESS, args)


# ------------------------------------------------------------------ rows

@dataclass
class Rows:
    """A span of rows from one device, oldest first."""
    header: np.ndarray          # int64 [m, len(HEADER)]
    values: np.ndarray          # float32 [m, N_VALUES]

    def __len__(self) -> int:
        return self.header.shape[0]

    def h(self, name: str) -> np.ndarray:
        return self.header[:, HCOL[name]]

    def v(self, name: str) -> np.ndarray:
        s = COL[name]
        col = self.values[:, s]
        return col[:, 0] if s.stop - s.start == 1 else col

    def __getitem__(self, idx) -> "Rows":
        return Rows(self.header[idx], self.values[idx])

    @property
    def beat(self) -> np.ndarray:
        return self.h("beat_x1e6") / 1e6


def empty_rows() -> Rows:
    return Rows(np.zeros((0, len(HEADER)), np.int64), np.zeros((0, N_VALUES), np.float32))


class Ring:
    """The last RING_ROWS rows of one device instance, keyed by (slot, nonce)
    in the Receiver. Header columns are int64 so the sample count and the nonce
    stay exact; values are float32 as the device sends them."""

    def __init__(self, capacity: int = RING_ROWS):
        self.cap = capacity
        self.header = np.zeros((capacity, len(HEADER)), np.int64)
        self.values = np.zeros((capacity, N_VALUES), np.float32)
        self.count = 0                       # rows ever written
        self.last_seen = 0.0                 # time.monotonic() of the last row
        self.last_seq = -1
        self.seq_gaps = 0                    # rows the device numbered but never arrived
        self.seq_rewinds = 0                 # seq went back under one nonce (should not happen)
        self.lock = threading.Lock()

    def push(self, h: np.ndarray, v: np.ndarray, now: float | None = None) -> None:
        with self.lock:
            seq = int(h[HCOL["seq"]])
            # The device draws a new nonce whenever its count restarts, so a
            # seq at or below the last one under the same nonce is an anomaly:
            # counted, never treated as a gap.
            if self.last_seq >= 0 and seq > self.last_seq + 1:
                self.seq_gaps += seq - self.last_seq - 1
            elif self.last_seq >= 0 and seq <= self.last_seq:
                self.seq_rewinds += 1
            self.last_seq = seq
            i = self.count % self.cap
            self.header[i] = h
            self.values[i] = v
            self.count += 1
            self.last_seen = time.monotonic() if now is None else now

    def rows(self, last: int | None = None) -> Rows:
        """The newest `last` rows (all held, by default), oldest first."""
        with self.lock:
            held = min(self.count, self.cap)
            k = held if last is None else max(0, min(last, held))
            if k == 0:
                return empty_rows()
            end = self.count % self.cap
            idx = (np.arange(end - k, end) % self.cap)
            return Rows(self.header[idx].copy(), self.values[idx].copy())


# ------------------------------------------------------------------ receiver

class PortInUse(RuntimeError):
    pass


def default_sock_path() -> pathlib.Path:
    return pathlib.Path.home() / "Library" / "Application Support" / "Sideman" / "listen.sock"


FIELDS_HASH = hashlib.sha1(repr((HEADER, VALUES)).encode()).hexdigest()[:12]
CONNECT_RETRY_S = 1.0
MAX_FRAME = 64 << 20                 # a share frame larger than this is refused
RING_IDLE_S = RING_ROWS * BLOCK_S    # a ring silent this long is dropped
MAX_RINGS_PER_SLOT = 3
MAX_RINGS = 600


def _send_frame(conn: socket.socket, payload: bytes) -> None:
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("share socket closed")
        buf += chunk
    return bytes(buf)


def _recv_frame(conn: socket.socket) -> bytes:
    (n,) = struct.unpack(">I", _recv_exact(conn, 4))
    if n > MAX_FRAME:
        raise ConnectionError("share frame of %d bytes refused" % n)
    return _recv_exact(conn, n)


def _shut(sock: socket.socket | None) -> None:
    """Shut down, then close: on Linux a bare close() does not wake a thread
    blocked in recv() on the same socket."""
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class Receiver:
    """Device rows for this process, as the owner of the stream or its client.

    The UDP bind on 127.0.0.1:port is the lock. The process that wins it owns
    the stream: it receives rows in a daemon thread, and serves them to other
    Sideman servers on a Unix socket (`sock_path`, mode 0600), replacing any
    socket file a crashed owner left behind. A process that finds the port taken
    becomes a client of that socket, retrying for CONNECT_RETRY_S; if nothing
    answers, the port is held by something that is not a Sideman listener.

    Share wire: length-prefixed frames. A request is one JSON frame
    {"op": "hello" | "slots" | "rows", ...}; a reply is a JSON frame, plus for
    "rows" one frame of raw bytes (int64 header then float32 values). Never
    pickle. A protocol or field-table mismatch is refused.

    When the owner exits, a client's next call takes over ownership (the
    history the old owner held is gone)."""

    protocol = PROTOCOL
    fields_hash = FIELDS_HASH

    def __init__(self, port: int = DEFAULT_PORT, sock_path: str | pathlib.Path | None = None):
        self.port = port
        self.sock_path = pathlib.Path(sock_path) if sock_path else default_sock_path()
        self.role: str | None = None
        self.rings: dict[tuple[int, int], Ring] = {}
        self.bad_packets = 0
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None          # UDP, owner
        self._share: socket.socket | None = None         # Unix listener, owner
        self._client: socket.socket | None = None        # Unix connection, client
        self._client_lock = threading.Lock()
        self._conns: list[socket.socket] = []            # open share connections, owner
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._start_lock = threading.RLock()
        self._sock_ino: int | None = None                # inode of the socket file we bound

    # ---------------------------------------------------------- lifecycle
    def start(self) -> str:
        with self._start_lock:
            if self.role is not None:
                return self.role
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.bind((HOST, self.port))
            except OSError as e:
                s.close()
                if e.errno != errno.EADDRINUSE:
                    raise
                self._connect()
                self.role = "client"
                return self.role
            s.settimeout(0.2)
            self._sock = s
            self.port = s.getsockname()[1]
            self._stop.clear()
            try:
                self._serve_share()
            except OSError:
                s.close()
                self._sock = None
                raise
            self._spawn(self._udp_loop, "sideman-listen-udp")
            self.role = "owner"
            atexit.register(self.stop)
            return self.role

    def _spawn(self, fn, name, *args) -> None:
        t = threading.Thread(target=fn, args=args, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def address(self) -> tuple[str, int]:
        if self._sock is None:
            raise RuntimeError("not the owner")
        return self._sock.getsockname()

    def stop(self) -> None:
        with self._start_lock:
            self._stop.set()
            if self.role == "owner":
                # Unlink first, and only the socket file this owner bound: a
                # client may already have taken over and bound a new one there.
                try:
                    if self._sock_ino is not None and self.sock_path.stat().st_ino == self._sock_ino:
                        self.sock_path.unlink()
                except OSError:
                    pass
            for sock in (self._share, self._client, *list(self._conns), self._sock):
                _shut(sock)
            for t in self._threads:
                t.join(timeout=2)
            self._sock = self._share = self._client = None
            self._sock_ino = None
            self._conns, self._threads = [], []
            self.role = None

    # ---------------------------------------------------------- owner
    def _udp_loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self.feed(data)
            except Exception:           # nothing a sender does may stop the receiver
                self.bad_packets += 1

    def _serve_share(self) -> None:
        self.sock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.sock_path.exists() or self.sock_path.is_symlink():
            self.sock_path.unlink()             # left by an owner that crashed
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)                   # born 0600, no window at a wider mode
        try:
            srv.bind(str(self.sock_path))
        except OSError:
            srv.close()
            raise
        finally:
            os.umask(old)
        os.chmod(self.sock_path, 0o600)
        self._sock_ino = self.sock_path.stat().st_ino
        srv.listen(8)
        srv.settimeout(0.2)
        self._share = srv
        self._spawn(self._accept_loop, "sideman-listen-share")

    def _accept_loop(self) -> None:
        assert self._share is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._share.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._conns = [c for c in self._conns if c.fileno() != -1] + [conn]
            self._threads = [t for t in self._threads if t.is_alive()]
            self._spawn(self._handle, "sideman-listen-client", conn)

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(None)       # stop() closes it, which ends the recv below
        with conn:
            while not self._stop.is_set():
                try:
                    req = json.loads(_recv_frame(conn))
                except (ConnectionError, OSError, ValueError):
                    return
                try:
                    op = req.get("op")
                    if op == "hello":
                        _send_frame(conn, json.dumps({"protocol": self.protocol,
                                                      "fields_hash": self.fields_hash}).encode())
                    elif op == "slots":
                        _send_frame(conn, json.dumps(
                            {str(k): v for k, v in self.slots().items()}).encode())
                    elif op == "rows":
                        r = self._local_rows(int(req["slot"]), req.get("last_s"))
                        _send_frame(conn, json.dumps({"m": len(r)}).encode())
                        _send_frame(conn, r.header.astype(">i8").tobytes()
                                    + r.values.astype(">f4").tobytes())
                    else:
                        _send_frame(conn, json.dumps({"error": "unknown op %r" % op}).encode())
                except (ConnectionError, OSError):
                    return

    # ---------------------------------------------------------- client
    def _connect(self) -> None:
        deadline = time.monotonic() + CONNECT_RETRY_S
        last: Exception | None = None
        while True:
            try:
                c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                c.settimeout(2.0)
                c.connect(str(self.sock_path))
                break
            except OSError as e:
                c.close()
                last = e
                if time.monotonic() >= deadline:
                    raise PortInUse(
                        "UDP %s:%d is held by a program that is not a Sideman listener "
                        "(no share socket at %s: %s)" % (HOST, self.port, self.sock_path, last))
                time.sleep(0.05)
        try:
            _send_frame(c, json.dumps({"op": "hello"}).encode())
            hello = json.loads(_recv_frame(c))
        except (ConnectionError, OSError, ValueError) as e:
            c.close()
            raise PortInUse("the Sideman listener at %s did not answer (%s: %s)"
                            % (self.sock_path, type(e).__name__, e))
        if hello.get("protocol") != self.protocol or hello.get("fields_hash") != self.fields_hash:
            c.close()
            raise PortInUse("the Sideman listener on UDP %s:%d speaks protocol %s/%s, this "
                            "one %s/%s: restart the older Sideman session"
                            % (HOST, self.port, hello.get("protocol"), hello.get("fields_hash"),
                               self.protocol, self.fields_hash))
        self._client = c

    def _ask(self, req: dict, raw: bool = False):
        for attempt in (0, 1):
            with self._client_lock:
                try:
                    if self._client is None:
                        self._connect()
                    assert self._client is not None
                    _send_frame(self._client, json.dumps(req).encode())
                    head = json.loads(_recv_frame(self._client))
                    body = _recv_frame(self._client) if raw else None
                    return head, body
                except (ConnectionError, OSError, PortInUse):
                    if self._client is not None:
                        self._client.close()
                    self._client = None
                    if attempt:
                        raise
            # the owner went away: take over the stream, or reconnect to a new owner
            self.role = None
            if self.start() == "owner":
                return None, None
        return None, None

    # ---------------------------------------------------------- data
    def feed(self, data: bytes, now: float | None = None) -> None:
        """Handle one datagram (the receive thread calls this; tests call it
        directly)."""
        try:
            addr, args = osc_decode(data)
            if addr != ADDRESS:
                raise ValueError("address %s" % addr)
            h, v = decode_row(args)
        except (ValueError, IndexError, struct.error, OverflowError, TypeError):
            self.bad_packets += 1
            return
        key = (int(h[HCOL["slot"]]), int(h[HCOL["nonce"]]))
        with self._lock:
            ring = self.rings.get(key)
            if ring is None:
                self._evict(key[0], time.monotonic() if now is None else now)
                ring = self.rings[key] = Ring()
        ring.push(h, v, now)

    def _evict(self, slot: int, now: float) -> None:
        """Before a new ring: drop rings silent for RING_IDLE_S, then the oldest
        beyond MAX_RINGS_PER_SLOT on this slot and MAX_RINGS overall. Every device
        reload draws a new nonce, so without this a long session keeps them all.
        Caller holds self._lock."""
        for k in [k for k, r in self.rings.items() if now - r.last_seen > RING_IDLE_S]:
            del self.rings[k]
        same = sorted((r.last_seen, k) for k, r in self.rings.items() if k[0] == slot)
        for _, k in same[:max(0, len(same) - (MAX_RINGS_PER_SLOT - 1))]:
            del self.rings[k]
        if len(self.rings) >= MAX_RINGS:
            for _, k in sorted((r.last_seen, k) for k, r in self.rings.items())[
                    :len(self.rings) - MAX_RINGS + 1]:
                del self.rings[k]

    def _local_slots(self, now: float | None = None) -> dict[int, dict]:
        now = time.monotonic() if now is None else now
        out: dict[int, dict] = {}
        with self._lock:
            items = list(self.rings.items())
        for (slot, nonce), ring in items:
            age = now - ring.last_seen
            d = out.setdefault(slot, {"nonces": [], "last_seen_s": None, "sr": None,
                                      "seq_gaps": 0, "seq_rewinds": 0, "_ring": None})
            if age <= STALE_S:
                d["nonces"].append(nonce)
            if d["last_seen_s"] is None or age < d["last_seen_s"]:
                d["last_seen_s"] = round(age, 3)
                d["_ring"] = ring
                last = ring.rows(1)
                d["sr"] = int(last.h("sr")[0]) if len(last) else None
            d["seq_gaps"] += ring.seq_gaps
            d["seq_rewinds"] += ring.seq_rewinds
        return out

    def _local_rows(self, slot: int, last_s: float | None) -> Rows:
        info = self._local_slots().get(slot)
        if info is None or info["_ring"] is None:
            return empty_rows()
        ring: Ring = info["_ring"]
        return ring.rows(None if last_s is None else int(round(last_s / BLOCK_S)))

    def slots(self, now: float | None = None) -> dict[int, dict]:
        """slot -> {nonces seen within STALE_S, last_seen_s, sr, seq_gaps}.
        Two live nonces on one slot means a duplicated track (same slot copied
        with the device)."""
        if self.role == "client":
            head, _ = self._ask({"op": "slots"})
            if head is not None:
                return {int(k): v for k, v in head.items()}
        return {k: {kk: vv for kk, vv in d.items() if kk != "_ring"}
                for k, d in self._local_slots(now).items()}

    def rows(self, slot: int, *, last_s: float | None = None) -> Rows:
        """The newest rows of a slot's most recently heard device instance."""
        if self.role == "client":
            head, body = self._ask({"op": "rows", "slot": slot, "last_s": last_s}, raw=True)
            if head is not None:
                m = int(head["m"])
                nh = m * len(HEADER) * 8
                h = np.frombuffer(body[:nh], ">i8").astype(np.int64).reshape(m, len(HEADER))
                v = np.frombuffer(body[nh:], ">f4").astype(np.float32).reshape(m, N_VALUES)
                return Rows(h, v)
        return self._local_rows(slot, last_s)


# ------------------------------------------------------------------ the DSP model

@dataclass
class _Biquad:
    b: np.ndarray
    a: np.ndarray


def _k_filters(sr: int) -> list[_Biquad]:
    return [_Biquad(b, a) for b, a in listen.k_weighting(sr)]


def _iir(filters: list[_Biquad], x: np.ndarray, taps: int = listen.K_TAPS) -> np.ndarray:
    """Run a cascade of biquads over x (float64, one channel) by FFT-convolving
    its impulse response, as listen.py does for K-weighting. The device runs the
    same biquads sample by sample; the response is below -170 dB by `taps`."""
    h = np.zeros(taps)
    h[0] = 1.0
    for f in filters:
        h = listen._biquad(f.b, f.a, h)
    n = len(x) + taps - 1
    nfft = 1 << int(math.ceil(math.log2(n)))
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:len(x)]


@dataclass
class ModelState:
    """Transport stamps for blocks_from_array, the same for every block except
    beat (advancing at `tempo`), seq and host_sample (advancing by n)."""
    beat0: float = 0.0
    playing: int = 1
    playing_slot: int = -1
    host_sample0: int = 0
    tempo: float = 120.0
    slot: int = 1
    nonce: int = 12345


def blocks_from_array(x: np.ndarray, sr: int, *, phase: int = 0,
                      state: ModelState | None = None) -> Rows:
    """The rows the device would send for audio x ([n, 1|2] float), starting a
    block every n = round(sr * 0.1) samples from sample `phase`.

    Item 1 (device v0) fills the broadband values only: pow_l, pow_r, pow_lr,
    k_pow_l, k_pow_r and peak. The other columns are zero until item 5."""
    st = state or ModelState()
    x = listen._stereo(np.asarray(x, dtype=np.float64).reshape(len(x), -1))[phase:]
    n = int(round(sr * BLOCK_S))
    m = x.shape[0] // n
    if m == 0:
        return empty_rows()
    xs = x[:m * n]
    k = _k_filters(sr)
    kl = _iir(k, x[:, 0])[:m * n]
    kr = _iir(k, x[:, 1])[:m * n]
    left, right = xs[:, 0].reshape(m, n), xs[:, 1].reshape(m, n)
    vals = np.zeros((m, N_VALUES), np.float32)
    vals[:, COL["pow_l"]] = (left * left).mean(axis=1)[:, None]
    vals[:, COL["pow_r"]] = (right * right).mean(axis=1)[:, None]
    vals[:, COL["pow_lr"]] = (left * right).mean(axis=1)[:, None]
    vals[:, COL["k_pow_l"]] = (kl.reshape(m, n) ** 2).mean(axis=1)[:, None]
    vals[:, COL["k_pow_r"]] = (kr.reshape(m, n) ** 2).mean(axis=1)[:, None]
    vals[:, COL["peak"]] = np.maximum(np.abs(left).max(axis=1), np.abs(right).max(axis=1))[:, None]
    head = np.zeros((m, len(HEADER)), np.int64)
    beats_per_block = BLOCK_S * st.tempo / 60.0
    i = np.arange(m)
    head[:, HCOL["slot"]] = st.slot
    head[:, HCOL["nonce"]] = st.nonce
    head[:, HCOL["seq"]] = i
    head[:, HCOL["playing"]] = st.playing
    head[:, HCOL["playing_slot"]] = st.playing_slot
    head[:, HCOL["beat_x1e6"]] = np.round((st.beat0 + i * beats_per_block) * 1e6).astype(np.int64)
    head[:, HCOL["host_sample"]] = st.host_sample0 + phase + i * n
    head[:, HCOL["sr"]] = sr
    head[:, HCOL["n"]] = n
    return Rows(head, vals)


# ------------------------------------------------------------------ analysis

def analyze_rows(rows: Rows) -> dict:
    """Measurements over a span of rows, with `analyze_array`'s keys where the
    rows can give them, plus the current momentary (last 400 ms) and short-term
    (last 3 s) loudness. `approximate` lists keys that are close to, but not
    computed the same way as, the file analysis; keys the device does not yet
    send are left out. Readings are pre-fader (the device sits before the
    mixer)."""
    m = len(rows)
    out: dict = {"blocks": m, "duration_s": round(m * BLOCK_S, 2), "channels": 2,
                 "approximate": []}
    if m == 0:
        out.update({"integrated_lufs": None, "short_term_max_lufs": None,
                    "momentary_max_lufs": None, "momentary_lufs": None,
                    "short_term_lufs": None, "sample_peak_db": None, "rms_db": None,
                    "crest_db": None})
        return out
    sub = (rows.v("k_pow_l").astype(np.float64) + rows.v("k_pow_r"))
    out.update(listen.loudness_from_subblocks(sub))
    mom, sh = listen.MOMENTARY_SUBBLOCKS, listen.SHORT_TERM_SUBBLOCKS

    def window(k: int) -> float | None:
        if m < k:
            return None
        v = float(listen._lufs(sub[-k:].mean()))
        return round(v, 2) if v > listen.ABS_GATE_LUFS else None

    out["momentary_lufs"] = window(mom)
    out["short_term_lufs"] = window(sh)
    p = (rows.v("pow_l").astype(np.float64) + rows.v("pow_r")) / 2.0
    ms = float(p.mean())
    peak = float(rows.v("peak").max())
    out["sample_peak_db"] = round(listen._db_amp(peak), 2)
    out["rms_db"] = round(listen._db(ms), 2)
    out["crest_db"] = round(listen._db_amp(peak) - listen._db(ms), 2) if ms > 0 else None
    return out
