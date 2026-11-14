"""sideman.client.Live's two ways of connecting, against a small server on the loopback: one connection
per request (the default), or one held for a `with` block. No Live needed."""
import json
import pathlib
import socket
import sys
import threading

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman import Live, LiveError  # noqa: E402


class Server:
    """Answers each request line with {"ok": true, "result": {"n": <request number>, "op": <op>}} and counts
    the connections it accepted. `drop_after` closes a connection after that many replies; `silent` never replies."""

    def __init__(self, drop_after=None, silent=False):
        self.drop_after, self.silent = drop_after, silent
        self.connections, self.requests = 0, []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        buf, replies = b"", 0
        with conn:
            while True:
                try:
                    chunk = conn.recv(65536)
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    req = json.loads(line)
                    self.requests.append(req)
                    if self.silent:
                        continue
                    if req["op"] == "refuse":
                        reply = {"ok": False, "error": {"type": "PermissionError", "message": "no"}}
                    else:
                        reply = {"ok": True, "result": {"n": len(self.requests), "op": req["op"]}}
                    conn.sendall(json.dumps(reply).encode() + b"\n")
                    replies += 1
                    if self.drop_after and replies >= self.drop_after:
                        return

    def close(self):
        self.sock.close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


def test_outside_a_block_every_request_opens_its_own_connection(server):
    live = Live(port=server.port, timeout=2)
    assert [live.request("ping")["n"] for _ in range(3)] == [1, 2, 3]
    assert server.connections == 3 and live._sock is None


def test_a_with_block_sends_every_request_down_one_connection_and_closes_it(server):
    with Live(port=server.port, timeout=2) as live:
        assert [live.get("live_set", "tempo")["n"] for _ in range(5)] == [1, 2, 3, 4, 5]
        assert live.count("live_set", "tracks")["op"] == "count"
        assert server.connections == 1
        held = live._sock
    assert live._sock is None and held.fileno() == -1          # closed on leaving the block
    assert [r["op"] for r in server.requests] == ["get"] * 5 + ["count"]
    assert server.requests[0] == {"id": "py", "op": "get", "params": {"path": "live_set", "property": "tempo"}}
    assert live.request("ping")["n"] == 7 and server.connections == 2 and live._sock is None   # per request again


def test_a_refusal_keeps_the_held_connection(server):
    with Live(port=server.port, timeout=2) as live:
        live.request("ping")
        with pytest.raises(LiveError) as e:
            live.request("refuse")
        assert e.value.type == "PermissionError"
        assert live.request("ping")["n"] == 3 and server.connections == 1


def test_a_held_connection_live_closed_is_opened_again_without_losing_the_request():
    server = Server(drop_after=1)                # the server hangs up after each reply, as a reload does
    try:
        with Live(port=server.port, timeout=2) as live:
            got = []
            for _ in range(3):
                try:
                    got.append(live.request("ping")["op"])
                except LiveError as e:           # the hang-up can also be seen while reading: reported, not retried
                    assert e.type == "ConnectionClosed"
                    got.append("closed")
            assert got[0] == "ping" and "ping" in got[1:]
            # nothing was sent twice: every request the server saw got exactly one reply or one error
            assert len(server.requests) <= 3
    finally:
        server.close()


def test_a_timeout_drops_the_held_connection_so_a_late_reply_cannot_answer_the_next_request():
    server = Server(silent=True)
    try:
        with Live(port=server.port, timeout=0.3) as live:
            with pytest.raises(LiveError) as e:
                live.request("ping")
            assert e.value.type == "Timeout" and live._sock is None
            with pytest.raises(LiveError):
                live.request("ping")
            assert server.connections == 2       # the second request did not reuse the first socket
    finally:
        server.close()


def test_nothing_listening_is_a_connection_refused_error_in_a_block_too():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with Live(port=port, timeout=1) as live:
        with pytest.raises(LiveError) as e:
            live.ping()
    assert e.value.type == "ConnectionRefused"


def test_close_is_safe_to_call_twice_and_outside_a_block(server):
    live = Live(port=server.port, timeout=2)
    live.close()
    live.__enter__()
    live.request("ping")
    live.close()
    live.close()
    assert live._sock is None and live._held is False
