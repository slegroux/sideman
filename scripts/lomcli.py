#!/usr/bin/env python3
"""Raw socket client for the AbletonLOM remote script.

Talks the wire protocol directly, so the engine can be exercised without the
MCP layer in the way. This is the Phase 1 verification tool.

  ./lomcli.py ping
  ./lomcli.py reload
  ./lomcli.py describe "live_set"
  ./lomcli.py get "live_set" tempo
  ./lomcli.py set "live_set" tempo 124
  ./lomcli.py call "live_set" create_midi_track -1
  ./lomcli.py count "live_set" tracks
  ./lomcli.py types
"""
import json
import socket
import sys

HOST, PORT = "127.0.0.1", 9878


def request(op, params=None, timeout=20.0):
    s = socket.create_connection((HOST, PORT), timeout=timeout)
    try:
        s.sendall(json.dumps({"id": "cli", "op": op,
                              "params": params or {}}).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode())
    finally:
        s.close()


def _num(x):
    try:
        return int(x)
    except ValueError:
        pass
    try:
        return float(x)
    except ValueError:
        pass
    return {"true": True, "false": False, "null": None}.get(x, x)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    op, rest = argv[0], argv[1:]
    params = {}
    if op in ("describe", "types", "ping", "reload"):
        if rest:
            params["path"] = rest[0]
    elif op == "get":
        params = {"path": rest[0], "property": rest[1]}
    elif op == "set":
        params = {"path": rest[0], "property": rest[1], "value": _num(rest[2])}
    elif op == "call":
        params = {"path": rest[0], "function": rest[1],
                  "args": [_num(a) for a in rest[2:]], "confirm": True}
    elif op == "count":
        params = {"path": rest[0], "child": rest[1]}
    else:
        params = json.loads(rest[0]) if rest else {}

    try:
        resp = request(op, params)
    except ConnectionRefusedError:
        print("REFUSED: nothing listening on %s:%d.\n"
              "Is Live running with AbletonLOM enabled as a Control Surface?"
              % (HOST, PORT))
        return 1
    except socket.timeout:
        print("TIMEOUT waiting for Live")
        return 1

    # No truncation: callers pipe this into jq/python, and a silent cut
    # produces invalid JSON downstream.
    print(json.dumps(resp, indent=2))
    return 0 if resp.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
