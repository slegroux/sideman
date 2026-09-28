# AbletonLOM - thin socket shell.
#
# DESIGN: this file changes rarely, because changing it requires restarting Live.
# All real logic lives in handlers.py, which is hot-reloadable via the "reload" op.
# Keep this file boring.
#
# Threading contract (non-negotiable):
#   accept-thread (daemon) -> client-thread (daemon) -> schedule_message(0, task)
#   The Live API is ONLY touched inside `task`, which runs on Live's main thread.
#   The client thread blocks on a Queue until the main thread posts a result.
# Violating this freezes or crashes the Live GUI.

from __future__ import absolute_import, print_function

import json
import socket
import threading
import traceback

try:
    import queue
except ImportError:  # pragma: no cover - Python 2 fallback
    import Queue as queue

from _Framework.ControlSurface import ControlSurface

HOST = "127.0.0.1"
PORT = 9878          # deliberately NOT 9877 - coexists with an existing AbletonMCP
MAIN_THREAD_TIMEOUT = 15.0
RECV_SIZE = 65536


def create_instance(c_instance):
    return AbletonLOM(c_instance)


class AbletonLOM(ControlSurface):

    def __init__(self, c_instance):
        ControlSurface.__init__(self, c_instance)
        self._server = None
        self._running = False
        self._handlers = None
        self._handler_error = None
        self._load_handlers()
        self._start_server()
        self.log_message("AbletonLOM: listening on %s:%d" % (HOST, PORT))
        self.show_message("AbletonLOM ready on port %d" % PORT)

    # ---------------------------------------------------------------- handlers

    def _load_handlers(self):
        """(Re)import the engine module. Never let a bad engine kill the server."""
        # Tear down BEFORE reloading. reload() re-executes the module, so any
        # listener still registered would be orphaned: Live keeps firing the
        # callback and nothing can remove it any more.
        try:
            if self._handlers and hasattr(self._handlers, "teardown"):
                self._handlers.teardown(self)
        except Exception:
            self.log_message("AbletonLOM: teardown before reload failed\n"
                             + traceback.format_exc())
        try:
            import importlib
            from . import handlers
            importlib.reload(handlers)
            self._handlers = handlers
            self._handler_error = None
            self.log_message("AbletonLOM: handlers loaded")
            return True
        except Exception:
            self._handlers = None
            self._handler_error = traceback.format_exc()
            self.log_message("AbletonLOM: handler load FAILED\n" + self._handler_error)
            return False

    # ----------------------------------------------------------------- server

    def _start_server(self):
        try:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((HOST, PORT))
            self._server.listen(5)
            self._server.settimeout(1.0)
            self._running = True
            t = threading.Thread(target=self._accept_loop)
            t.daemon = True
            t.start()
        except Exception:
            self.log_message("AbletonLOM: bind failed\n" + traceback.format_exc())

    def _accept_loop(self):
        while self._running:
            try:
                client, _addr = self._server.accept()
                t = threading.Thread(target=self._client_loop, args=(client,))
                t.daemon = True
                t.start()
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    self.log_message("AbletonLOM: accept error\n"
                                     + traceback.format_exc())

    def _client_loop(self, client):
        """Newline-delimited JSON. One request per line, one response per line."""
        buf = b""
        try:
            client.settimeout(None)
            while self._running:
                chunk = client.recv(RECV_SIZE)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    resp = self._handle_line(line)
                    client.sendall(json.dumps(resp).encode("utf-8") + b"\n")
        except Exception:
            pass
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _handle_line(self, line):
        req_id = None
        try:
            req = json.loads(line.decode("utf-8"))
            req_id = req.get("id")
            op = req.get("op")
            params = req.get("params") or {}
        except Exception as e:
            return {"id": req_id, "ok": False,
                    "error": {"type": "BadRequest", "message": str(e)}}

        # --- ops served by the shell itself, engine not required -------------
        if op == "ping":
            return {"id": req_id, "ok": True,
                    "result": {"pong": True, "port": PORT,
                               "handlers": self._handlers is not None,
                               "handler_error": self._handler_error}}
        if op == "reload":
            # Marshalled like every other op: _load_handlers tears down live
            # observers first, and removing a listener IS a Live API call.
            # Serving it on the socket thread violated the contract above.
            return self._run_on_main_thread(req_id, self._reload_response)

        if self._handlers is None:
            return {"id": req_id, "ok": False,
                    "error": {"type": "HandlerLoadError",
                              "message": self._handler_error or "handlers not loaded"}}

        def dispatch():
            return {"ok": True,
                    "result": self._handlers.dispatch(self, op, params)}

        return self._run_on_main_thread(req_id, dispatch)

    def _reload_response(self):
        ok = self._load_handlers()
        return {"ok": ok,
                "result": {"reloaded": ok},
                "error": None if ok else {"type": "HandlerLoadError",
                                          "message": self._handler_error}}

    def _run_on_main_thread(self, req_id, fn):
        """Run `fn` on Live's main thread and wait for its response dict."""
        result_q = queue.Queue(1)

        def task():
            try:
                result_q.put(fn())
            except Exception as e:
                result_q.put({"ok": False,
                              "error": {"type": type(e).__name__,
                                        "message": str(e),
                                        "traceback": traceback.format_exc()}})

        try:
            self.schedule_message(0, task)
        except Exception as e:
            return {"id": req_id, "ok": False,
                    "error": {"type": "ScheduleError", "message": str(e)}}

        try:
            out = result_q.get(timeout=MAIN_THREAD_TIMEOUT)
        except queue.Empty:
            return {"id": req_id, "ok": False,
                    "error": {"type": "Timeout",
                              "message": "main thread did not respond in %.1fs"
                                         % MAIN_THREAD_TIMEOUT}}
        out["id"] = req_id
        return out

    # ------------------------------------------------------------- lifecycle

    def disconnect(self):
        self._running = False
        try:
            if self._server:
                self._server.close()
        except Exception:
            pass
        try:
            if self._handlers and hasattr(self._handlers, "teardown"):
                self._handlers.teardown(self)
        except Exception:
            pass
        ControlSurface.disconnect(self)
