from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .collect import fetch, summarize

PAGE = Path(__file__).with_name("index.html")


class Viewer:
    def __init__(self, namespace: str, pool: str, interval: float, keep: int, snapshot: str | None = None):
        self.namespace, self.pool, self.interval = namespace, pool, interval
        self.snapshot = json.loads(Path(snapshot).read_text(encoding="utf-8")) if snapshot else None
        self.samples: dict[str, deque] = {}
        self.keep = keep
        self.lock = threading.Lock()
        self.body = b"{}"

    def raw(self) -> dict:
        return self.snapshot if self.snapshot is not None else fetch(self.namespace, self.pool)

    def tick(self) -> None:
        now = time.time()
        try:
            raw = self.raw()
            for ep in raw.get("executionplans") or []:
                st = (raw.get("progress") or {}).get((ep.get("spec") or {}).get("workloadRef")) or {}
                if st.get("step") is None:
                    continue
                q = self.samples.setdefault(ep["metadata"]["name"], deque(maxlen=self.keep))
                q.append([round(now, 1), st["step"], st.get("stepsPerSec"), (ep.get("status") or {}).get("generation"),
                          st.get("status")])
            out = summarize(raw, raw.get("samples") or {k: list(v) for k, v in self.samples.items()})
            out.update(namespace=self.namespace, time=now, error=None)
        except Exception as e:
            out = {"namespace": self.namespace, "time": now, "error": str(e), "workloads": [], "requests": []}
        with self.lock:
            self.body = json.dumps(out).encode()

    def run(self) -> None:
        while True:
            self.tick()
            time.sleep(self.interval)


def handler(viewer: Viewer):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                with viewer.lock:
                    body = viewer.body
                self._send(200, body, "application/json")
            elif path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):
            pass

    return H


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m grp_runtime.viewer")
    ap.add_argument("--namespace", default="grp-runtime")
    ap.add_argument("--pool", default="gpu-pool")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--keep", type=int, default=1800)
    ap.add_argument("--snapshot")
    a = ap.parse_args()
    if not a.snapshot:
        from kubernetes import config
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
    viewer = Viewer(a.namespace, a.pool, a.interval, a.keep, a.snapshot)
    viewer.tick()
    threading.Thread(target=viewer.run, daemon=True).start()
    ThreadingHTTPServer((a.host, a.port), handler(viewer)).serve_forever()


if __name__ == "__main__":
    main()
