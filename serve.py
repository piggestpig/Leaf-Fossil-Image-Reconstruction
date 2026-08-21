#!/usr/bin/env python3
"""Minimal static server with correct MIME for ORT ES modules.

Default port avoids Windows Hyper-V / WSL excluded ranges (e.g. 8867–8966),
which cause WinError 10013 on 8899/8900 even when nothing is listening.
"""
from __future__ import annotations

import os
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
# Prefer ports outside Hyper-V/WSL excludedportrange (this PC often
# reserves 8667–9366, which blocks old 8899/8900 and even 8765).
PREFERRED_PORTS = (7681, 7682, 8080, 5173, 3000, 9400, 9401, 18080)

MIME_OVERRIDE = {
    ".mjs": "text/javascript",
    ".js": "text/javascript",
    ".wasm": "application/wasm",
    ".onnx": "application/octet-stream",
    ".json": "application/json",
    ".css": "text/css",
    ".html": "text/html",
}

NO_CACHE = (".mjs", ".js", ".wasm", ".html", ".css")


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def guess_type(self, path):  # noqa: N802
        low = str(path).lower().split("?", 1)[0]
        for ext, ctype in MIME_OVERRIDE.items():
            if low.endswith(ext):
                return ctype
        return super().guess_type(path)

    def _nocache(self) -> bool:
        return self.path.split("?", 1)[0].lower().endswith(NO_CACHE)

    def do_GET(self):  # noqa: N802
        if self._nocache():
            for k in ("If-Modified-Since", "If-None-Match"):
                if k in self.headers:
                    del self.headers[k]
        return super().do_GET()

    def do_HEAD(self):  # noqa: N802
        if self._nocache():
            for k in ("If-Modified-Since", "If-None-Match"):
                if k in self.headers:
                    del self.headers[k]
        return super().do_HEAD()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if self._nocache():
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def bind_server(host: str, ports: tuple[int, ...]) -> tuple[ReusableServer, int]:
    last_err: OSError | None = None
    for port in ports:
        try:
            return ReusableServer((host, port), Handler), port
        except OSError as e:
            last_err = e
            print(f"[warn] cannot bind :{port} — {e}", file=sys.stderr)
    assert last_err is not None
    raise last_err


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    required = (
        "index.html",
        "app.js",
        "vendor/ort/ort.webgpu.min.mjs",
    )
    for need in required:
        if not (root / need).exists():
            print(f"[ERROR] missing {need}", file=sys.stderr)
            sys.exit(1)
    yoloe_ok = (root / "yoloe_stone_ruler_26s.onnx").exists() or (root / "yoloe_stone_26s.onnx").exists()
    if not yoloe_ok:
        print("[ERROR] missing yoloe_stone_ruler_26s.onnx (or yoloe_stone_26s.onnx)", file=sys.stderr)
        sys.exit(1)
    amodal_ok = (root / "amodal_dino_q4.manifest.json").exists() or (root / "amodal_dino_q4.onnx").exists()
    if not amodal_ok:
        print(
            "[ERROR] missing amodal_dino_q4.manifest.json (chunks) or amodal_dino_q4.onnx",
            file=sys.stderr,
        )
        sys.exit(1)

    env_port = os.environ.get("LEAF_FOSSIL_PORT", "").strip()
    ports = (int(env_port),) + PREFERRED_PORTS if env_port.isdigit() else PREFERRED_PORTS

    try:
        httpd, port = bind_server(HOST, ports)
    except OSError as e:
        print(f"[ERROR] Cannot bind any preferred port — {e}", file=sys.stderr)
        print("        Try: set LEAF_FOSSIL_PORT=9400", file=sys.stderr)
        print("        Or check: netsh interface ipv4 show excludedportrange protocol=tcp", file=sys.stderr)
        sys.exit(1)

    url = f"http://{HOST}:{port}/index.html"
    print(f"Minimal YOLOE + Amodal Q4: {url}")
    print(f"Python: {sys.executable}")
    print("Press Ctrl+C to stop. Keep this window open while using the page.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
