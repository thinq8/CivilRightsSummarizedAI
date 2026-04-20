#!/usr/bin/env python3
"""Tiny local launcher and CORS proxy for the standalone Clearinghouse tools.

Run from the repository root:

    python tools/clearinghouse_api_proxy.py

Then open:

    http://127.0.0.1:8765/

The generator will be served from localhost and will use this same helper for
Clearinghouse API requests. The proxy only forwards requests to
clearinghouse.net/api/v2p1.
"""

from __future__ import annotations

import json
import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import webbrowser

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
ALLOWED_HOST = "clearinghouse.net"
ALLOWED_PREFIX = "/api/v2p1/"
TOOLS_DIR = Path(__file__).resolve().parent
GENERATOR_HTML = TOOLS_DIR / "case-summary-generator.html"
EVALUATOR_HTML = TOOLS_DIR / "case-summary-evaluator.html"


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "ClearinghouseLocalProxy/1.0"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_empty(204)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/case-summary-generator.html"}:
            self._send_file(GENERATOR_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/case-summary-evaluator.html":
            self._send_file(EVALUATOR_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(
            404,
            {
                "detail": "Not found. Open http://127.0.0.1:8765/ for the generator.",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            target_url = str(payload.get("url", ""))
            token = str(payload.get("token", "")).strip()

            self._validate_target(target_url)
            if not token:
                self._send_json(400, {"detail": "A Clearinghouse API token is required."})
                return

            request = Request(
                target_url,
                headers={
                    "Authorization": f"Token {token.removeprefix('Token ').strip()}",
                    "Accept": "application/json",
                    "User-Agent": "CivilRightsSummarizedAI-StandaloneProxy/1.0",
                },
                method="GET",
            )
            with urlopen(request, timeout=30) as response:
                body = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")

            self._send_bytes(status, body, content_type)

        except ValueError as exc:
            self._send_json(400, {"detail": str(exc)})
        except HTTPError as exc:
            body = exc.read() or json.dumps({"detail": str(exc)}).encode("utf-8")
            self._send_bytes(exc.code, body, exc.headers.get("Content-Type", "application/json"))
        except (URLError, TimeoutError) as exc:
            self._send_json(502, {"detail": f"Could not reach Clearinghouse API: {exc}"})
        except Exception as exc:  # pragma: no cover - defensive local utility
            self._send_json(500, {"detail": f"Proxy error: {exc}"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _validate_target(self, target_url: str) -> None:
        parsed = urlparse(target_url)
        if parsed.scheme != "https" or parsed.netloc != ALLOWED_HOST:
            raise ValueError("Proxy only allows https://clearinghouse.net requests.")
        if not parsed.path.startswith(ALLOWED_PREFIX):
            raise ValueError("Proxy only allows Clearinghouse API v2.1 paths.")

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self._cors_headers()
        self.end_headers()

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json(404, {"detail": f"Missing file: {path.name}"})
            return
        self._send_bytes(200, path.read_bytes(), content_type)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the standalone Clearinghouse tools with a local API proxy.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local port to bind. Default: 8765")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()

    try:
        server = ThreadingHTTPServer((HOST, args.port), ProxyHandler)
    except OSError as exc:
        print(f"Could not start on http://{HOST}:{args.port}/: {exc}", file=sys.stderr)
        print("If an older proxy is running, stop it with Ctrl+C and run this command again.", file=sys.stderr)
        print("Or choose another port, for example: python tools/clearinghouse_api_proxy.py --port 8766", file=sys.stderr)
        raise SystemExit(1)

    print(f"Clearinghouse local proxy running at http://{HOST}:{args.port}/")
    print(f"Generator: http://{HOST}:{args.port}/")
    print(f"Evaluator:  http://{HOST}:{args.port}/case-summary-evaluator.html")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        try:
            webbrowser.open(f"http://{HOST}:{args.port}/")
        except Exception:
            pass
    server.serve_forever()


if __name__ == "__main__":
    main()
