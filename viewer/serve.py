"""Static file server for the viewer, with HTTP Range support (needed to seek in big MP4 files).

Python's `python -m http.server` ignores Range requests, so browsers cannot seek in a 10 GB video.
Usage (repo root):  uv run python viewer/serve.py [port]      then open http://localhost:8000/viewer/index.html
"""
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    range = None

    def send_head(self):
        self.range = None
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        a, b = m.groups()
        start, end = (size - int(b), size - 1) if not a else (int(a), int(b) if b else size - 1)
        end = min(end, size - 1)
        if start > end:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(start)
        self.range = (start, end)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return f

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, src, dst):
        try:
            if self.range is None:
                return super().copyfile(src, dst)
            left = self.range[1] - self.range[0] + 1
            while left > 0:
                buf = src.read(min(1 << 20, left))
                if not buf:
                    break
                dst.write(buf)
                left -= len(buf)
        except (BrokenPipeError, ConnectionResetError):
            pass                                          # browser aborted a seek

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving {ROOT} on http://localhost:{port}/viewer/index.html")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT))).serve_forever()
