"""
server_handler.py
-----------------

Simple HTTP handler factory. This file keeps the web server code separate
from the model code and the drawing code.

Routes we serve:
- "/"            → main static HTML page (from static_dir)
- "/static/..."  → CSS/JS/assets (from static_dir)
- "/video"       → MJPEG (a stream of JPEG images) with the latest frame
- "/device"      → ask the app to switch device (CPU or MYRIAD)
- "/heatmaps"    → ask the app to show or hide the heatmap grid

This separation makes it easier to learn: web server vs model vs drawing.
"""

import os
import time
import mimetypes
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


def create_handler(static_dir: str,
                   get_latest_jpeg,
                   request_device_callback,
                   request_heatmaps_callback,
                   is_running):
    """Create a simple HTTP request handler bound to your app callbacks.

    Args:
        static_dir: where index.html and /static files live
        get_latest_jpeg: () -> Optional[bytes], returns most recent JPEG frame
        request_device_callback: (str) -> None, called with 'CPU' or 'MYRIAD'
        request_heatmaps_callback: (bool) -> None, show or hide the heatmap grid
        is_running: () -> bool, tells the streaming loop when to stop

    Returns:
        A subclass of BaseHTTPRequestHandler you can pass to ThreadingHTTPServer.
    """
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)

            # Root -> serve static index.html
            if parsed.path == "/":
                index_path = os.path.join(static_dir, "index.html")
                if os.path.exists(index_path) and os.path.isfile(index_path):
                    with open(index_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(content)
                    return
                self.send_response(404)
                self.end_headers()
                return

            # Static assets under /static/
            if parsed.path.startswith("/static/"):
                rel = parsed.path[len("/static/"):]  # strip prefix
                asset_path = os.path.normpath(os.path.join(static_dir, rel))
                if not asset_path.startswith(os.path.normpath(static_dir)):
                    # Prevent path traversal
                    self.send_response(403)
                    self.end_headers()
                    return
                if os.path.exists(asset_path) and os.path.isfile(asset_path):
                    mime_type, _ = mimetypes.guess_type(asset_path)
                    if not mime_type:
                        mime_type = "application/octet-stream"
                    with open(asset_path, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", mime_type)
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(404)
                self.end_headers()
                return

            # MJPEG video stream
            if parsed.path == "/video":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    # Continuously write JPEG frames in a multipart response
                    while is_running():
                        jpeg = get_latest_jpeg()
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            # Device switch request
            if parsed.path == "/device":
                params = parse_qs(parsed.query)
                requested = params.get("name", [None])[0]
                if requested not in ("CPU", "MYRIAD"):
                    self.send_response(400)
                    self.end_headers()
                    return
                request_device_callback(requested)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write((f"Switching to {requested}").encode())
                return

            # Heatmap grid show/hide request
            if parsed.path == "/heatmaps":
                params = parse_qs(parsed.query)
                show_text = params.get("show", [None])[0]
                if show_text not in ("0", "1"):
                    self.send_response(400)
                    self.end_headers()
                    return

                show_heatmaps = show_text == "1"
                request_heatmaps_callback(show_heatmaps)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                message = "Heatmaps shown" if show_heatmaps else "Heatmaps hidden"
                self.wfile.write(message.encode())
                return

            self.send_response(404)
            self.end_headers()

    return Handler
