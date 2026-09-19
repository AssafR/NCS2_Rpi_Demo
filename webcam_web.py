import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import mimetypes
import os

import cv2
import numpy as np
import threading
model_lock = threading.Lock()
from pose_model_runner import PoseModelRunner
from pose_result_processor import render_pose_on_frame, create_pose_mask


# Pose estimation is now implemented in a separate module: pose_estimation.PoseEstimator


# =========================================================
# Configuration
# =========================================================

MODEL_PATH = "model/human-pose-estimation-0001.xml"

CAMERA_ID = 0

MODEL_W = 456
MODEL_H = 256

CAMERA_W = 640
CAMERA_H = 480
CAMERA_FPS = 15

WEB_PORT = 8080


# Pose model loading is delegated to PoseModelRunner; no global OpenVINO setup here.


runner = PoseModelRunner(
    MODEL_PATH,
    initial_device="MYRIAD",
    model_w=MODEL_W,
    model_h=MODEL_H,
    camera_w=CAMERA_W,
    camera_h=CAMERA_H,
    camera_fps=CAMERA_FPS
)


# =========================================================
# Webcam setup
# =========================================================

cap = cv2.VideoCapture(
    CAMERA_ID,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_W
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_H
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam"
    )

print("Webcam opened successfully.")


# =========================================================
# Shared frame
# =========================================================

latest_jpeg = None
frame_lock = threading.Lock()

running = True
# Device change requests are signaled by HTTP handler; applied safely in the inference thread
requested_device = None


# =========================================================
# Image preprocessing
# =========================================================

def prepare_frame(frame):
    resized = cv2.resize(
        frame,
        (MODEL_W, MODEL_H)
    )

    tensor = resized.transpose(
        2, 0, 1
    )

    tensor = tensor[
        np.newaxis, ...
    ]

    tensor = tensor.astype(
        np.float32
    )

    return tensor


# =========================================================
# Pose estimation logic moved to external modules (pose_model_runner, pose_result_processor)


def draw_pose(frame, points):
    # Draw skeleton
    for part_a, part_b in POSE_PAIRS:
        a = points.get(part_a)
        b = points.get(part_b)

        if (
            a is not None
            and b is not None
        ):
            cv2.line(
                frame,
                (a[0], a[1]),
                (b[0], b[1]),
                (0, 255, 255),
                3
            )

    # Draw joints
    for point in points.values():
        if point is not None:
            cv2.circle(
                frame,
                (point[0], point[1]),
                5,
                (0, 0, 255),
                -1
            )


# =========================================================
# Find the heatmap output
# =========================================================

def get_heatmaps(result, compiled):
    """
    human-pose-estimation-0001 has:

        38-channel output -> PAFs
        19-channel output -> heatmaps

    Find the heatmap output by shape rather
    than relying on output order.
    """

    for output in compiled.outputs:
        shape = output.shape

        if len(shape) == 4 and shape[1] == 19:
            return result[output]

    raise RuntimeError(
        "Could not find the 19-channel "
        "pose heatmap output"
    )


# =========================================================
# Inference thread
# =========================================================

def inference_loop():
    global latest_jpeg
    global requested_device

    smoothed_inference_ms = None
    smoothed_loop_fps = None

    last_frame_time = (
        time.perf_counter()
    )

    while running:
        ret, frame = cap.read()

        if not ret:
            print(
                "Failed to capture webcam frame"
            )

            time.sleep(0.1)
            continue

        # Apply any pending device switch request here (single-writer pattern)
        if requested_device is not None:
            to_set = requested_device
            requested_device = None
            print(f"\nApplying device switch to: {to_set}...")
            ok = runner.set_device(to_set)
            print("Switch successful." if ok else "Switch failed.")

        tensor, frame_for_processing = None, frame
        # Use the new PoseModelRunner to execute and obtain results
        res = runner.run(frame_for_processing)
        # Draw pose using the result processor
        points = res["points"]
        render_pose_on_frame(frame_for_processing, points)
        heatmaps = res["heatmaps"]
        device_name = res["device"]
        inference_ms = res["elapsed_ms"]
        # Update FPS calculations (simple approach using elapsed time)
        now = time.perf_counter()
        loop_time = now - last_frame_time
        last_frame_time = now
        instant_loop_fps = 1.0 / loop_time if loop_time > 0 else 0.0
        if smoothed_loop_fps is None:
            smoothed_loop_fps = instant_loop_fps
        else:
            smoothed_loop_fps = (
                0.9 * smoothed_loop_fps
                + 0.1 * instant_loop_fps
            )

        # Compute instantaneous inference FPS from the last inference time
        inference_fps = (1000.0 / inference_ms) if inference_ms > 0 else 0.0

        # Overlay text using device_name and timings
        cv2.putText(
            frame_for_processing,
            f"Device: {device_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame_for_processing,
            f"Inference: {inference_ms:.0f} ms",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame_for_processing,
            f"Inference FPS: {inference_fps:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame_for_processing,
            f"Actual loop FPS: {smoothed_loop_fps:.2f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        success, jpeg = cv2.imencode(
            ".jpg",
            frame_for_processing,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if success:
            with frame_lock:
                latest_jpeg = jpeg.tobytes()


# =========================================================
# Web page
# =========================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>
        Raspberry Pi Neural Accelerator Demo
    </title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #111;
            color: white;
            text-align: center;
            margin: 30px;
        }

        h1 {
            font-size: 28px;
        }

        img {
            max-width: 90%;
            border: 2px solid #555;
            margin-top: 20px;
        }

        button {
            font-size: 22px;
            padding: 12px 30px;
            margin: 15px;
            cursor: pointer;
        }

        .note {
            color: #aaa;
            margin-top: 10px;
        }

    </style>

</head>

<body>

    <h1>
        Raspberry Pi 3 + Neural Accelerator
    </h1>

    <div>

        <button onclick="setDevice('CPU')">
            Raspberry Pi CPU
        </button>

        <button onclick="setDevice('MYRIAD')">
            Intel NCS2
        </button>

    </div>

    <div class="note">
        Same CNN. Same camera.
        Different inference hardware.
    </div>

    <img src="/video">

    <script>

        function setDevice(device) {

            fetch(
                '/device?name=' + device
            )
            .then(
                response => response.text()
            )
            .then(
                text => console.log(text)
            );

        }

    </script>

</body>

</html>
"""


# =========================================================
# HTTP server
# =========================================================

class Handler(
    BaseHTTPRequestHandler
):

    def do_GET(self):
        # Device selection handled via runner

        parsed = urlparse(
            self.path
        )

        # ---------------------------------------------
        # Main page (static assets)
        # ---------------------------------------------

        if parsed.path == "/":
            # Serve static index.html if available
            base_dir = os.path.dirname(__file__)
            fs_path = os.path.join(base_dir, "static", "index.html")
            if os.path.exists(fs_path) and os.path.isfile(fs_path):
                with open(fs_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(content)
                return
            # Fallback to inline HTML if static not present
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html"
            )
            self.end_headers()
            self.wfile.write(
                HTML.encode("utf-8")
            )

        # ---------------------------------------------
        # Video stream
        # ---------------------------------------------

        elif parsed.path == "/video":
            self.send_response(200)

            self.send_header(
                "Content-Type",
                (
                    "multipart/x-mixed-replace; "
                    "boundary=frame"
                )
            )

            self.end_headers()

            try:
                while running:
                    with frame_lock:
                        jpeg = latest_jpeg

                    if jpeg is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(
                        b"--frame\r\n"
                    )

                    self.wfile.write(
                        b"Content-Type: "
                        b"image/jpeg\r\n\r\n"
                    )

                    self.wfile.write(
                        jpeg
                    )

                    self.wfile.write(
                        b"\r\n"
                    )

                    time.sleep(0.03)

            except (
                BrokenPipeError,
                ConnectionResetError
            ):
                pass

        # ---------------------------------------------
        # Device switching
        # ---------------------------------------------

        elif parsed.path == "/device":
            params = parse_qs(parsed.query)
            requested = params.get("name", [None])[0]
            if requested not in ("CPU", "MYRIAD"):
                self.send_response(400)
                self.end_headers()
                return

            # Signal the inference thread to switch devices at a safe point
            global requested_device
            requested_device = requested
            print(f"Device switch requested: {requested}")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write((f"Switching to {requested}").encode())

        # ---------------------------------------------
        # Static assets (CSS/JS)
        # ---------------------------------------------
        elif parsed.path.startswith("/static/"):
            base_dir = os.path.dirname(__file__)
            asset_rel = parsed.path.lstrip("/")
            asset_path = os.path.join(base_dir, asset_rel)
            if os.path.exists(asset_path) and os.path.isfile(asset_path):
                mime_type, _ = mimetypes.guess_type(asset_path)
                if mime_type is None:
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

        else:
            self.send_response(404)
            self.end_headers()


# =========================================================
# Start
# =========================================================

worker = threading.Thread(
    target=inference_loop,
    daemon=True
)

worker.start()

server = ThreadingHTTPServer(
    ("0.0.0.0", WEB_PORT),
    Handler
)

print()
print(
    "=========================================="
)
print(
    "Web demo running"
)
print(
    f"Port: {WEB_PORT}"
)
print(
    "=========================================="
)
print()
print(
    "Open the Raspberry Pi's IP address"
)
print(
    f"in a browser using port {WEB_PORT}."
)
print()
print(
    "Press Ctrl+C to stop."
)
print()


try:
    server.serve_forever()

except KeyboardInterrupt:
    print(
        "\nStopping..."
    )

finally:
    running = False

    server.shutdown()
    cap.release()

    print(
        "Finished."
    )


