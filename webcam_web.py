"""
webcam_web.py
-------------

High-level script that:
- Captures frames from the camera
- Asks the PoseModelRunner to run the model
- Uses pose_result_processor utilities to draw results and overlays
- Streams the annotated frames via an HTTP server (created by server_handler)

This file intentionally avoids model internals and HTML content to keep the
lesson focused on "glue code" and system wiring.
"""

import time
import threading
from http.server import ThreadingHTTPServer
import os

import cv2
# STUDENT NOTE: Model EXECUTION is kept in PoseModelRunner (pose_model_runner.py).
# It returns raw results (points, heatmaps, device, elapsed_ms). Drawing lives in
# pose_result_processor.py, and the HTTP server lives in server_handler.py. This
# separation makes lessons clearer: execution vs visualization vs serving.
from pose_model_runner import PoseModelRunner
from pose_result_processor import render_pose_on_frame, annotate_metrics
from server_handler import create_handler


# Pose estimation execution lives in PoseModelRunner (pose_model_runner.py)


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
)


# =========================================================
# Webcam setup helper
# =========================================================

def setup_webcam(camera_id: int, width: int, height: int, fps: int):
    """Create and configure a VideoCapture for the given camera.

    For students: this is separate from the inference loop so you can clearly
    see the camera configuration in one place.
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    print("Webcam opened successfully.")
    return cap


# =========================================================
# Webcam setup
# =========================================================

cap = setup_webcam(CAMERA_ID, CAMERA_W, CAMERA_H, CAMERA_FPS)


# =========================================================
# Shared frame
# =========================================================

latest_jpeg = None
frame_lock = threading.Lock()

running = True
# Device change requests are signaled by HTTP handler; applied safely in the inference thread
requested_device = None


## Pose estimation and result processing are handled in external modules


# =========================================================
# Inference thread
# =========================================================

def inference_loop():
    """Capture → infer → visualize → encode loop.

    Student roadmap for this loop:
    1) Read a frame from the webcam
    2) Check if a device switch was requested and apply it here (single writer)
    3) Ask the model runner to run inference and return results
    4) Draw the pose and friendly overlays using visualization helpers
    5) Encode to JPEG and publish for the HTTP streamer
    """
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
        #
        # STUDENT NOTE — why do we switch devices in this thread?
        # ------------------------------------------------------
        # Our HTTP server can handle multiple requests at once (it starts a
        # new thread per request). The /device endpoint does NOT touch the
        # model directly; it only sets a simple flag: `requested_device`.
        #
        # This inference loop is the ONLY place that actually changes the
        # model/device. That means there is ONE WRITER (this loop) and many
        # READERS (HTTP threads that just stream the latest JPEG). Keeping a
        # single writer prevents "race conditions" (two threads changing the
        # model at the same time) and avoids complicated locks.
        #
        # Flow:
        #   1) A user clicks a button in the web page -> /device?name=CPU
        #   2) The HTTP thread sets `requested_device = "CPU"` and returns
        #   3) This loop sees `requested_device`, performs the safe switch
        #      between frames (not in the middle of inference), and clears it
        #
        # This is a simple and safe pattern for beginners because:
        #   - No model swapping while a frame is being processed
        #   - No complicated locking is needed at the server level
        #   - Easy to reason about: "HTTP requests only set a flag; the loop
        #     applies the change at a good time"
        if requested_device is not None:
            to_set = requested_device
            requested_device = None
            print(f"\nApplying device switch to: {to_set}...")
            ok = runner.set_device(to_set)
            print("Switch successful." if ok else "Switch failed.")

        tensor, frame_for_processing = None, frame
        # Use the new PoseModelRunner to execute and obtain results
        # Ask the model runner to execute the CNN on this frame.
        # NOTE (for students): 'res' is a small dictionary with multiple results:
        #   - res['points']   : decoded body keypoints you can draw
        #   - res['heatmaps'] : raw model heatmaps (useful for advanced lessons)
        #   - res['device']   : which device ran the model (CPU / MYRIAD)
        #   - res['frame']    : the same frame we passed in
        #   - res['elapsed_ms']: inference time in milliseconds
        res = runner.run(frame_for_processing)

        # Draw the skeleton using the decoded keypoints (visualization step).
        device_name = res["device"]
        inference_ms = res["elapsed_ms"]

        points = res["points"]
        render_pose_on_frame(frame_for_processing, points)
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

        # Overlay metrics via utility
        annotate_metrics(
            frame_for_processing,
            device_name,
            inference_ms,
            smoothed_loop_fps,
        )

        success, jpeg = cv2.imencode(
            ".jpg",
            frame_for_processing,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if success:
            with frame_lock:
                latest_jpeg = jpeg.tobytes()


## HTML UI is now served from the static directory


## HTTP request handling moved to server_handler.create_handler


# =========================================================
# Start
# =========================================================

worker = threading.Thread(
    target=inference_loop,
    daemon=True
)

worker.start()

def get_latest_jpeg():
    """Return the latest encoded JPEG frame for the /video stream."""
    with frame_lock:
        return latest_jpeg

def request_device_callback(name: str):
    """Signal the inference loop to switch device at a safe point."""
    global requested_device
    requested_device = name

def is_running():
    """Tell the HTTP streamer whether to continue sending frames."""
    return running

base_dir = os.path.dirname(__file__)
static_dir = os.path.join(base_dir, "static")
Handler = create_handler(static_dir, get_latest_jpeg, request_device_callback, is_running)

server = ThreadingHTTPServer(("0.0.0.0", WEB_PORT), Handler)

print(
    f"""
==========================================
Web demo running
Port: {WEB_PORT}
==========================================

Open the Raspberry Pi's IP address
in a browser using port {WEB_PORT}.

Press Ctrl+C to stop.
"""
)


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


