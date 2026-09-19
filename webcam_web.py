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

# FILE MAP (for students):
# 1) Configuration constants (model path, camera size, port)
# 2) Webcam helper (setup_webcam)
# 3) Shared state (latest_jpeg, requested_device)
# 4) Callbacks used by the HTTP server (get_latest_jpeg, request_device_callback, is_running)
# 5) Inference loop (capture → infer → visualize → compress)
# 6) main() and the if __name__ == "__main__" guard (startup and shutdown)

import time
import threading
from http.server import ThreadingHTTPServer
from contextlib import contextmanager
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


# The model runner is created in main() so object lifetimes are easy to follow.


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
# Webcam setup happens in main() so the capture lives there
# =========================================================


@contextmanager
def open_webcam(camera_id: int, width: int, height: int, fps: int):
    """Context manager wrapper for the webcam capture.

    Usage:
        with open_webcam(CAMERA_ID, CAMERA_W, CAMERA_H, CAMERA_FPS) as cap:
            ... use cap ...

    Ensures cap.release() is always called, even if an exception occurs.
    """
    cap = setup_webcam(camera_id, width, height, fps)
    try:
        yield cap
    finally:
        cap.release()


# =========================================================
# Shared frame (threading notes for students)
# =========================================================
# The inference loop runs in a background thread and produces JPEG bytes that
# the HTTP server streams to browsers. We keep the "latest frame" in
# `latest_jpeg`. Access to this shared variable is guarded by `frame_lock` so
# the producer (inference thread) and consumers (HTTP threads) don't step on
# each other while reading/writing it.
#
# Why a lock?
# - While assigning a new Python bytes object reference is atomic, using a
#   small lock makes intent explicit and avoids subtle races if you later
#   extend the shared state (e.g., add timestamps or counters).
# - It's also a good first exposure to thread coordination for beginners.

latest_jpeg = None
frame_lock = threading.Lock()

running = True
# Device change requests are signaled by HTTP handler; applied safely in the inference thread
requested_device = None


## Pose estimation and result processing are handled in external modules


# =========================================================
# Callbacks for the HTTP server (top-level for readability)
# =========================================================
# A "callback" is a function we pass to another piece of code so that it can
# call us back later at the right moment. Keeping these at the top level makes
# them easier to find and read.

def get_latest_jpeg_callback():
    """Callback: return the most recent JPEG bytes for the /video MJPEG stream.

    Note:
    - May return None at startup until the first frame is processed.
    - The server writes whatever bytes it gets into the HTTP response.
    - We use a small lock to read a consistent value while the producer may
      be updating it in the background.
    """
    with frame_lock:
        return latest_jpeg


def request_device_callback(name: str):
    """Callback: record a requested device (CPU or MYRIAD) from the HTTP handler.

    Important:
    - We do NOT switch devices here. We only set a flag.
    - The inference loop (single writer) will see this flag between frames
      and call runner.set_device(name) at a safe moment.
    """
    global requested_device
    requested_device = name


def is_running_callback():
    """Callback: return True while the application is active.

    The streaming loop on the server side checks this to know when to stop
    sending frames (e.g., during shutdown after Ctrl+C).
    """
    return running


# =========================================================
# Inference thread
# =========================================================

def inference_loop(cap, runner):
    """Capture → infer → visualize → compress (JPEG) loop.

    Student roadmap for this loop:
    1) Read a frame from the webcam
    2) Check if a device switch was requested and apply it here (single writer)
    3) Ask the model runner to run inference and return results
    4) Draw the pose and friendly overlays using visualization helpers
     5) Compress the frame to JPEG (image compression, not a neural network)
         and publish it for the HTTP streamer
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

        device_name = res["device"]
        inference_ms = res["elapsed_ms"]

        # Draw the skeleton using the decoded keypoints (visualization step).
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

        # Compress the annotated frame to JPEG for streaming over HTTP
        success, jpeg = cv2.imencode(
            ".jpg",
            frame_for_processing,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if success:
            with frame_lock:
                latest_jpeg = jpeg.tobytes()


# =========================================================
# Start
# =========================================================
# We launch the background inference loop (produces JPEGs), then build an
# HTTP request handler by providing tiny callback functions below. This keeps
# the web server unaware of model internals and makes the wiring explicit.
#
# Threading primer (students):
# - We use ONE background thread for the model loop so the main thread can
#   focus on serving HTTP requests without blocking on inference.
# - The inference thread is marked daemon=True, which means it won't keep the
#   process alive on exit; when main finishes, the daemon thread will stop.
# - Shared state between threads is minimized: a single `latest_jpeg` buffer
#   guarded by a small lock, and a `requested_device` flag handled by the
#   single-writer pattern inside the inference loop.

def main():
    """Program entry point: set up worker thread and HTTP server once.

    Student note:
    - Putting startup code inside main() (and guarding with
      `if __name__ == "__main__":`) ensures this only runs once when you run
      the script, and not each time the module is imported elsewhere.
    """
    # Use a context manager so the camera is always released.
    with open_webcam(CAMERA_ID, CAMERA_W, CAMERA_H, CAMERA_FPS) as cap:
        # Create the model runner here so lifetimes are obvious (created → used → closed)
        runner = PoseModelRunner(
            MODEL_PATH,
            initial_device="MYRIAD",
            model_w=MODEL_W,
            model_h=MODEL_H,
        )

        # Start the background inference thread.
        # daemon=True: the thread will not prevent the program from exiting.
        worker = threading.Thread(target=inference_loop, args=(cap, runner), daemon=True)
        worker.start()

        base_dir = os.path.dirname(__file__)
        static_dir = os.path.join(base_dir, "static")
        Handler = create_handler(
            static_dir,
            get_latest_jpeg_callback,  # how to fetch the latest JPEG bytes
            request_device_callback,    # how to signal a requested device change
            is_running_callback,        # how to know when to stop streaming
        )

        # Use a context manager for the HTTP server so server_close() is
        # called automatically on exit. We still call shutdown() to stop the
        # serve_forever() loop cleanly before leaving the with-block.
        with ThreadingHTTPServer(("0.0.0.0", WEB_PORT), Handler) as server:
            # Output a friendly message to the console so users know where to point their browser.
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
                print("\nStopping...")
            finally:
                # Signal the worker to stop, then shut down the server.
                global running
                running = False
                server.shutdown()
                print("Finished.")


if __name__ == "__main__":
    main()


