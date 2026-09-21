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
# 2) Camera process (keeps the newest captured frame)
# 3) Shared state (latest_jpeg, requested_device, requested_heatmaps)
# 4) Callbacks used by the HTTP server (get_latest_jpeg_callback, request_device_callback, request_heatmaps_callback, is_running_callback)
# 5) Inference loop (reads newest captured frame, runs the model, draws overlays)
# 6) main() and the if __name__ == "__main__" guard (startup and shutdown)

import time
import threading
from http.server import ThreadingHTTPServer
from typing import Optional, Final
import os

import cv2
# STUDENT NOTE: Model EXECUTION is kept in PoseModelRunner (pose_model_runner.py).
# It returns raw results (points, heatmaps, device, elapsed_ms). Drawing lives in
# pose_result_processor.py, and the HTTP server lives in server_handler.py. This
# separation makes lessons clearer: execution vs visualization vs serving.
from pose_model_runner import PoseModelRunner
from pose_result_processor import (
    render_pose_on_frame,
    annotate_metrics,
    put_text_with_outline,
    heatmap_to_image,
    heatmaps_grid_to_image,
)
from pose_defs import BODY_PARTS
from camera_capture import ProcessCameraFrameGrabber
from server_handler import create_handler


# Pose estimation execution lives in PoseModelRunner (pose_model_runner.py)


# =========================================================
# Configuration
# =========================================================

MODEL_PATH: Final[str] = "model/human-pose-estimation-0001.xml"

CAMERA_ID: Final[int] = 0

MODEL_W: Final[int] = 456
MODEL_H: Final[int] = 256

CAMERA_W: Final[int] = 640
CAMERA_H: Final[int] = 480
CAMERA_FPS: Final[int] = 5 # 15

WEB_PORT: Final[int] = 8080

# Print one short timing report every few seconds for troubleshooting.
DIAGNOSTIC_REPORT_SECONDS: Final[float] = 5.0


# =========================================================
# Shared frame and sampling gate (threading notes for students)
# =========================================================
# The inference thread makes JPEG images. The web server threads send those
# images to the browser. Both parts share one variable: `latest_jpeg`.
#
# What is a race?
# - Two threads use the same data at the same time. Results can change based on
#   timing and may look broken.
#
# Why use a lock?
# - A lock is a simple gate. Only one thread can pass at a time.
# - We take the lock, read or write `latest_jpeg`, then release the lock.
# - This prevents mixed or half-updated data.
# - Python usually updates a bytes variable in one step, but we still use a
#   lock so the code stays safe if we add more shared values later.
#
# A separate camera process captures frames. The model loop processes the
# newest received frame and stores a JPEG.
# The browser stream simply shows the latest JPEG when it can.
#
# How to use the shared state:
# 1) frame_lock: protects the shared latest_jpeg bytes
# 2) read newest frame -> process -> publish JPEG -> continue

latest_jpeg: Optional[bytes] = None # Holds image bytes for the MJPEG stream. Updated by inference thread, read by HTTP threads.
frame_lock: threading.Lock = threading.Lock()

# Device change requests are signaled by HTTP handler; applied safely in the
# inference thread (single-writer pattern). We do NOT lock this flag because:
# - Only the handler thread writes the flag, and only the inference thread
#   reads-and-clears it between frames, so there is no concurrent write/write.
# - Assigning a small string reference is atomic in CPython. Even so, the
#   read-then-clear pattern guarantees we either see the request this frame or
#   the next one.
requested_device: Optional[str] = None

# None means "no new request". True means show the grid; False means hide it.
# The inference loop reads and clears this request between frames.
requested_heatmaps: Optional[bool] = None

running: bool = True

## Pose estimation and result processing are handled in external modules


# =========================================================
# Callbacks for the HTTP server (top-level for readability)
# =========================================================
# A "callback" is a function we pass to another piece of code so that it can
# call us back later at the right moment. Keeping these at the top level makes
# them easier to find and read.

def get_latest_jpeg_callback() -> Optional[bytes]:
    """Callback: return the most recent JPEG bytes for /video.

    Note:
    - May return None at startup until the first frame is processed.
    - The server writes whatever bytes it gets into the HTTP response.
    - The /video endpoint serves MJPEG (a stream of JPEG images).
        - We use the lock so we do not read while the background thread is writing.
    """
    with frame_lock:
        return latest_jpeg


def request_device_callback(name: str) -> None:
    """Callback: record a requested device (CPU or MYRIAD) from the HTTP handler.

    Important:
    - We do NOT switch devices here. We only set a flag.
    - The inference loop (single writer) will see this flag between frames
        and call set_device(name) on the model runner at a safe moment.
    """
    global requested_device  # Optional[str]
    requested_device = name


def request_heatmaps_callback(show: bool) -> None:
    """Callback: ask the inference loop to show or hide the heatmap grid."""
    global requested_heatmaps
    requested_heatmaps = show


def is_running_callback() -> bool:
    """Callback: return True while the application is active.

    The streaming loop on the server side checks this to know when to stop
    sending frames (e.g., during shutdown after Ctrl+C).
    """
    return running


# =========================================================
# Inference thread
# =========================================================

def inference_loop(capture_grabber: ProcessCameraFrameGrabber, model_runner: PoseModelRunner) -> None:
    """Infer → visualize → compress (JPEG) loop.

    Student roadmap for this loop:
    1) Wait until the capture helper has at least one frame
    2) Copy the newest camera frame
    3) Check if a device switch was requested and apply it here (single writer)
    4) Ask the model runner to run inference and return results
    5) Draw the pose and friendly overlays using visualization helpers
    6) Compress the frame to JPEG (image compression, not a neural network)
       and send it to the HTTP streamer
    """
    global latest_jpeg  # Optional[bytes]
    global requested_device  # Optional[str]
    global requested_heatmaps  # Optional[bool]

    show_heatmaps: bool = False  # Start hidden. The user can show them from the web page.
    device_switch_message: Optional[str] = None
    last_diagnostic_report_time = time.perf_counter()
    last_processed_frame_number = 0

    while running:
        # BUG FIX: Wait for a genuinely new camera frame.
        #
        # It is not enough to ask "Do we have a frame?" The answer stays yes
        # after the first frame. If the camera later becomes slow or times out,
        # that old frame would still exist and the model could process it again.
        #
        # Instead, we remember the number of the last frame we processed and
        # wait for the camera helper to give us a higher number. This means
        # one successful camera capture can be processed only once.
        if not capture_grabber.wait_for_new_frame(last_processed_frame_number, 0.1):
            continue

        # Copy the newest sampled frame so the capture thread can keep running.
        frame_for_processing, frame_capture_time, frame_number = capture_grabber.get_latest_frame()
        if frame_for_processing is None:
            continue
        # Remember this number before inference starts. The next loop must wait
        # for a newer frame number, even if the camera has a temporary timeout.
        last_processed_frame_number = frame_number

        # Apply any pending device switch request here.
        #
        # STUDENT NOTE — why do we switch devices in this thread?
        # ------------------------------------------------------
        # Our HTTP server can handle multiple requests at once (it starts a
        # new thread per request). The /device endpoint does NOT touch the
        # model directly; it only sets a simple flag: `requested_device`.
        #
        # This inference loop is the only place that changes the model.
        # The HTTP threads only set a request flag.
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
        device_switch_ms = 0.0
        if requested_device is not None:
            to_set = requested_device
            requested_device = None
            print(f"\nApplying device switch to: {to_set}...")
            device_switch_start_time = time.perf_counter()
            ok = model_runner.set_device(to_set)
            device_switch_ms = (time.perf_counter() - device_switch_start_time) * 1000.0
            device_switch_message = f"Timing reset after switching to {to_set}"
            print("Switch successful." if ok else "Switch failed.")

        # Apply a requested heatmap show/hide change between frames.
        if requested_heatmaps is not None:
            show_heatmaps = requested_heatmaps
            requested_heatmaps = None

        # Use the new PoseModelRunner to execute and obtain results.
        # Ask the model runner to execute the CNN on this frame.
        # NOTE (for students): 'res' is a small dictionary with multiple results:
        #   - res['points']   : decoded body keypoints you can draw
        #   - res['heatmaps'] : raw model heatmaps (useful for advanced lessons)
        #   - res['device']   : which device ran the model (CPU / MYRIAD)
        #   - res['frame']    : the same frame we passed in
        #   - res['elapsed_ms']: inference time in milliseconds
        model_call_start_time = time.perf_counter()
        frame_wait_before_model_ms = (model_call_start_time - frame_capture_time) * 1000.0
        res = model_runner.run_inference(frame_for_processing)
        inference_finished_time = time.perf_counter()
        full_model_call_ms = (inference_finished_time - model_call_start_time) * 1000.0

        device_name = res["device"]
        inference_ms = res["elapsed_ms"]

        # Draw the skeleton using the decoded keypoints (visualization step).
        points = res["points"]
        render_pose_on_frame(frame_for_processing, points)

        # How long this frame waited before it was ready to show.
        display_latency_ms = (time.perf_counter() - frame_capture_time) * 1000.0

        # The heatmap grid is optional. It starts hidden so the first view is
        # the simple camera image and stick figure.
        if show_heatmaps:
            # Remove batch dimension: [1, num_parts, h, w] -> [num_parts, h, w]
            raw_heatmaps = res["heatmaps"]
            heatmaps_3d = raw_heatmaps[0]
            heat_img = heatmaps_grid_to_image(
                heatmaps_3d,
                (frame_for_processing.shape[1], frame_for_processing.shape[0]),
                BODY_PARTS,
                cols=5,
            )

        # Overlay metrics text on the picture, via utility
        annotate_metrics(
            frame_for_processing,
            device_name,
            inference_ms,
            display_latency_ms,
        )

        # Show a short note on the first frame after a device switch.
        # This helps students see that the timing restarted.
        if device_switch_message is not None:
            put_text_with_outline(
                frame_for_processing,
                device_switch_message,
                (20, 180),
                0.6,
                (0, 255, 255),
                2,
            )
            device_switch_message = None

        # If enabled, put the heatmap grid to the right of the camera image.
        output_image = frame_for_processing
        if show_heatmaps:
            output_image = cv2.hconcat([frame_for_processing, heat_img])

        # Compress the output image to JPEG for streaming over HTTP.
        # This timing includes drawing overlays and optional heatmaps.
        post_processing_ms = (time.perf_counter() - inference_finished_time) * 1000.0
        jpeg_start_time = time.perf_counter()
        success, jpeg = cv2.imencode(
            ".jpg",
            output_image,
            [cv2.IMWRITE_JPEG_QUALITY, 90]  # 0-100, higher = better quality
        )
        jpeg_encoding_ms = (time.perf_counter() - jpeg_start_time) * 1000.0
        ready_to_send_delay_ms = (time.perf_counter() - frame_capture_time) * 1000.0

        if success:
            # Producer writes the most recent JPEG under the lock so readers
            # (HTTP threads) always see a consistent value.
            with frame_lock:
                latest_jpeg = jpeg.tobytes()

        # Print a short report periodically instead of printing once per frame.
        # This lets us compare camera work, model work, and the rest of the path.
        now = time.perf_counter()
        if now - last_diagnostic_report_time >= DIAGNOSTIC_REPORT_SECONDS:
            (
                capture_fps,
                average_read_ms,
                average_read_cpu_ms,
                maximum_read_ms,
                failed_reads,
                replaced_frames,
            ) = (
                capture_grabber.take_diagnostics()
            )
            newer_frames = capture_grabber.get_latest_frame_number() - frame_number
            load_average = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0

            print(
                "\n--- Pipeline diagnostic ---\n"
                f"Device: {device_name}\n"
                f"Camera: {capture_fps:.1f} frames/s, read average {average_read_ms:.1f} ms, "
                f"read CPU {average_read_cpu_ms:.1f} ms, read maximum {maximum_read_ms:.1f} ms, "
                f"failed reads {failed_reads}\n"
                f"Frame wait before model: {frame_wait_before_model_ms:.1f} ms\n"
                f"Model call: {full_model_call_ms:.1f} ms, network only: {inference_ms:.1f} ms\n"
                f"Device switch in this frame: {device_switch_ms:.1f} ms\n"
                f"After model: {post_processing_ms:.1f} ms, JPEG: {jpeg_encoding_ms:.1f} ms\n"
                f"Ready to send delay: {ready_to_send_delay_ms:.1f} ms\n"
                f"Newer frames captured during this model pass: {newer_frames}\n"
                f"Older frames replaced in the last report: {replaced_frames}\n"
                f"System load average (1 minute): {load_average:.2f}\n"
                "---------------------------"
            )
            last_diagnostic_report_time = now


# =========================================================
# Start
# =========================================================
# We launch the background Neural Network inference loop. The NN produces
# activation (heat) maps; we then decode them into human-readable keypoints,
# draw the skeleton/metrics on the frame, and finally compress the frame to
# a JPEG for streaming (MJPEG = a stream of JPEG images). After that, we build
# an HTTP handler that streams the latest JPEGs using the tiny callback
# functions below. This keeps
# the web server unaware of model internals and makes the wiring explicit.
#
# Concurrency primer (students):
# - The camera uses a separate process. This keeps slow CPU inference from
#   slowing down V4L2 camera reads in the main process.
# - The inference loop uses one daemon thread, so the main thread can serve
#   HTTP requests without waiting for inference to finish.
# - The camera process sends only its newest frame. Old queued frames are
#   replaced instead of forming a long queue.
# - Shared data in this process is minimized: one `latest_jpeg` buffer guarded
#   by a lock, plus simple device and heatmap request flags.

def main() -> None:
    """Program entry point: set up worker thread and HTTP server once.

    Student note:
    - Putting startup code inside main() (and guarding with
      `if __name__ == "__main__":`) ensures this only runs once when you run
      the script, and not each time the module is imported elsewhere.
    """
    # Create the model runner here so lifetimes are obvious (created -> used -> closed).
    model_runner: PoseModelRunner = PoseModelRunner(
        MODEL_PATH,
        initial_device="MYRIAD",
        model_w=MODEL_W,
        model_h=MODEL_H,
    )

    # The camera gets its own process. This keeps V4L2 camera reads separate
    # from slow CPU inference in this main process.
    capture_grabber = ProcessCameraFrameGrabber(
        CAMERA_ID,
        CAMERA_W,
        CAMERA_H,
        CAMERA_FPS,
    )
    capture_grabber.start()

    # Start the background inference thread in the main process.
    inference_thread: threading.Thread = threading.Thread(
        target=inference_loop, args=(capture_grabber, model_runner), daemon=True
    )
    inference_thread.start()

    base_dir = os.path.dirname(__file__)
    static_dir = os.path.join(base_dir, "static")
    Handler = create_handler(
        static_dir,
        get_latest_jpeg_callback,   # how to fetch the latest JPEG bytes
        request_device_callback,    # how to signal a requested device change
        request_heatmaps_callback,  # how to show or hide the heatmap grid
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
            # Signal workers to stop, then shut down the server.
            global running  # bool
            running = False
            capture_grabber.stop()
            server.shutdown()
            print("Finished.")


if __name__ == "__main__":
    main()


