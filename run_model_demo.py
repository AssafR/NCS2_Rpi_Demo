"""
run_model_demo.py
-----------------

Beginner-friendly, minimal demo that runs the pose-estimation model on the
webcam and shows the result in a window (no web server).

- Model execution: PoseModelRunner (pose_model_runner.py)
- Visualization: render_pose_on_frame + annotate_metrics (pose_result_processor.py)

Press ESC to quit.
"""

import time
import cv2
from contextlib import contextmanager
from pose_model_runner import PoseModelRunner
from pose_result_processor import render_pose_on_frame, annotate_metrics

# --- Simple webcam setup (same parameters as the web demo) ---
CAMERA_ID = 0
CAMERA_W = 640
CAMERA_H = 480
CAMERA_FPS = 15

MODEL_PATH = "model/human-pose-estimation-0001.xml"
MODEL_W = 456
MODEL_H = 256


def setup_webcam(camera_id: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    return cap


@contextmanager
def open_webcam(camera_id: int, width: int, height: int, fps: int):
    """Context manager so the camera is always released.

    Usage:
        with open_webcam(CAMERA_ID, CAMERA_W, CAMERA_H, CAMERA_FPS) as cap:
            ... use cap ...
    """
    cap = setup_webcam(camera_id, width, height, fps)
    try:
        yield cap
    finally:
        cap.release()


def main():
    runner = PoseModelRunner(
        MODEL_PATH,
        initial_device="MYRIAD",  # try "CPU" if you don’t have NCS2
        model_w=MODEL_W,
        model_h=MODEL_H,
    )

    smoothed_loop_fps = None
    last_t = time.perf_counter()

    # Use a with-block so the camera is always closed
    with open_webcam(CAMERA_ID, CAMERA_W, CAMERA_H, CAMERA_FPS) as cap:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to capture webcam frame")
                time.sleep(0.1)
                continue

            # Run the model and get structured results
            res = runner.run(frame)
            device_name = res["device"]
            inference_ms = res["elapsed_ms"]
            points = res["points"]

            # Draw pose and friendly overlays
            render_pose_on_frame(frame, points)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            inst_loop_fps = 1.0 / dt if dt > 0 else 0.0
            if smoothed_loop_fps is None:
                smoothed_loop_fps = inst_loop_fps
            else:
                smoothed_loop_fps = 0.9 * smoothed_loop_fps + 0.1 * inst_loop_fps

            annotate_metrics(frame, device_name, inference_ms, smoothed_loop_fps)

            cv2.imshow("Pose Estimation (ESC to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
