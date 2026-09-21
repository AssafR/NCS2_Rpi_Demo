"""diagnose_capture_process.py
--------------------------------

Test whether model inference interferes with webcam capture.

This diagnostic uses two processes:
- Camera process: reads the webcam and reports only capture timing.
- Model process: runs repeated CPU or MYRIAD inference on a test image.

Why use two processes instead of two threads?
Two processes are more separate. If camera capture stays near its requested
rate here, but becomes slow in webcam_web.py, the same-process thread setup is
likely part of the problem.

This is a diagnostic only. It does not stream video or draw pose results.
"""

import argparse
import multiprocessing
import queue
import time

import numpy as np

from camera_capture import setup_webcam
from pose_model_runner import PoseModelRunner


MODEL_PATH = "model/human-pose-estimation-0001.xml"
MODEL_W = 456
MODEL_H = 256
CAMERA_W = 640
CAMERA_H = 480
CAMERA_FPS = 5
REPORT_SECONDS = 5.0


def camera_worker(stop_event, report_queue, camera_id, width, height, fps, buffer_size) -> None:
    """Read the camera and periodically send timing summaries to the parent."""
    cap = setup_webcam(camera_id, width, height, fps, buffer_size)
    report_start_time = time.perf_counter()
    successful_reads = 0
    failed_reads = 0
    total_read_ms = 0.0
    maximum_read_ms = 0.0

    try:
        while not stop_event.is_set():
            read_start_time = time.perf_counter()
            ok, _frame = cap.read()
            read_ms = (time.perf_counter() - read_start_time) * 1000.0

            total_read_ms += read_ms
            maximum_read_ms = max(maximum_read_ms, read_ms)
            if ok:
                successful_reads += 1
            else:
                failed_reads += 1
                time.sleep(0.05)

            now = time.perf_counter()
            if now - report_start_time >= REPORT_SECONDS:
                elapsed_seconds = now - report_start_time
                total_attempts = successful_reads + failed_reads
                average_read_ms = total_read_ms / total_attempts if total_attempts else 0.0
                report_queue.put(
                    (
                        successful_reads / elapsed_seconds,
                        average_read_ms,
                        maximum_read_ms,
                        failed_reads,
                    )
                )
                report_start_time = now
                successful_reads = 0
                failed_reads = 0
                total_read_ms = 0.0
                maximum_read_ms = 0.0
    finally:
        cap.release()


def print_camera_reports(report_queue) -> None:
    """Print every capture report the camera process has sent so far."""
    while True:
        try:
            capture_fps, average_read_ms, maximum_read_ms, failed_reads = report_queue.get_nowait()
        except queue.Empty:
            return

        print(
            "Camera process: "
            f"{capture_fps:.2f} FPS, read average {average_read_ms:.1f} ms, "
            f"read maximum {maximum_read_ms:.1f} ms, failed reads {failed_reads}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test separate-process camera capture while model inference runs."
    )
    parser.add_argument(
        "--device",
        choices=("CPU", "MYRIAD", "NONE"),
        default="CPU",
        help="NONE runs the same OpenCV capture code without model inference.",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument(
        "--no-buffer-limit",
        action="store_true",
        help="Do not request OpenCV/V4L2's one-frame buffer limit.",
    )
    args = parser.parse_args()

    buffer_size = None if args.no_buffer_limit else 1
    model_runner = None
    test_frame = None
    if args.device != "NONE":
        # Use a simple black image. We measure inference work, not pose accuracy.
        test_frame = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        model_runner = PoseModelRunner(
            MODEL_PATH,
            initial_device=args.device,
            model_w=MODEL_W,
            model_h=MODEL_H,
        )
        print(f"Warming up the model on {args.device} before starting the camera test...")
        model_runner.run_inference(test_frame)

    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    report_queue = context.Queue()
    camera_process = context.Process(
        target=camera_worker,
        args=(stop_event, report_queue, 0, CAMERA_W, CAMERA_H, CAMERA_FPS, buffer_size),
    )
    camera_process.start()

    if args.device == "NONE":
        print(f"Running separate-process capture only for {args.seconds:.0f} seconds.")
    else:
        print(f"Running {args.device} inference and separate-process capture for {args.seconds:.0f} seconds.")
    inference_count = 0
    inference_total_ms = 0.0
    report_time = time.perf_counter()
    finish_time = report_time + args.seconds

    try:
        while time.perf_counter() < finish_time:
            if model_runner is None:
                time.sleep(0.05)
            else:
                result = model_runner.run_inference(test_frame)
                inference_count += 1
                inference_total_ms += result["elapsed_ms"]

            now = time.perf_counter()
            if now - report_time >= REPORT_SECONDS:
                if model_runner is not None:
                    average_model_ms = inference_total_ms / inference_count if inference_count else 0.0
                    print(f"Model process: {inference_count} inferences, average {average_model_ms:.1f} ms")
                print_camera_reports(report_queue)
                inference_count = 0
                inference_total_ms = 0.0
                report_time = now
    finally:
        stop_event.set()
        camera_process.join(timeout=2.0)
        if camera_process.is_alive():
            camera_process.terminate()
        print_camera_reports(report_queue)


if __name__ == "__main__":
    main()