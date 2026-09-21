"""camera_capture.py
--------------------

Small helper for camera capture.

This file keeps the webcam grabbing code separate from the model loop and
the HTTP server so students can see one job per file:
- this file: read frames from the camera
- webcam_web.py: run the model and serve the page

The grabber keeps only the newest frame. If the model is slow, older frames
are replaced by newer ones instead of piling up in the app.
"""

import multiprocessing
import queue
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional, Tuple

import cv2


def setup_webcam(
    camera_id: int,
    width: int,
    height: int,
    fps: int,
    buffer_size: Optional[int] = None,
) -> cv2.VideoCapture:
    """Create and configure a VideoCapture for the given camera.

    This stays in the camera module so all webcam-specific setup is in one
    place for students.
    """
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    # Some cameras slow down when OpenCV requests a one-frame V4L2 buffer.
    # Leave the driver buffer alone by default. CameraFrameGrabber still keeps
    # only one newest frame inside this app, so old app-level frames are replaced.
    if buffer_size is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(
        "Webcam opened successfully. "
        f"Requested {width}x{height} at {fps} FPS; "
        f"camera reports {actual_width:.0f}x{actual_height:.0f} at {actual_fps:.1f} FPS."
    )
    return cap


@contextmanager
def open_webcam(
    camera_id: int,
    width: int,
    height: int,
    fps: int,
    buffer_size: Optional[int] = None,
) -> Generator[cv2.VideoCapture, None, None]:
    """Context manager wrapper for the webcam capture.

    Usage:
        with open_webcam(CAMERA_ID, CAMERA_W, CAMERA_H, CAMERA_FPS) as cap:
            ... use cap ...

    Ensures cap.release() is always called, even if an exception occurs.
    """
    cap = setup_webcam(camera_id, width, height, fps, buffer_size)
    try:
        yield cap
    finally:
        cap.release()


class CameraFrameGrabber:
    """Read camera frames in a background thread.

    The thread only grabs frames and stores the latest one. It does not run
    the model and it does not draw anything.

    Important idea for students:
    - We keep one newest frame, not a long list of frames.
    - Each successful camera read gets the next frame number: 1, 2, 3, ...
    - The model can use each frame number only once.

    The frame number is important if the camera is slow or times out. In that
    case, the old image is still stored here, but it is not a new image.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self.latest_frame: Optional[object] = None
        self.latest_capture_time: float = 0.0
        self.latest_frame_number: int = 0
        self._lock = threading.Lock()
        self._frame_condition = threading.Condition(self._lock)
        self._frame_available = threading.Event()
        self._running = True
        self._report_start_time = time.perf_counter()
        self._read_attempts_since_report = 0
        self._reads_since_report = 0
        self._failed_reads_since_report = 0
        self._read_time_total_ms = 0.0
        self._read_cpu_time_total_ms = 0.0
        self._read_time_max_ms = 0.0
        self._replaced_frames_since_report = 0

    def run(self) -> None:
        """Capture loop for the background thread."""
        while self._running:
            read_start_time = time.perf_counter()
            read_start_cpu_time = time.thread_time()
            ret, frame = self.cap.read()
            read_time_ms = (time.perf_counter() - read_start_time) * 1000.0
            read_cpu_time_ms = (time.thread_time() - read_start_cpu_time) * 1000.0

            with self._lock:
                self._read_attempts_since_report += 1
                self._read_time_total_ms += read_time_ms
                self._read_cpu_time_total_ms += read_cpu_time_ms
                self._read_time_max_ms = max(self._read_time_max_ms, read_time_ms)

            if not ret:
                with self._lock:
                    self._failed_reads_since_report += 1
                time.sleep(0.05)
                continue

            capture_time = time.perf_counter()
            with self._frame_condition:
                if self.latest_frame is not None:
                    self._replaced_frames_since_report += 1
                self.latest_frame = frame
                self.latest_capture_time = capture_time
                self.latest_frame_number += 1
                self._reads_since_report += 1
                self._frame_available.set()
                self._frame_condition.notify_all()

    def stop(self) -> None:
        """Tell the background thread to stop."""
        self._running = False
        self._frame_available.set()

    def wait_for_frame(self, timeout: float = 0.1) -> bool:
        """Wait until at least one frame has been captured."""
        return self._frame_available.wait(timeout)

    def wait_for_new_frame(self, previous_frame_number: int, timeout: float = 0.1) -> bool:
        """Wait until the camera captures a frame newer than the previous one.

        This fixes a stale-frame bug:
        an "image is available" signal stays true after the first image.
        If the camera later times out, that signal alone could make the model
        process the same old image again and again. Frame numbers let us ask
        the clearer question: "Did a new camera image arrive?"
        """
        with self._frame_condition:
            return self._frame_condition.wait_for(
                lambda: self.latest_frame_number > previous_frame_number or not self._running,
                timeout,
            )

    def get_latest_frame(self) -> Tuple[Optional[object], float, int]:
        """Return a copy of the latest frame, its capture time, and its number."""
        with self._lock:
            if self.latest_frame is None:
                return None, 0.0, 0
            return self.latest_frame.copy(), self.latest_capture_time, self.latest_frame_number

    def get_latest_frame_number(self) -> int:
        """Return the number of the newest frame without copying the image."""
        with self._lock:
            return self.latest_frame_number

    def take_diagnostics(self) -> Tuple[float, float, float, float, int, int]:
        """Return capture statistics since the previous diagnostic report.

        Returns: successful capture frames per second, average read time in ms,
        average CPU time used during a read, maximum read time in ms, failed
        read attempts, and replaced older frames.
        """
        with self._lock:
            elapsed_seconds = time.perf_counter() - self._report_start_time
            capture_fps = self._reads_since_report / elapsed_seconds if elapsed_seconds > 0 else 0.0
            average_read_ms = (
                self._read_time_total_ms / self._read_attempts_since_report
                if self._read_attempts_since_report > 0
                else 0.0
            )
            average_read_cpu_ms = (
                self._read_cpu_time_total_ms / self._read_attempts_since_report
                if self._read_attempts_since_report > 0
                else 0.0
            )
            maximum_read_ms = self._read_time_max_ms
            failed_reads = self._failed_reads_since_report
            replaced_frames = self._replaced_frames_since_report

            self._report_start_time = time.perf_counter()
            self._read_attempts_since_report = 0
            self._reads_since_report = 0
            self._failed_reads_since_report = 0
            self._read_time_total_ms = 0.0
            self._read_cpu_time_total_ms = 0.0
            self._read_time_max_ms = 0.0
            self._replaced_frames_since_report = 0

            return (
                capture_fps,
                average_read_ms,
                average_read_cpu_ms,
                maximum_read_ms,
                failed_reads,
                replaced_frames,
            )


def _put_newest_frame(frame_queue, frame_data) -> int:
    """Put a frame into a one-item queue, dropping an older queued frame.

    Returns how many old queued frames were dropped. The capture process must
    never wait for inference to finish before it can read the next camera frame.
    """
    dropped_frames = 0
    while True:
        try:
            frame_queue.put_nowait(frame_data)
            return dropped_frames
        except queue.Full:
            try:
                frame_queue.get_nowait()
                dropped_frames += 1
            except queue.Empty:
                pass


def _capture_process_worker(
    stop_event,
    frame_queue,
    diagnostic_queue,
    camera_id: int,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Own the webcam in a separate process and send only the newest frame."""
    cap = setup_webcam(camera_id, width, height, fps)
    frame_number = 0
    report_start_time = time.perf_counter()
    read_attempts = 0
    successful_reads = 0
    failed_reads = 0
    total_read_ms = 0.0
    total_read_cpu_ms = 0.0
    maximum_read_ms = 0.0
    replaced_frames = 0

    try:
        while not stop_event.is_set():
            read_start_time = time.perf_counter()
            read_start_cpu_time = time.thread_time()
            ok, frame = cap.read()
            read_ms = (time.perf_counter() - read_start_time) * 1000.0
            read_cpu_ms = (time.thread_time() - read_start_cpu_time) * 1000.0

            read_attempts += 1
            total_read_ms += read_ms
            total_read_cpu_ms += read_cpu_ms
            maximum_read_ms = max(maximum_read_ms, read_ms)

            if ok:
                successful_reads += 1
                frame_number += 1
                capture_time = time.perf_counter()
                replaced_frames += _put_newest_frame(
                    frame_queue,
                    (frame, capture_time, frame_number),
                )
            else:
                failed_reads += 1
                time.sleep(0.05)

            now = time.perf_counter()
            if now - report_start_time >= 5.0:
                elapsed_seconds = now - report_start_time
                average_read_ms = total_read_ms / read_attempts if read_attempts else 0.0
                average_read_cpu_ms = total_read_cpu_ms / read_attempts if read_attempts else 0.0
                diagnostic_queue.put(
                    (
                        successful_reads / elapsed_seconds,
                        average_read_ms,
                        average_read_cpu_ms,
                        maximum_read_ms,
                        failed_reads,
                        replaced_frames,
                    )
                )
                report_start_time = now
                read_attempts = 0
                successful_reads = 0
                failed_reads = 0
                total_read_ms = 0.0
                total_read_cpu_ms = 0.0
                maximum_read_ms = 0.0
                replaced_frames = 0
    finally:
        cap.release()


class ProcessCameraFrameGrabber:
    """Read webcam frames in a process separate from model inference.

    A CPU model can keep the main Python process busy for a long time. This
    class gives the camera its own process, so V4L2 capture continues while
    inference runs elsewhere.

    The processes use a one-item queue. If inference is slow, a newer frame
    replaces an older queued frame. We keep the newest view, not a backlog.
    """

    def __init__(self, camera_id: int, width: int, height: int, fps: int) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._context = multiprocessing.get_context("spawn")
        self._stop_event = self._context.Event()
        self._frame_queue = self._context.Queue(maxsize=1)
        self._diagnostic_queue = self._context.Queue()
        self._process = None
        self._latest_frame: Optional[object] = None
        self._latest_capture_time = 0.0
        self._latest_frame_number = 0
        self._latest_diagnostics = (0.0, 0.0, 0.0, 0.0, 0, 0)

    def start(self) -> None:
        """Start the separate camera process."""
        self._process = self._context.Process(
            target=_capture_process_worker,
            args=(
                self._stop_event,
                self._frame_queue,
                self._diagnostic_queue,
                self.camera_id,
                self.width,
                self.height,
                self.fps,
            ),
        )
        self._process.start()

    def stop(self) -> None:
        """Stop the camera process and wait briefly for it to exit."""
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()

    def _collect_new_messages(self) -> None:
        """Copy queued camera frames and diagnostics into this process."""
        while True:
            try:
                frame, capture_time, frame_number = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            self._latest_frame = frame
            self._latest_capture_time = capture_time
            self._latest_frame_number = frame_number

        while True:
            try:
                self._latest_diagnostics = self._diagnostic_queue.get_nowait()
            except queue.Empty:
                break

    def wait_for_new_frame(self, previous_frame_number: int, timeout: float = 0.1) -> bool:
        """Wait until the camera process sends a frame with a higher number."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            self._collect_new_messages()
            if self._latest_frame_number > previous_frame_number:
                return True
            time.sleep(0.01)
        return False

    def get_latest_frame(self) -> Tuple[Optional[object], float, int]:
        """Return a safe copy of the newest frame and its capture information."""
        self._collect_new_messages()
        if self._latest_frame is None:
            return None, 0.0, 0
        return self._latest_frame.copy(), self._latest_capture_time, self._latest_frame_number

    def get_latest_frame_number(self) -> int:
        """Return the newest received frame number without copying its image."""
        self._collect_new_messages()
        return self._latest_frame_number

    def take_diagnostics(self) -> Tuple[float, float, float, float, int, int]:
        """Return the most recent five-second camera-process diagnostic report."""
        self._collect_new_messages()
        return self._latest_diagnostics