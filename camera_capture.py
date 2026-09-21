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
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CaptureReport:
    """One readable summary of camera capture work during a time window."""

    capture_fps: float = 0.0
    average_read_ms: float = 0.0
    average_read_cpu_ms: float = 0.0
    maximum_read_ms: float = 0.0
    average_loop_gap_ms: float = 0.0
    average_lock_wait_ms: float = 0.0
    average_shared_copy_ms: float = 0.0
    failed_reads: int = 0
    replaced_frames: int = 0


@dataclass
class CaptureDiagnostics:
    """Collect capture timings until the program asks for a report.

    Keeping these counters together makes it clear which measurements belong
    to one camera-report window.
    """

    report_start_time: float = field(default_factory=time.perf_counter)
    read_attempts: int = 0
    successful_reads: int = 0
    failed_reads: int = 0
    total_read_ms: float = 0.0
    total_read_cpu_ms: float = 0.0
    maximum_read_ms: float = 0.0
    total_loop_gap_ms: float = 0.0
    loop_gap_count: int = 0
    total_lock_wait_ms: float = 0.0
    total_shared_copy_ms: float = 0.0
    replaced_frames: int = 0

    def record_read(self, read_ms: float, read_cpu_ms: float, loop_gap_ms: float = 0.0) -> None:
        """Record one call to cap.read()."""
        self.read_attempts += 1
        self.total_read_ms += read_ms
        self.total_read_cpu_ms += read_cpu_ms
        self.maximum_read_ms = max(self.maximum_read_ms, read_ms)
        if loop_gap_ms > 0.0:
            self.total_loop_gap_ms += loop_gap_ms
            self.loop_gap_count += 1

    def record_success(
        self,
        lock_wait_ms: float = 0.0,
        shared_copy_ms: float = 0.0,
        replaced_frame: bool = False,
    ) -> None:
        """Record one successful frame capture."""
        self.successful_reads += 1
        self.total_lock_wait_ms += lock_wait_ms
        self.total_shared_copy_ms += shared_copy_ms
        if replaced_frame:
            self.replaced_frames += 1

    def record_failure(self) -> None:
        """Record one failed frame capture."""
        self.failed_reads += 1

    def take_report(self) -> CaptureReport:
        """Return the current summary and start a fresh measurement window."""
        elapsed_seconds = time.perf_counter() - self.report_start_time
        report = CaptureReport(
            capture_fps=self.successful_reads / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            average_read_ms=self.total_read_ms / self.read_attempts if self.read_attempts else 0.0,
            average_read_cpu_ms=self.total_read_cpu_ms / self.read_attempts if self.read_attempts else 0.0,
            maximum_read_ms=self.maximum_read_ms,
            average_loop_gap_ms=(
                self.total_loop_gap_ms / self.loop_gap_count if self.loop_gap_count else 0.0
            ),
            average_lock_wait_ms=(
                self.total_lock_wait_ms / self.successful_reads if self.successful_reads else 0.0
            ),
            average_shared_copy_ms=(
                self.total_shared_copy_ms / self.successful_reads if self.successful_reads else 0.0
            ),
            failed_reads=self.failed_reads,
            replaced_frames=self.replaced_frames,
        )
        self.reset()
        return report

    def reset(self) -> None:
        """Clear counters and begin a new diagnostic time window."""
        self.report_start_time = time.perf_counter()
        self.read_attempts = 0
        self.successful_reads = 0
        self.failed_reads = 0
        self.total_read_ms = 0.0
        self.total_read_cpu_ms = 0.0
        self.maximum_read_ms = 0.0
        self.total_loop_gap_ms = 0.0
        self.loop_gap_count = 0
        self.total_lock_wait_ms = 0.0
        self.total_shared_copy_ms = 0.0
        self.replaced_frames = 0


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
        self._diagnostics = CaptureDiagnostics()

    def run(self) -> None:
        """Capture loop for the background thread."""
        while self._running:
            read_start_time = time.perf_counter()
            read_start_cpu_time = time.thread_time()
            ret, frame = self.cap.read()
            read_time_ms = (time.perf_counter() - read_start_time) * 1000.0
            read_cpu_time_ms = (time.thread_time() - read_start_cpu_time) * 1000.0

            with self._lock:
                self._diagnostics.record_read(read_time_ms, read_cpu_time_ms)

            if not ret:
                with self._lock:
                    self._diagnostics.record_failure()
                time.sleep(0.05)
                continue

            capture_time = time.perf_counter()
            with self._frame_condition:
                replaced_frame = self.latest_frame is not None
                self.latest_frame = frame
                self.latest_capture_time = capture_time
                self.latest_frame_number += 1
                self._diagnostics.record_success(replaced_frame=replaced_frame)
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

    def take_diagnostics(self) -> CaptureReport:
        """Return capture statistics since the previous diagnostic report.

        The thread version has no shared-memory handoff, so its loop-gap,
        lock-wait, and frame-copy values stay zero.
        """
        with self._lock:
            return self._diagnostics.take_report()


def _capture_process_worker(
    stop_event,
    shared_memory_name: str,
    frame_lock,
    frame_number,
    capture_time,
    diagnostic_queue,
    camera_id: int,
    width: int,
    height: int,
    fps: int,
) -> None:
    """Own the webcam in a separate process and update the newest frame slot."""
    cap = setup_webcam(camera_id, width, height, fps)
    shared_frame_memory = shared_memory.SharedMemory(name=shared_memory_name)
    shared_frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shared_frame_memory.buf)
    diagnostics = CaptureDiagnostics()
    has_previous_frame = False
    previous_read_end_time = None

    try:
        while not stop_event.is_set():
            read_start_time = time.perf_counter()
            loop_gap_ms = 0.0
            if previous_read_end_time is not None:
                loop_gap_ms = (read_start_time - previous_read_end_time) * 1000.0
            read_start_cpu_time = time.thread_time()
            ok, frame = cap.read()
            read_end_time = time.perf_counter()
            read_ms = (read_end_time - read_start_time) * 1000.0
            read_cpu_ms = (time.thread_time() - read_start_cpu_time) * 1000.0
            previous_read_end_time = read_end_time

            diagnostics.record_read(read_ms, read_cpu_ms, loop_gap_ms)

            if ok:
                if frame.shape != shared_frame.shape:
                    raise RuntimeError(
                        f"Camera returned {frame.shape}, expected {shared_frame.shape}."
                    )
                lock_wait_start_time = time.perf_counter()
                with frame_lock:
                    lock_wait_ms = (time.perf_counter() - lock_wait_start_time) * 1000.0
                    shared_copy_start_time = time.perf_counter()
                    shared_frame[:, :, :] = frame
                    shared_copy_ms = (time.perf_counter() - shared_copy_start_time) * 1000.0
                    frame_number.value += 1
                    capture_time.value = time.perf_counter()
                    diagnostics.record_success(
                        lock_wait_ms,
                        shared_copy_ms,
                        replaced_frame=has_previous_frame,
                    )
                    has_previous_frame = True
            else:
                diagnostics.record_failure()
                time.sleep(0.05)

            if time.perf_counter() - diagnostics.report_start_time >= 5.0:
                diagnostic_queue.put(diagnostics.take_report())
    finally:
        cap.release()
        shared_frame_memory.close()


class ProcessCameraFrameGrabber:
    """Read webcam frames in a process separate from model inference.

    A CPU model can keep the main Python process busy for a long time. This
    class gives the camera its own process, so V4L2 capture continues while
    inference runs elsewhere.

    The processes share one frame-sized memory slot. If inference is slow, the
    camera writes a newer frame over the old one. We keep the newest view, not
    a backlog, and avoid copying large images through a multiprocessing queue.
    """

    def __init__(self, camera_id: int, width: int, height: int, fps: int) -> None:
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._context = multiprocessing.get_context("spawn")
        self._stop_event = self._context.Event()
        self._frame_lock = self._context.Lock()
        self._frame_number = self._context.Value("L", 0, lock=False)
        self._capture_time = self._context.Value("d", 0.0, lock=False)
        self._shared_frame_memory = shared_memory.SharedMemory(
            create=True,
            size=width * height * 3,
        )
        self._shared_frame = np.ndarray(
            (height, width, 3),
            dtype=np.uint8,
            buffer=self._shared_frame_memory.buf,
        )
        self._diagnostic_queue = self._context.Queue()
        self._process = None
        self._latest_diagnostics = CaptureReport()

    def start(self) -> None:
        """Start the separate camera process."""
        self._process = self._context.Process(
            target=_capture_process_worker,
            args=(
                self._stop_event,
                self._shared_frame_memory.name,
                self._frame_lock,
                self._frame_number,
                self._capture_time,
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
                self._process.join()
        self._shared_frame_memory.close()
        self._shared_frame_memory.unlink()

    def _collect_new_messages(self) -> None:
        """Copy diagnostic reports from the camera process into this process."""
        while True:
            try:
                self._latest_diagnostics = self._diagnostic_queue.get_nowait()
            except queue.Empty:
                break

    def wait_for_new_frame(self, previous_frame_number: int, timeout: float = 0.1) -> bool:
        """Wait until the camera process sends a frame with a higher number."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._frame_lock:
                current_frame_number = self._frame_number.value
            if current_frame_number > previous_frame_number:
                return True
            time.sleep(0.01)
        return False

    def get_latest_frame(self) -> Tuple[Optional[object], float, int]:
        """Return a safe copy of the newest frame and its capture information."""
        self._collect_new_messages()
        with self._frame_lock:
            frame_number = self._frame_number.value
            if frame_number == 0:
                return None, 0.0, 0
            return self._shared_frame.copy(), self._capture_time.value, frame_number

    def get_latest_frame_number(self) -> int:
        """Return the newest received frame number without copying its image."""
        with self._frame_lock:
            return self._frame_number.value

    def take_diagnostics(self) -> CaptureReport:
        """Return the most recent five-second camera-process diagnostic report."""
        self._collect_new_messages()
        return self._latest_diagnostics