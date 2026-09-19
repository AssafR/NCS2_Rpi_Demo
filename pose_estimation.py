import time
import threading
import numpy as np
import cv2
from openvino.runtime import Core


class PoseEstimator:
    # Pose model definitions
    BODY_PARTS = [
        "Nose", "Neck",
        "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist",
        "RHip", "RKnee", "RAnkle",
        "LHip", "LKnee", "LAnkle",
        "REye", "LEye", "REar", "LEar"
    ]

    POSE_PAIRS = [
        ("Neck", "RShoulder"),
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("Neck", "LShoulder"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("Neck", "RHip"),
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("Neck", "LHip"),
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
        ("Neck", "Nose"),
        ("Nose", "REye"),
        ("REye", "REar"),
        ("Nose", "LEye"),
        ("LEye", "LEar"),
    ]

    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self,
                 model_path: str,
                 initial_device: str = "MYRIAD",
                 model_w: int = 456,
                 model_h: int = 256,
                 camera_w: int = 640,
                 camera_h: int = 480,
                 camera_fps: int = 15):
        self.model_path = model_path
        self.model_w = model_w
        self.model_h = model_h
        self.camera_w = camera_w
        self.camera_h = camera_h
        self.camera_fps = camera_fps
        self.current_device = initial_device

        self.core = Core()
        self.model = self.core.read_model(self.model_path)
        self.compiled_model = None
        self.model_lock = threading.Lock()

        self.last_frame_time = time.perf_counter()
        self.smoothed_inference_ms = None
        self.smoothed_loop_fps = None

        # Caches
        self._prepared_dummy = np.zeros((1, 3, self.model_h, self.model_w), dtype=np.float32)

    def _compile_model(self, device: str):
        start = time.perf_counter()
        compiled = self.core.compile_model(self.model, device)
        elapsed = time.perf_counter() - start
        print(f"Compiled for {device} in {elapsed:.2f} s")
        # Warm-up
        compiled([self._prepared_dummy])
        return compiled

    def set_device(self, device: str) -> bool:
        if device not in ("CPU", "MYRIAD"):
            return False
        if device == self.current_device:
            return True
        try:
            new_compiled = self._compile_model(device)
        except Exception:
            return False
        with self.model_lock:
            self.compiled_model = new_compiled
            self.current_device = device
        return True

    def _prepare_frame(self, frame: object):
        resized = cv2.resize(frame, (self.model_w, self.model_h))
        tensor = resized.transpose(2, 0, 1)
        tensor = tensor[np.newaxis, ...]
        tensor = tensor.astype(np.float32)
        return tensor

    def _extract_keypoints(self, heatmaps, frame_width, frame_height):
        points = {}
        heatmaps = heatmaps[0]
        for i, part_name in enumerate(self.BODY_PARTS):
            heatmap = heatmaps[i]
            (_, confidence, _, point) = cv2.minMaxLoc(heatmap)
            if confidence >= self.CONFIDENCE_THRESHOLD:
                x = int(point[0] * frame_width / heatmap.shape[1])
                y = int(point[1] * frame_height / heatmap.shape[0])
                points[part_name] = (x, y, confidence)
            else:
                points[part_name] = None
        return points

    def _draw_pose(self, frame, points):
        for part_a, part_b in self.POSE_PAIRS:
            a = points.get(part_a)
            b = points.get(part_b)
            if a is not None and b is not None:
                cv2.line(frame, (a[0], a[1]), (b[0], b[1]), (0, 255, 255), 3)
        for point in points.values():
            if point is not None:
                cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)

    def _get_heatmaps(self, result):
        for output in self.compiled_model.outputs:
            shape = output.shape
            if len(shape) == 4 and shape[1] == 19:
                return result[output]
        raise RuntimeError("Could not find the 19-channel heatmap output")

    def infer(self, frame):
        if self.compiled_model is None:
            with self.model_lock:
                self.compiled_model = self._compile_model(self.current_device)

        tensor = self._prepare_frame(frame)

        start = time.perf_counter()
        with self.model_lock:
            result = self.compiled_model([tensor])
            heatmaps = self._get_heatmaps(result)
            device_name = self.current_device
        inference_ms = (time.perf_counter() - start) * 1000

        if self.smoothed_inference_ms is None:
            self.smoothed_inference_ms = inference_ms
        else:
            self.smoothed_inference_ms = (
                0.9 * self.smoothed_inference_ms
                + 0.1 * inference_ms
            )

        inf_fps = 1000.0 / self.smoothed_inference_ms if self.smoothed_inference_ms > 0 else 0.0

        h, w = frame.shape[:2]
        points = self._extract_keypoints(heatmaps, w, h)
        self._draw_pose(frame, points)

        now = time.perf_counter()
        loop_time = now - self.last_frame_time
        self.last_frame_time = now
        instant_loop_fps = 1.0 / loop_time if loop_time > 0 else 0.0
        if self.smoothed_loop_fps is None:
            self.smoothed_loop_fps = instant_loop_fps
        else:
            self.smoothed_loop_fps = (
                0.9 * self.smoothed_loop_fps
                + 0.1 * instant_loop_fps
            )

        cv2.putText(
            frame,
            f"Device: {device_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Inference: {self.smoothed_inference_ms:.0f} ms",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Inference FPS: {inf_fps:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Actual loop FPS: {self.smoothed_loop_fps:.2f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        success, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if success:
            return jpeg.tobytes(), device_name
        else:
            return None, device_name
