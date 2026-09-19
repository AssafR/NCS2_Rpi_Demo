import time
import threading
import numpy as np
import cv2
from openvino.runtime import Core


class PoseModelRunner:
    # Pose model definitions (mirrored from the original file)
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

        # Tracking for performance metrics (optional)
        self.last_frame_time = time.perf_counter()

        # Warm-up dummy tensor (not strictly necessary here but keeps parity)
        self._prepared_dummy = np.zeros((1, 3, self.model_h, self.model_w), dtype=np.float32)

    def _compile_model(self, device: str):
        start = time.perf_counter()
        compiled = self.core.compile_model(self.model, device)
        elapsed = time.perf_counter() - start
        print(f"Compiled for {device} in {elapsed:.2f} s")
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

    def _get_heatmaps(self, result):
        for output in self.compiled_model.outputs:
            shape = output.shape
            if len(shape) == 4 and shape[1] == 19:
                return result[output]
        raise RuntimeError("Could not find the 19-channel heatmap output")

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

    def run(self, frame: object):
        if self.compiled_model is None:
            with self.model_lock:
                self.compiled_model = self._compile_model(self.current_device)

        tensor = self._prepare_frame(frame)
        t0 = time.perf_counter()
        with self.model_lock:
            result = self.compiled_model([tensor])
            heatmaps = self._get_heatmaps(result)
            device_name = self.current_device
        elapsed = (time.perf_counter() - t0) * 1000
        points = self._extract_keypoints(heatmaps, frame.shape[1], frame.shape[0])
        return {
            "heatmaps": heatmaps,
            "points": points,
            "device": device_name,
            "frame": frame,
            "elapsed_ms": elapsed,
        }
