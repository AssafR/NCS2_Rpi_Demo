"""
pose_estimation.py (refactored)
--------------------------------

Beginner note:
- This module is focused on RUNNING the pose-estimation model only.
- It should NOT draw on images or smooth FPS/metrics — keep that in a
    visualization/util module (e.g., pose_result_processor.py) or the caller.

Separation of concerns makes lessons clearer:
- Model execution (this file): returns structured results (points, heatmaps).
- Result processing (another file): draws skeletons, overlays text, etc.
- Web serving (another file): streams JPEG frames, handles HTTP.
"""

import time
import threading
import numpy as np
import cv2
from openvino.runtime import Core
from pose_defs import BODY_PARTS as DEF_BODY_PARTS, POSE_PAIRS as DEF_POSE_PAIRS


class PoseEstimator:
    # Pose model definitions
    # Use shared constants to keep definitions in one place
    BODY_PARTS = DEF_BODY_PARTS
    POSE_PAIRS = DEF_POSE_PAIRS

    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self,
                                model_path: str,
                                initial_device: str = "MYRIAD",
                                model_w: int = 456,
                                model_h: int = 256):
            """Initialize a pose-estimation executor.

            Note for students:
            - Camera configuration (resolution/FPS) belongs to whoever captures
                frames (e.g., a webcam setup function), not the model executor.
            - Here we only care about the MODEL's input size (model_w/model_h).
            """
            self.model_path = model_path
            self.model_w = model_w
            self.model_h = model_h
            self.current_device = initial_device

            self.core = Core()
            self.model = self.core.read_model(self.model_path)
            self.compiled_model = None
            self.model_lock = threading.Lock()

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

    # Note: drawing/overlays are intentionally NOT in this module. Use
    # pose_result_processor.render_pose_on_frame(...) for visualization.

    def _get_heatmaps(self, result):
        for output in self.compiled_model.outputs:
            shape = output.shape
            if len(shape) == 4 and shape[1] == 19:
                return result[output]
        raise RuntimeError("Could not find the 19-channel heatmap output")

    def infer(self, frame):
        """Run inference on `frame` and return structured results only.

        Returns a dict:
            {
                'heatmaps': np.ndarray,   # raw model heatmaps
                'points': Dict[str, Tuple[int, int, float]],
                'device': str,            # 'CPU' or 'MYRIAD'
                'elapsed_ms': float,      # single-pass inference time
                'frame': np.ndarray       # the same input frame (not copied)
            }

        No drawing, no JPEG encoding, and no FPS smoothing in this module.
        """
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

        h, w = frame.shape[:2]
        points = self._extract_keypoints(heatmaps, w, h)

        return {
            'heatmaps': heatmaps,
            'points': points,
            'device': device_name,
            'elapsed_ms': inference_ms,
            'frame': frame,
        }
