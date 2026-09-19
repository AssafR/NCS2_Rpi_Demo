"""
pose_model_runner.py
--------------------

Beginner-friendly wrapper around an OpenVINO pose-estimation model.

This module is responsible ONLY for:
- Loading the model and compiling it for a target device (CPU / MYRIAD)
- Preparing input frames and running inference
- Returning structured, model-agnostic results that other code can consume

It does NOT draw on images or serve HTTP. That separation keeps lessons clear:
- "Model execution" (here)
- "Result processing / visualization" (pose_result_processor.py)
- "Web serving" (server_handler.py used by webcam_web.py)
"""

import time
import threading
import numpy as np
import cv2
from openvino.runtime import Core
from pose_defs import BODY_PARTS as DEF_BODY_PARTS, POSE_PAIRS as DEF_POSE_PAIRS


class PoseModelRunner:
    """Run a pose-estimation model and return structured results.

    Typical usage in the inference loop:

        res = runner.run(frame)
        # 'res' is a dictionary with several useful fields:
        #   - 'heatmaps': raw model heatmaps (19 channels for body parts)
        #   - 'points'  : decoded keypoints as a dict: name -> (x, y, confidence)
        #   - 'device'  : the device used for inference (e.g. 'CPU' or 'MYRIAD')
        #   - 'frame'   : the same input frame (not copied)
        #   - 'elapsed_ms': wall-clock inference time in milliseconds

    Keep this class focused on execution so students can clearly see what
    the model "returns" versus how the results are later drawn or served.
    """
    # Pose model definitions (mirrored from the original file)
    # Use shared constants to avoid duplication across modules
    BODY_PARTS = DEF_BODY_PARTS
    POSE_PAIRS = DEF_POSE_PAIRS

    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self,
                                model_path: str,
                                initial_device: str = "MYRIAD",
                                model_w: int = 456,
                                model_h: int = 256):
        """Initialize the model executor.

        Student note:
        - Camera configuration (width/height/FPS) belongs to the capture layer.
            The runner only needs the MODEL's input size (model_w/model_h) and
            which device to target (CPU or MYRIAD).
        """
        self.model_path = model_path
        self.model_w = model_w
        self.model_h = model_h
        self.current_device = initial_device

        self.core = Core()
        self.model = self.core.read_model(self.model_path)
        self.compiled_model = None
        self.model_lock = threading.Lock()

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
        """Compile and switch the active device.

        For simplicity, this compiles on demand when switching.
        Returns True on success; False if the device string is invalid
        or compilation fails.
        """
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
        """Convert heatmaps to a simple set of single-person keypoints.

        This is a "strongest-activation" approach for classroom demos.
        It ignores PAFs (part affinity fields) and multi-person decoding
        to keep the lesson focused and readable.
        """
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
        """Run one inference pass on the given BGR `frame`.

        Returns: dict with keys
            - 'heatmaps': np.ndarray, model heatmaps (shape depends on model)
            - 'points'  : Dict[str, Tuple[int, int, float]] of keypoints
            - 'device'  : str, the device used for inference
            - 'frame'   : the same input frame
            - 'elapsed_ms': float, inference time in milliseconds
        """
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
