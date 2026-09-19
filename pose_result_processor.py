"""
pose_result_processor.py
------------------------

Beginner-friendly visualization helpers. These functions take the structured
results from the model runner (points, heatmaps, timings) and turn them into
annotated images or masks. Keeping this separate from model execution helps
students see the difference between "what the model outputs" and
"how we choose to visualize it".
"""

import cv2
import numpy as np

# Default pose pairs matching the model's BODY_PARTS order (copied from the runner)
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

BODY_PARTS = [
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar"
]

def render_pose_on_frame(frame, points, color=(0, 255, 0)):
    """Draw a simple stick-figure skeleton and joints onto `frame`.

    Args:
        frame: np.ndarray, BGR image to draw on (modified in place)
        points: Dict[str, Tuple[int, int, float]] keypoints from the runner
        color: BGR line color for the skeleton
    Returns:
        The same `frame` for convenience chaining.
    """
    # Draw skeleton
    for part_a, part_b in POSE_PAIRS:
        a = points.get(part_a)
        b = points.get(part_b)
        if a is not None and b is not None:
            cv2.line(frame, (a[0], a[1]), (b[0], b[1]), color, 3)
    # Draw joints
    for point in points.values():
        if point is not None:
            cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)
    return frame

def create_pose_mask(frame_shape, points, color=255):
    """Create a grayscale mask image highlighting the pose skeleton.

    Useful for blending, segmentation-style overlays, or teaching image masks.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for part_a, part_b in POSE_PAIRS:
        a = points.get(part_a)
        b = points.get(part_b)
        if a is not None and b is not None:
            cv2.line(mask, (a[0], a[1]), (b[0], b[1]), color, 2)
    for point in points.values():
        if point is not None:
            cv2.circle(mask, (point[0], point[1]), 5, color, -1)
    return mask

def overlay_mask(frame, mask):
    """Overlay a grayscale mask onto the original frame with transparency."""
    if len(mask.shape) == 2:
        colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 0.8, colored, 0.2, 0)
    return frame

def annotate_metrics(frame, device: str, inference_ms: float, loop_fps: float):
    """Draw friendly text overlays for device, inference time, and FPS.

    This keeps "presentation" concerns out of the inference loop and makes it
    clear to students that these numbers are just drawn text, not part of the
    model itself.
    """
    # Device name
    cv2.putText(
        frame,
        f"Device: {device}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    # Inference time
    cv2.putText(
        frame,
        f"Inference: {inference_ms:.0f} ms",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    # Inference FPS (derived)
    inf_fps = (1000.0 / inference_ms) if inference_ms and inference_ms > 0 else 0.0
    cv2.putText(
        frame,
        f"Inference FPS: {inf_fps:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    # Actual loop FPS (smoothed)
    cv2.putText(
        frame,
        f"Actual loop FPS: {loop_fps:.2f}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    return frame
