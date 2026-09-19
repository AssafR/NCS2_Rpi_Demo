"""
pose_result_processor.py
------------------------

Simple drawing helpers for students. These functions take the results from
the model runner (points, heatmaps, timings) and draw on images or create a
mask. We keep drawing separate from model code so it is easier to learn:
"what the model gives" vs "how we show it".
"""

import cv2
import numpy as np
from pose_defs import POSE_PAIRS  # Shared source of truth for skeleton edges

def render_pose_on_frame(frame, points, color=(0, 255, 0)):
    """Draw a simple stick-figure skeleton and joints on the frame.

    Args:
        frame: BGR image to draw on (changed in place)
        points: keypoints from the runner: name -> (x, y, confidence)
        color: BGR color for lines
    Returns:
        The same frame, so you can chain calls.
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
    """Create a grayscale mask image that shows the skeleton.

    You can overlay this mask on the original image to highlight the pose.
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
    """Overlay a grayscale mask on the original frame with some transparency."""
    if len(mask.shape) == 2:
        colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 0.8, colored, 0.2, 0)
    return frame

def annotate_metrics(frame, device: str, inference_ms: float, loop_fps: float):
    """Draw text: device name, inference time (ms), and FPS.

    Note: These numbers are just text drawn on the image. They are not part
    of the model output.
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
