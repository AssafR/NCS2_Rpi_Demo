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
    if len(mask.shape) == 2:
        colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 0.8, colored, 0.2, 0)
    return frame
