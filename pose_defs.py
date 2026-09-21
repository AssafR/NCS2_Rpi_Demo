"""
pose_defs.py
------------

Single source of truth for pose landmark names and connectivity.

Keeping BODY_PARTS and POSE_PAIRS here avoids duplication and makes it easy
for both the model runner and the visualization code to agree on naming and
skeletal connections. This is a simplified, OpenPose-like skeleton for a
single-person classroom demo.

Reference:
- These names and connections are adapted from the OpenPose convention and the
    Intel Open Model Zoo "human-pose-estimation-0001" model.
- Model docs: https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/human-pose-estimation-0001/README.md
"""

BODY_PARTS = [
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar"
]

# POSE_PAIRS explains which two body parts should be connected by a line.
# Example: ("Neck", "RShoulder") means draw a line from Neck to Right Shoulder.
# This is used by the drawing helper to build the "stick figure" skeleton.
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
