"""
pose_estimation.py (compatibility shim)
--------------------------------------

Older lessons may import `PoseEstimator` from this module. New code should
import `PoseModelRunner` from pose_model_runner.py instead. To keep older
notebooks and scripts working, we re-export `PoseModelRunner` under the
name `PoseEstimator` here.

Summary for students:
- Use `PoseModelRunner` (preferred). It returns a dict with 'points',
  'heatmaps', 'device', 'elapsed_ms', and 'frame'. It does NOT draw.
- Keep drawing/overlays in pose_result_processor.py and web serving in
  server_handler.py.
"""

from pose_model_runner import PoseModelRunner as PoseEstimator

__all__ = ["PoseEstimator"]
