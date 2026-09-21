"""
pose_result_processor.py
------------------------

Simple drawing helpers. These functions take the results from
the model runner (points, heatmaps, timings) and draw on images or create a
mask. We keep drawing separate from model code so it is easier to learn:
"what the model gives" vs "how we show it".
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import math
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

def heatmap_to_image(heatmap: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Convert one heatmap channel into a color image for display.

    Steps (simple for students):
    1) Normalize values to 0..255 (uint8)
    2) Resize to match the given output size (width, height)
    3) Apply a color map so high values are brighter/warmer
    """
    # Normalize heatmap to [0, 255]
    hm = heatmap.astype(np.float32)
    min_val, max_val = float(hm.min()), float(hm.max())
    if max_val - min_val > 1e-6:
        hm = (hm - min_val) / (max_val - min_val)
    else:
        hm = hm * 0.0
    hm_uint8 = (hm * 255).astype(np.uint8)

    # Resize to output size
    w, h = out_size
    hm_resized = cv2.resize(hm_uint8, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply color map for easier viewing
    heat_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
    return heat_color

def heatmaps_grid_to_image(
    heatmaps_3d: np.ndarray,
    frame_size: Tuple[int, int],
    part_names: Optional[List[str]] = None,
    cols: int = 5,
) -> np.ndarray:
    """Build a grid image from all heatmap channels, matching frame height.

    Args:
        heatmaps_3d: array shaped [num_parts, h, w]
        frame_size: (frame_width, frame_height) used to set grid height
        part_names: optional list of names to draw on each tile
        cols: number of columns in the grid (default 5)

    Returns:
        BGR image with size (grid_width, frame_height, 3)
    """
    fw, fh = frame_size
    num_parts = int(heatmaps_3d.shape[0])
    rows = int(math.ceil(num_parts / float(cols))) if num_parts > 0 else 1

    # Tile size tries to fill the frame height with the chosen number of rows
    tile_h = max(1, fh // rows)
    tile_w = tile_h  # square tiles for a simple layout

    tiles: List[np.ndarray] = []
    for i in range(rows * cols):
        if i < num_parts:
            tile = heatmap_to_image(heatmaps_3d[i], (tile_w, tile_h))
            if part_names and i < len(part_names):
                cv2.putText(
                    tile,
                    part_names[i],
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
        else:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        tiles.append(tile)

    # Assemble rows then stack vertically
    row_images: List[np.ndarray] = []
    for r in range(rows):
        row_tiles = tiles[r * cols : (r + 1) * cols]
        row_img = cv2.hconcat(row_tiles)
        row_images.append(row_img)

    grid = cv2.vconcat(row_images)

    # Ensure final grid height exactly matches frame height
    gh, gw = grid.shape[:2]
    if gh != fh:
        scale = fh / float(gh)
        grid = cv2.resize(grid, (int(gw * scale), fh), interpolation=cv2.INTER_LINEAR)

    return grid

def put_text_with_outline(
    frame,
    text,
    position,
    font_scale,
    color,
    thickness=1.5,
    outline_color=(0, 0, 0),
    outline_thickness=2
):
    """
    Draw text with an outline for improved readability and
    resistance to video compression.

    Args:
        frame: BGR image to draw on.
        text: Text to draw.
        position: (x, y) coordinates of the text baseline.
        font_scale: OpenCV font scale.
        color: BGR color of the main text.
        thickness: Thickness of the main text.
        outline_color: BGR color of the outline.
        outline_thickness: Additional thickness around the text.

    Returns:
        The modified frame.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw the outline first
    cv2.putText(
        frame,
        text,
        position,
        font,
        font_scale,
        outline_color,
        thickness + outline_thickness,
        cv2.LINE_AA
    )

    # Draw the main text over the outline
    cv2.putText(
        frame,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return frame


def annotate_metrics(frame, device: str, model_time_ms: float, display_delay_ms: float):
    """Draw the main pipeline timings for students.

    The overlay shows three simple ideas:
    - which device ran the model
    - how long the model took
    - how long the frame waited before the browser saw it

    Note: These numbers are just text drawn on the image. They are not part
    of the model output.
    """
    # Device name with outline
    put_text_with_outline(
        frame,
        f"Device: {device}",
        (20, 40),
        0.8,
        (0, 255, 0),
        2
    )
    # How long the model needed to process the frame
    put_text_with_outline(
        frame,
        f"Model time: {model_time_ms:.0f} ms",
        (20, 75),
        0.7,
        (0, 255, 0),
        2
    )
    # How long the frame waited before the browser saw it
    put_text_with_outline(
        frame,
        f"Display delay: {display_delay_ms:.0f} ms",
        (20, 110),
        0.7,
        (0, 255, 0),
        2
    )
    return frame