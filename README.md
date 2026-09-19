# Raspberry Pi Pose Estimation Web Demo

This lesson splits the app into three simple concerns:

- Model execution (run a CNN and get results)
  - `pose_model_runner.py`: loads the OpenVINO pose model, compiles for a device (CPU or MYRIAD/NCS2), runs inference, and returns structured results.
- Result processing / visualization (decide what to draw)
  - `pose_result_processor.py`: draws a stick-figure skeleton and friendly text overlays; can also create a grayscale pose mask for blending.
- Web serving (UI + streaming)
  - `server_handler.py`: plain HTTP handler factory that serves the static UI, MJPEG video stream, and a device-switch endpoint.
  - `webcam_web.py`: captures camera frames, asks the runner to infer, uses the processor to draw, then streams the JPEGs via the server handler.

## What does the model return?

When you call the runner:

```python
res = runner.run(frame)
```

You get a small dictionary with:

- `res['points']`: Decoded keypoints as a dict mapping part-name → `(x, y, confidence)`.
- `res['heatmaps']`: The raw 19-channel heatmaps from the model (advanced exploration).
- `res['device']`: Which device actually ran inference (e.g., `CPU` or `MYRIAD`).
- `res['frame']`: The same input frame (no copy).
- `res['elapsed_ms']`: Inference time in milliseconds.

Keep “what the model outputs” (runner) separate from “how we draw it” (processor) and “how we serve it” (server) — this makes the lesson easier to follow.

## Requirements

- Raspberry Pi OS (Linux)
- Python 3.9+
- Packages:
  - `openvino` (runtime)
  - `opencv-python` (cv2)
  - `numpy`
- Model files in `model/` (e.g., `human-pose-estimation-0001.xml` and its `.bin`).

Install packages (example):

```bash
python3 -m pip install --upgrade pip
python3 -m pip install openvino opencv-python numpy
```

> Note: On Raspberry Pi, you may prefer prebuilt wheels from your platform vendor or use OpenVINO’s official install guide if available for your board.

## Run it

```bash
python3 webcam_web.py
```

- Open a browser to: `http://<raspberry-pi-ip>:8080/`
- Click the buttons to switch between `CPU` and `MYRIAD` (Intel NCS2). The same CNN, same camera — different inference hardware.

## HTTP endpoints

- `/` — Serves the static UI (index.html)
- `/static/...` — Serves JS/CSS/assets from the static folder
- `/video` — MJPEG video stream (the `<img src="/video">` in the page uses this)
- `/device?name=CPU|MYRIAD` — Signals a device switch; the inference loop applies it safely

## How device switching works (simple & safe)

- The HTTP handler only sets a `requested_device` flag.
- The inference loop (single writer) sees the flag and calls `runner.set_device(...)` at a safe point.
- This avoids cross-thread model swaps while a request is streaming.

## Glossary (for students)

- Heatmaps: The model's raw, per-body-part confidence maps (19 channels in this demo). They are useful for advanced visualization and understanding what the model "sees".
- Points (keypoints): A simple, single-person extraction of the most confident location per body part, returned as a dict: name → `(x, y, confidence)`.
- Device: Hardware that runs inference: `CPU` (Raspberry Pi) or `MYRIAD` (Intel NCS2). Same network, different execution backends.
- Inference time (`elapsed_ms`): How long the model call took for the last frame.
- Inference FPS vs. loop FPS: Inference FPS is derived from the last model call. Loop FPS includes grabbing frames, drawing overlays, encoding JPEGs, etc., so it may be different.
- Single-writer pattern: The HTTP server only sets a `requested_device` flag. The inference loop (single writer) applies the change between frames. This is safer and simpler than swapping models from multiple threads.

## File overview

- `webcam_web.py` — Camera + inference loop + HTTP server wiring (no HTML inside).
- `pose_model_runner.py` — OpenVINO model load/compile/run; returns results (points, heatmaps, device, elapsed).
- `pose_estimation.py` — Compatibility shim: re-exports `PoseModelRunner` as `PoseEstimator` for older lessons.
- `pose_result_processor.py` — Draw skeleton and overlays; build/overlay a grayscale mask.
- `server_handler.py` — Serves `/`, `/static/...`, `/video`, and `/device?name=...`.
- `static/` — Contains `index.html` and small JS for device switching.
- `model/` — Pose model files (XML and BIN).

## Next steps (ideas for lessons)

- Visualize the heatmaps directly (false-color render per part).
- Explain confidence thresholds and keypoint detection.
- Compare performance across devices with a small chart.
- Add `/status` endpoint returning JSON with device and performance metrics.
