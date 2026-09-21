# Raspberry Pi Pose Estimation Web Demo

This lesson splits the app into three simple concerns:

- Model execution (run a CNN and get results)
  - `pose_model_runner.py`: loads the OpenVINO pose model, compiles for a device (CPU or MYRIAD/NCS2), runs inference, and returns structured results.
- Result processing / visualization (decide what to draw)
  - `pose_result_processor.py`: draws a stick-figure skeleton and friendly text overlays; can also create a grayscale pose mask for blending.
- Web serving (UI + streaming)
  - `server_handler.py`: plain HTTP handler factory that serves the static UI, MJPEG video stream, and a device-switch endpoint.
  - `camera_capture.py`: runs camera capture in a separate process and keeps only the newest frame.
  - `webcam_web.py`: asks the runner to infer, uses the processor to draw, then streams the JPEGs via the server handler.

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

Keep “what the model outputs” (runner) separate from “how we draw it” (processor) and “how we serve it” (server). This makes the lesson easier to follow.

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

> Note: On Raspberry Pi, you can also use prebuilt packages from your vendor or follow OpenVINO’s install guide for your board.

## Run it

```bash
python3 webcam_web.py
```

- Open a browser to: `http://<raspberry-pi-ip>:8080/`
- Click the buttons to switch between `CPU` and `MYRIAD` (Intel NCS2). The same CNN, same camera — different inference hardware.

## HTTP endpoints

- `/` — Serves the static UI (index.html)
- `/static/...` — Serves JS/CSS/assets from the static folder
- `/video` — MJPEG video stream (a stream of JPEG images). The `<img src="/video">` in the page uses this.
- `/device?name=CPU|MYRIAD` — Signals a device switch; the inference loop applies it safely

## How device switching works (simple & safe)

- The HTTP handler only sets a `requested_device` flag.
- The inference loop (single writer) sees the flag and calls `runner.set_device(...)` at a safe point.
- This avoids cross-thread model swaps while a request is streaming.

## Camera And CPU Diagnostic

Use this optional test if the webcam becomes slow only while CPU inference is
running. It puts the camera and model in separate processes, then prints their
timing reports. It does not open the web page or show pose results.

```bash
python3 diagnose_capture_process.py --device CPU --seconds 30
python3 diagnose_capture_process.py --device MYRIAD --seconds 30
python3 diagnose_capture_process.py --device NONE --seconds 30
```

How to read the result:

- Camera stays near its requested FPS with CPU inference: the same-process
  thread arrangement in the web app is likely interfering with capture.
- Camera becomes slow even in this test: the problem is more likely in the
  camera, V4L2 driver, USB connection, or whole-system resource use.

For a second camera-only comparison, omit the one-frame OpenCV buffer request:

```bash
python3 diagnose_capture_process.py --device NONE --seconds 30 --no-buffer-limit
```

For the included UVC webcam, requesting `CAP_PROP_BUFFERSIZE = 1` reduced
capture from about 5 FPS to about 2.5 FPS. The main app therefore leaves the
driver buffer unchanged and drops old frames in `CameraFrameGrabber` instead.

## Capture Process Bug Fix

This section explains a real bug we found while testing CPU inference.

### Before: Camera Thread And Model Thread

At first, the camera and the CPU model ran as threads inside one Python
process:

```text
one Python process
  camera thread -> cap.read()
  inference thread -> CPU model
```

The camera worked well by itself at about 5 FPS. However, when the CPU model
ran in the same process, `cap.read()` became slow and uneven. The camera often
returned only a few frames per second. This made the web page update slowly,
even though drawing and JPEG compression took only a few milliseconds.

We also found that asking OpenCV for a one-frame V4L2 buffer
(`CAP_PROP_BUFFERSIZE = 1`) reduced this webcam from about 5 FPS to about
2.5 FPS. The app no longer requests that driver buffer setting.

### Tests

We compared these situations:

| Test | Measured camera result |
| --- | --- |
| Camera by itself with `v4l2-ctl` | About 4.6 FPS |
| Separate camera process with CPU model running | About 5 FPS |
| Separate camera process with MYRIAD model running | About 5 FPS |

These tests showed that the webcam can keep its normal speed while CPU
inference runs in another process. The main problem was the old design where
camera capture and CPU inference shared one Python process.

### After: Camera Process And Inference Process

The production app now uses this design:

```text
camera process
  reads the webcam
  keeps only the newest frame
  sends newest frame -> one-item queue

main process
  runs CPU or MYRIAD inference
  draws overlays
  sends JPEG images to the browser
```

The one-item queue is important. If the model is slow, a newer camera frame
replaces an older queued frame. The app does not build a long list of old
frames. The model therefore works on a recent image when it is ready.

In simple words:

> The camera gets its own process, so slow CPU inference does not block camera reads in the main process.

## Why Frames Have Numbers

The camera runs in its own process. It keeps only the newest image, not a long
list of images. Each successful camera capture receives a number: 1, 2, 3,
and so on.

The inference loop remembers the last number it processed. It waits for a
higher number before it runs the model again. This is important if the camera
is slow or reports a timeout. The old image may still be stored, but it is not
a new camera image, so the program must not process it again.

In short:

```text
new camera frame -> new number -> run the model once
no new camera frame -> no new number -> wait
```

This prevents a stale image from being processed repeatedly during a camera
timeout.

## Model-only Quickstart (no web server)

Run a minimal demo that opens the webcam, runs the model, and shows a window:

```bash
python3 run_model_demo.py
```

What it does:
- Captures frames from the webcam
- Calls `PoseModelRunner.run(frame)` to get `points`, `heatmaps`, `device`, `elapsed_ms`
- Draws the pose and friendly overlays on the frame
- Shows the frame in a window (press ESC to quit)

If you prefer to write it yourself, see the Quickstart in `pose_model_runner.py`.

## Glossary (for students)

- Heatmaps: The model's raw, per-body-part confidence maps (19 channels in this demo). They are useful for advanced visualization and understanding what the model "sees".
- Points (keypoints): A simple, single-person extraction of the most confident location per body part, returned as a dict: name → `(x, y, confidence)`.
- Device: Hardware that runs inference: `CPU` (Raspberry Pi) or `MYRIAD` (Intel NCS2). Same network, different execution backends.
- Inference time (`elapsed_ms`): How long the model call took for the last frame.
- Inference FPS vs. loop FPS: Inference FPS is derived from the last model call. Loop FPS includes grabbing frames, drawing overlays, encoding JPEGs, etc., so it may be different.
- Single-writer pattern: The HTTP server only sets a `requested_device` flag. The inference loop (single writer) applies the change between frames. This is safer and simpler than swapping models from multiple threads.

## File overview

- `webcam_web.py` — Inference loop + HTTP server wiring (no HTML inside).
- `camera_capture.py` — Webcam setup and separate-process capture of newest frames.
- `pose_model_runner.py` — OpenVINO model load/compile/run; returns results (points, heatmaps, device, elapsed).
- `pose_result_processor.py` — Draw skeleton and overlays; build/overlay a grayscale mask.
- `server_handler.py` — Serves `/`, `/static/...`, `/video`, and `/device?name=...`.
- `static/` — Contains `index.html` and small JS for device switching.
- `model/` — Pose model files (XML and BIN).

<!-- Intentionally no lessons section for now to keep the focus on running the model. -->
