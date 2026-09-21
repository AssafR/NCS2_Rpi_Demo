# Raspberry Pi Pose Estimation Web Demo

This lesson shows how to run a pose estimation model on a Raspberry Pi. The app is divided into three simple parts:

- **Model execution** (run a CNN and get results)
  - pose_model_runner.py: loads the OpenVINO pose model, compiles it for a device (CPU or MYRIAD/NCS2), runs inference, and returns structured results.
- **Result processing / visualization** (decide what to draw)
  - pose_result_processor.py: draws a stick-figure skeleton and friendly text overlays; can also create a grayscale pose mask for blending.
- **Web serving** (UI + streaming)
  - server_handler.py: plain HTTP handler factory that serves the static UI, MJPEG video stream, and a device-switch endpoint.
  - webcam_web.py: captures camera frames, asks the runner to infer, uses the processor to draw, then streams the JPEGs via the server handler.

## What does the model return?

When you call the runner:

`python
res = runner.run(frame)
`

You get a small dictionary with:

- res['points']: Decoded keypoints as a dict mapping part-name -> (x, y, confidence).
- res['heatmaps']: The raw 19-channel heatmaps from the model (advanced exploration).
- res['device']: Which device actually ran inference (e.g., CPU or MYRIAD).
- res['frame']: The same input frame (no copy).
- res['elapsed_ms']: Inference time in milliseconds.

Keep what the model outputs (runner) separate from how we draw it (processor) and how we serve it (server). This makes the lesson easier to follow.

## Requirements

- Raspberry Pi OS (Linux)
- Python 3.9+
- Packages:
  - openvino (runtime)
  - opencv-python (cv2)
  - numpy
- Model files in model/ (e.g., human-pose-estimation-0001.xml and its .bin).

Install packages (example):

`ash
python3 -m pip install --upgrade pip
python3 -m pip install openvino opencv-python numpy
`

> Note: On Raspberry Pi, you can also use prebuilt packages from your vendor or follow OpenVINO's install guide for your board.

## Run it
