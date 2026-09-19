import time

import cv2
import numpy as np
from openvino.runtime import Core


MODEL_PATH = "model/human-pose-estimation-0001.xml"
CAMERA_ID = 0

# Model input dimensions
MODEL_W = 456
MODEL_H = 256


def compile_for_device(core, model, device):
    """Compile the model for CPU or MYRIAD."""
    print(f"\nCompiling for {device}...")

    start = time.perf_counter()
    compiled = core.compile_model(model, device)
    elapsed = time.perf_counter() - start

    print(f"Compiled for {device} in {elapsed:.2f} s")

    # Warm-up with a dummy frame
    dummy = np.zeros(
        (1, 3, MODEL_H, MODEL_W),
        dtype=np.float32
    )

    compiled([dummy])

    return compiled


def prepare_frame(frame):
    """
    Prepare an OpenCV BGR frame for the OpenVINO model.

    HWC: height x width x channels
        ->
    NCHW: batch x channels x height x width
    """

    resized = cv2.resize(frame, (MODEL_W, MODEL_H))

    tensor = resized.transpose(2, 0, 1)
    tensor = tensor[np.newaxis, ...]
    tensor = tensor.astype(np.float32)

    return tensor


# ---------------------------------------------------------
# OpenVINO
# ---------------------------------------------------------

core = Core()

print("Available OpenVINO devices:", core.available_devices)

model = core.read_model(MODEL_PATH)

print("Model input:", model.input(0).shape)

device = "MYRIAD"
compiled = compile_for_device(core, model, device)


# ---------------------------------------------------------
# Webcam
# ---------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

print()
print("Controls:")
print("  M - use MYRIAD / NCS2")
print("  C - use Raspberry Pi CPU")
print("  Q - quit")
print()

smoothed_ms = None

while True:

    # -------------------------
    # Capture
    # -------------------------

    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam frame")
        break

    # -------------------------
    # Preprocess
    # -------------------------

    tensor = prepare_frame(frame)

    # -------------------------
    # Inference
    # -------------------------

    start = time.perf_counter()

    result = compiled([tensor])

    inference_ms = (
        time.perf_counter() - start
    ) * 1000

    # Smooth the displayed number slightly
    if smoothed_ms is None:
        smoothed_ms = inference_ms
    else:
        smoothed_ms = (
            0.9 * smoothed_ms
            + 0.1 * inference_ms
        )

    fps = 1000 / smoothed_ms

    # -------------------------
    # Display information
    # -------------------------

    cv2.putText(
        frame,
        f"Device: {device}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Inference: {smoothed_ms:.0f} ms",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Inference FPS: {fps:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Raspberry Pi + OpenVINO",
        frame
    )

    # -------------------------
    # Keyboard
    # -------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("m") and device != "MYRIAD":

        device = "MYRIAD"
        compiled = compile_for_device(
            core,
            model,
            device
        )

        smoothed_ms = None

    elif key == ord("c") and device != "CPU":

        device = "CPU"
        compiled = compile_for_device(
            core,
            model,
            device
        )

        smoothed_ms = None


# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Finished.")

