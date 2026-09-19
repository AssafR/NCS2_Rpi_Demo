import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

print("Opened:", cap.isOpened())

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

for i in range(10):
    ret, frame = cap.read()
    print(i, ret, None if frame is None else frame.shape)

    if ret:
        cv2.imwrite("webcam_test.jpg", frame)
        print("Saved webcam_test.jpg")
        break

cap.release()

while True:
    ret, frame = cap.read()

    # Webcam frame -> model input
    resized = cv2.resize(frame, (456, 256))
    x = resized.transpose(2, 0, 1)[None].astype(np.float32)

    # NCS2
    start = time.perf_counter()
    result = compiled([x])
    inference_ms = (time.perf_counter() - start) * 1000

    # Display
    cv2.putText(
        frame,
        f"MYRIAD: {inference_ms:.0f} ms",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("NCS2 Webcam", frame)

    if cv2.waitKey(1) == ord("q"):
        break
