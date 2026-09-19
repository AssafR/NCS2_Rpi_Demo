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
