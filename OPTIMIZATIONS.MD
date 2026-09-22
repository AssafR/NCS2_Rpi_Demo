# Optimization Narrative

This document tells the story of the main problems we found in the Raspberry Pi pose demo. It shows what first looked wrong, how we tested it, what did not work, and which change actually helped.

The pattern is always the same:

- Clue or suspicion: what first looked wrong
- Analysis: how we tested it or proved it
- Unsuccessful solutions: ideas that did not fix the real problem
- Successful solution: the change that actually helped

Before we start:

- `image_delay` means the total time from a camera image being sampled until a ready image is shown or sent.
- `infer_delay` means the time spent inside the model itself.
- We expected `image_delay - infer_delay` to stay roughly similar on CPU and NPU because the rest of the pipeline was supposed to stay mostly the same. The camera read, frame size, drawing step, and JPEG step should not have changed much.
- When that gap was very different, it meant something outside the model was changing the result.

## 0. The CPU and NPU timings did not match

### Clue or suspicion

At the start, the gap between `image_delay` and `infer_delay` looked very different on CPU and NPU. That suggested the delay was not only coming from the model.

### Analysis

We looked at the whole path on both CPU and NPU, then split it into smaller parts: camera capture, waiting for frames, model inference, post-processing, JPEG encoding, and serving. That showed the model time was only one part of the total delay.

### Unsuccessful solutions

- Looking only at the raw model time.
- Assuming the CPU vs NPU gap was just a model speed difference.
- Treating the whole delay as one number without breaking it into pieces.

### Successful solution

We measured the pipeline in smaller pieces. That gave us the clues we needed for the later fixes: frame handling, buffering, queueing, and thermal behavior.

## 1. The page showed intermediate images before the newest frame was ready

### Clue or suspicion

When the user moved in front of the camera, the page sometimes showed several intermediate images before the newest one was ready. That looked wrong because we wanted one fresh frame at a time, not a chain of old ones.

### Analysis

We first looked at the browser stream and the server path because it looked like a display problem. Then we traced the full flow: camera capture, model inference, drawing, JPEG encoding, and MJPEG serving. That showed the issue was not only in the browser. The app itself was keeping a pipeline full of old work.

### Unsuccessful solutions

- A simple lock around the capture process did not solve the real problem.
- Browser-side confirmation was rejected because it would have added unnecessary complexity.
- Keeping the old pipeline design still allowed stale images to move forward.

### Successful solution

We changed to a one-frame-at-a-time design. The camera keeps only the newest frame, and a new frame replaces the old one instead of joining a backlog. The program waits for a new frame before it runs again.

## 2. Camera reads slowed down during CPU inference

### Clue or suspicion

The camera was fast when tested alone, but it became slow and uneven when CPU inference ran in the same Python process. We did not yet know whether the model was blocking the camera or whether something else was happening.

### Analysis

We compared camera-only tests with combined camera + inference tests. We also measured the time spent in different parts of the loop: camera reads, model calls, drawing, JPEG encoding, and waiting for the next frame.

### Unsuccessful solutions

- Keeping camera capture and CPU inference in the same process.
- Treating the delay as only a browser or JPEG problem.
- Assuming the camera itself was simply slow.

### Successful solution

We put camera capture in its own process.

A thread is one worker inside the same worker room. Threads share the same Python process, so if one thread stays busy too long, the other may have to wait.

A process is a separate worker room with its own space. The camera gets its own process, so it has its own place to run. That means the camera can keep reading frames even when the main process is busy with inference.

This stops starvation. The camera is no longer fighting the model for the same time and attention inside one Python process. V4L2 can keep reading from the camera, and the main process can do the model work at the same time.

## 3. Webcam FPS dropped

### Clue or suspicion

When we asked OpenCV for a one-frame V4L2 buffer, the webcam speed dropped from about 5 FPS to about 2.5 FPS. That looked suspicious, but we did not yet know whether the buffer setting was the real cause.

### Analysis

We compared the same webcam in two modes:

- normal driver buffering
- `CAP_PROP_BUFFERSIZE = 1`

The difference was large enough to show that the buffer request was not helping this camera.

### Unsuccessful solutions

- Forcing the driver to use a one-frame buffer.
- Assuming a smaller buffer must always mean lower latency.

### Successful solution

We stopped requesting a one-frame buffer. The camera keeps any extra frames on its own, and the program handles stale data in a smarter way.

## 4. The video looked stuck on the same picture after a timeout

### Clue or suspicion

After a camera timeout, the video sometimes looked stuck on the same picture instead of moving to a fresh one. We could see that something was not moving forward, but we did not yet know why.

### Analysis

We traced the meaning of `ok, frame = cap.read()`. A failed read does not create a fresh image. It only means the previous image is still there, so the code can be fooled if it asks only, "Do we have a frame?"

### Unsuccessful solutions

- Treating any stored image as if it were new.
- Letting the inference loop reuse the last buffer without checking freshness.

### Successful solution

We gave each successful frame a number. The inference loop now waits for a higher number before it runs again. That changes the question from "Do we have a frame?" to "Did a newer frame arrive?"

## 5. The waiting felt uneven

### Clue or suspicion

The camera and model were already separated more cleanly, but the app still had uneven delays. We did not yet know what was causing the waiting.

### Analysis

We looked at the path where full frames moved between processes. A multiprocessing queue has to copy each image from one process to another, and that copying can add time and jitter.

### Unsuccessful solutions

- Passing full images through a queue as the main transport.
- Keeping a backlog of older frames.

### Successful solution

We switched to one shared-memory slot for the newest frame. The camera writes the latest image there, and the inference process reads it when ready. Old frames are replaced, not piled up, so there is less copying and less waiting.

## 6. The app felt slow

### Clue or suspicion

We could see the app was slow, but we did not know where the delay came from. Was it the camera, the model, drawing, JPEG, or waiting for a new frame?

### Analysis

We added diagnostics for:

- camera read wall time
- camera read CPU time
- loop gap
- lock wait
- shared-memory copy time
- model call time
- post-processing time
- JPEG encode time
- ready-to-send delay

### Unsuccessful solutions

- Looking only at overall FPS.
- Assuming the model was the only slow part.

### Successful solution

We split the timing into smaller parts. That made it clear that the camera path, model path, and rendering path each had their own role.

## 7. Image delay sometimes spiked

### Clue or suspicion

We sometimes saw `image_delay` get much larger, even after the camera work was fixed. The delay was not steady, and it did not line up with the camera changes we had already made, so we suspected a machine-wide limit instead of a camera bug.

### Analysis

That made CPU throttling the next thing to test, because a thermal limit can make the same code run fine some of the time and much slower at other times. We checked temperature and throttling with `vcgencmd measure_temp` and `vcgencmd get_throttled`. When the Pi crossed about 80 C, throttling appeared.

### Unsuccessful solutions

- Treating the slowdown as only a camera problem.
- Expecting software changes to fix thermal throttling.

### Successful solution

We documented the thermal issue clearly and treated it as a separate explanation. Cooling, airflow, and using the NCS2/MYRIAD device all help reduce this bottleneck.

## 8. The diagnostic notes kept drifting out of sync

### Clue or suspicion

We found that the diagnostic notes had many counters and timing numbers. It felt fragile because resetting them by hand could get out of sync with the data fields.

### Analysis

We looked at the maintenance burden of keeping a reset method updated whenever a new field was added.

### Unsuccessful solutions

- Keeping a separate `reset()` method that had to be updated by hand.

### Successful solution

We used dataclasses and let the default values be the source of truth. That made the diagnostics easier to read and less fragile.

## 9. USB looked busy

### Clue or suspicion

We wondered if USB bandwidth was the real bottleneck. The bus looked busy, but we did not yet know for sure.

### Analysis

We calculated the camera input bandwidth, the model input bandwidth, and the model output bandwidth at 5 FPS. Then we compared each one with USB 2.0 on a Raspberry Pi 3.

### Unsuccessful solutions

- Treating the USB bus as the only performance limit.

### Successful solution

The math showed that the total USB load was still only a fraction of USB 2.0 capacity. That helped us focus on the real issues: buffering, CPU load, and thermal throttling.

## Final takeaway: the clues pointed to the same few root causes

The biggest improvements came from the same idea repeated in different places:

- keep only the newest frame
- do not let stale work pile up
- separate camera capture from CPU inference
- measure the real bottleneck instead of guessing

That is why the final design is easier to explain, easier to debug, and more stable for students.