import time
import numpy as np
from openvino.runtime import Core

MODEL = "model/human-pose-estimation-0001.xml"

with open("/proc/device-tree/model") as f:
    print("Computer:", f.read().strip("\x00\n"))

core = Core()
print("Available devices:", core.available_devices)

model = core.read_model(MODEL)

# Inspect the model
print("\nInputs:")
for inp in model.inputs:
    print(" ", inp.any_name, inp.shape, inp.element_type)

print("\nOutputs:")
for out in model.outputs:
    print(" ", out.any_name, out.shape, out.element_type)

# Use the model's actual input shape
input_layer = model.input(0)
input_shape = input_layer.shape

x = np.random.rand(*input_shape).astype(np.float32)

print("\nCompiling model for MYRIAD...")
start = time.perf_counter()
compiled = core.compile_model(model, "MYRIAD")
compile_time = time.perf_counter() - start

print(f"Compilation successful: {compile_time:.2f} s")

print("\nRunning inference...")
start = time.perf_counter()
result = compiled([x])
inference_time = time.perf_counter() - start

print(f"Inference successful: {inference_time * 1000:.1f} ms")

print("\nOutput shapes:")
for output in compiled.outputs:
    print(" ", result[output].shape)


def benchmark(core, model, device, x, warmup=5, runs=50):
    print(f"\n--- {device} ---")

    start = time.perf_counter()
    compiled = core.compile_model(model, device)
    compile_time = time.perf_counter() - start

    print(f"Compile time: {compile_time:.2f} s")

    # Warm-up
    for _ in range(warmup):
        compiled([x])

    # Benchmark
    times = []

    for _ in range(runs):
        start = time.perf_counter()
        compiled([x])
        times.append(time.perf_counter() - start)

    times = np.array(times)

    print(f"Mean latency:   {times.mean() * 1000:.1f} ms")
    print(f"Median latency: {np.median(times) * 1000:.1f} ms")
    print(f"FPS:            {1 / times.mean():.2f}")

    return times


cpu_times = benchmark(core, model, "CPU", x)
myriad_times = benchmark(core, model, "MYRIAD", x)

speedup = cpu_times.mean() / myriad_times.mean()

print("\n============================")
print(f"NCS2 speedup: {speedup:.2f}x")
print("============================")


