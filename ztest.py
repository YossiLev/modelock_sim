import cupy as cp
import numpy as np

import time

size = 10_000_000
ntimes = 10

frequencies = np.linspace(0.01, 0.06, ntimes)  
# Index array
n = cp.arange(size)
start = cp.cuda.Event()
end = cp.cuda.Event()

real_part = cp.random.random(size)
imag_part = cp.random.random(size)
complex_array = real_part + 1j * imag_part
for frequency in frequencies:
    start.record()
    # Perform the dot product
    exp_array = cp.exp(1j * n * frequency)

    dot_product = cp.dot(complex_array, exp_array)
    end.record()

    # Wait for the event to complete
    end.synchronize()

    # Time in milliseconds
    elapsed_time_ms = cp.cuda.get_elapsed_time(start, end)
    print(f"Elapsed time: {elapsed_time_ms:.3f} ms")

# test with numpy
n = np.arange(size)
real_part_np = np.random.random(size)
imag_part_np = np.random.random(size)
complex_array_np = real_part_np + 1j * imag_part_np
for frequency in frequencies:
    # Perform the dot product
    start_np = time.time()
    exp_array_np = np.exp(1j * n * frequency)

    dot_product_np = np.dot(complex_array_np, exp_array_np)
    end_np = time.time()

    # Time in milliseconds
    elapsed_time_ms = (end_np - start_np) * 1000
    print(f"Elapsed time numpy: {elapsed_time_ms:.3f} ms")
              