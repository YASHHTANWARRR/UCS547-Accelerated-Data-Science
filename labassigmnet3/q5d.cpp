import cupy as cp
import time

N = 5000000

A = cp.ones(N, dtype=cp.int32)
B = cp.ones(N, dtype=cp.int32)

start = time.time()

C = A + B

cp.cuda.Stream.null.synchronize()

end = time.time()

print("RAPIDS/CuPy Time:",
      (end-start)*1000,
      "ms")