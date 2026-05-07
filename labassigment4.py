import numpy as np
import math
import time
from numba import njit, vectorize, prange, cuda


print("\n================ QUESTION 1 =================")

N = 5_000_000

# ---------------- CPU VERSION ----------------
x_cpu32 = np.random.rand(N).astype(np.float32)
x_cpu64 = np.random.rand(N).astype(np.float64)

# float32 CPU
start = time.time()
y_cpu32 = x_cpu32**2 + 3*x_cpu32 + 5
cpu32_time = time.time() - start

# float64 CPU
start = time.time()
y_cpu64 = x_cpu64**2 + 3*x_cpu64 + 5
cpu64_time = time.time() - start

print(f"CPU float32 Time : {cpu32_time:.6f} sec")
print(f"CPU float64 Time : {cpu64_time:.6f} sec")


# ---------------- CUDA KERNELS ----------------

@cuda.jit
def poly_kernel_float32(x, y):
    idx = cuda.grid(1)
    if idx < x.size:
        y[idx] = x[idx] * x[idx] + 3 * x[idx] + 5


@cuda.jit
def poly_kernel_float64(x, y):
    idx = cuda.grid(1)
    if idx < x.size:
        y[idx] = x[idx] * x[idx] + 3 * x[idx] + 5


if cuda.is_available():

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    # ---------- FLOAT32 ----------
    d_x32 = cuda.to_device(x_cpu32)
    d_y32 = cuda.device_array_like(x_cpu32)

    start = time.time()
    poly_kernel_float32[blocks_per_grid, threads_per_block](d_x32, d_y32)
    cuda.synchronize()
    gpu32_time = time.time() - start

    result32 = d_y32.copy_to_host()

    # ---------- FLOAT64 ----------
    d_x64 = cuda.to_device(x_cpu64)
    d_y64 = cuda.device_array_like(x_cpu64)

    start = time.time()
    poly_kernel_float64[blocks_per_grid, threads_per_block](d_x64, d_y64)
    cuda.synchronize()
    gpu64_time = time.time() - start

    result64 = d_y64.copy_to_host()

    print(f"GPU float32 Time : {gpu32_time:.6f} sec")
    print(f"GPU float64 Time : {gpu64_time:.6f} sec")

    print(f"Speedup float32  : {cpu32_time/gpu32_time:.2f}x")
    print(f"Speedup float64  : {cpu64_time/gpu64_time:.2f}x")

else:
    print("CUDA GPU not available.")


print("\n================ QUESTION 2 =================")

N = 1_000_000
BINS = 100

data = np.random.randint(0, BINS, size=N)

# ---------------- PURE PYTHON ----------------
def histogram_python(data, bins):
    hist = [0] * bins
    for value in data:
        hist[value] += 1
    return hist

start = time.time()
hist_py = histogram_python(data, BINS)
python_time = time.time() - start

# ---------------- NUMPY ----------------
start = time.time()
hist_np = np.histogram(data, bins=BINS, range=(0, BINS))[0]
numpy_time = time.time() - start

# ---------------- NUMBA ----------------
@njit
def histogram_numba(data, bins):
    hist = np.zeros(bins, dtype=np.int64)
    for i in range(len(data)):
        hist[data[i]] += 1
    return hist

# warmup
histogram_numba(data[:1000], BINS)

start = time.time()
hist_nb = histogram_numba(data, BINS)
numba_time = time.time() - start

print(f"Pure Python Time : {python_time:.6f} sec")
print(f"NumPy Time       : {numpy_time:.6f} sec")
print(f"Numba Time       : {numba_time:.6f} sec")

print("Correctness Check:")
print("Python vs NumPy :", np.all(np.array(hist_py) == hist_np))
print("NumPy vs Numba  :", np.all(hist_np == hist_nb))


print("\n================ QUESTION 3 =================")

NSAMPLES = 5_000_000

# ---------------- PURE PYTHON ----------------
def monte_carlo_pi_python(nsamples):
    inside = 0

    for _ in range(nsamples):
        x = np.random.random()
        y = np.random.random()

        if x*x + y*y < 1:
            inside += 1

    return 4 * inside / nsamples


# ---------------- NUMBA VERSION ----------------
@njit
def monte_carlo_pi_numba(nsamples):

    inside = 0

    for _ in range(nsamples):
        x = np.random.random()
        y = np.random.random()

        if x*x + y*y < 1:
            inside += 1

    return 4 * inside / nsamples


# Python timing
start = time.time()
pi_py = monte_carlo_pi_python(NSAMPLES)
python_pi_time = time.time() - start

# First execution (includes compilation)
start = time.time()
pi_nb_first = monte_carlo_pi_numba(NSAMPLES)
numba_first_time = time.time() - start

# Second execution (compiled already)
start = time.time()
pi_nb_second = monte_carlo_pi_numba(NSAMPLES)
numba_second_time = time.time() - start

print(f"Python PI Estimate       : {pi_py}")
print(f"Numba PI Estimate        : {pi_nb_second}")

print(f"Python Time              : {python_pi_time:.6f} sec")
print(f"Numba First Run Time     : {numba_first_time:.6f} sec")
print(f"Numba Second Run Time    : {numba_second_time:.6f} sec")

print(f"Speedup Factor           : {python_pi_time/numba_second_time:.2f}x")

print("\nWhy first execution is slower?")
print("-> Because Numba performs JIT compilation during the first call.")
print("-> Second execution uses already compiled machine code.")


print("\n================ QUESTION 4 =================")

N = 10_000_000

pixels = np.random.randint(0, 256, size=N).astype(np.int64)

# ---------------- NORMAL VECTORIZE ----------------
@vectorize(['int64(int64)'])
def adjust_brightness(pixel):
    value = int(pixel * 1.2)

    if value > 255:
        return 255
    return value


start = time.time()
bright_pixels = adjust_brightness(pixels)
normal_vec_time = time.time() - start

print(f"Vectorize Time           : {normal_vec_time:.6f} sec")


# ---------------- PARALLEL VECTORIZE ----------------
@vectorize(['int64(int64)'], target='parallel')
def adjust_brightness_parallel(pixel):
    value = int(pixel * 1.2)

    if value > 255:
        return 255
    return value


start = time.time()
bright_parallel = adjust_brightness_parallel(pixels)
parallel_time = time.time() - start

print(f"Parallel Vectorize Time  : {parallel_time:.6f} sec")
print(f"Speedup                  : {normal_vec_time/parallel_time:.2f}x")

print("\nWhat if a Python list is passed?")
print("-> Numba automatically converts the list into a NumPy array internally.")
print("-> But performance is slower compared to directly using NumPy arrays.")


print("\n================ QUESTION 5 =================")

# Synthetic data
samples = 100_000
features = 10

X = np.random.randn(samples, features)
y = np.random.choice([-1, 1], size=samples)

learning_rate = 0.01
epochs = 100

# ---------------- NUMPY VERSION ----------------
def logistic_regression_numpy(X, y, lr, epochs):

    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    for _ in range(epochs):

        z = X @ w

        gradient = -(X.T @ (y / (1 + np.exp(y * z)))) / n_samples

        w -= lr * gradient

    return w


# ---------------- NUMBA VERSION ----------------
@njit(parallel=True)
def logistic_regression_numba(X, y, lr, epochs):

    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    for _ in range(epochs):

        gradient = np.zeros(n_features)

        for i in prange(n_samples):

            z = 0.0

            for j in range(n_features):
                z += X[i, j] * w[j]

            coeff = -y[i] / (1 + math.exp(y[i] * z))

            for j in range(n_features):
                gradient[j] += coeff * X[i, j]

        for j in range(n_features):
            w[j] -= lr * gradient[j] / n_samples

    return w


# NumPy timing
start = time.time()
w_numpy = logistic_regression_numpy(X, y, learning_rate, epochs)
numpy_lr_time = time.time() - start

# Warmup compile
logistic_regression_numba(X[:100], y[:100], learning_rate, 1)

# Numba timing
start = time.time()
w_numba = logistic_regression_numba(X, y, learning_rate, epochs)
numba_lr_time = time.time() - start

print(f"NumPy Logistic Regression Time : {numpy_lr_time:.6f} sec")
print(f"Numba Logistic Regression Time : {numba_lr_time:.6f} sec")

print("Weights Close:",
      np.allclose(w_numpy, w_numba, atol=1e-2))

print(f"Speedup : {numpy_lr_time/numba_lr_time:.2f}x")


print("\n================ QUESTION 6 =================")

ROWS = 1024
COLS = 1024

A = np.random.rand(ROWS, COLS).astype(np.float32)
B = np.random.rand(ROWS, COLS).astype(np.float32)

# ---------------- CUDA KERNEL ----------------
@cuda.jit
def matrix_add_kernel(A, B, C):

    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]


if cuda.is_available():

    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((ROWS, COLS), dtype=np.float32)

    threads_per_block = (16, 16)

    blocks_per_grid_x = math.ceil(ROWS / threads_per_block[0])
    blocks_per_grid_y = math.ceil(COLS / threads_per_block[1])

    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start = time.time()

    matrix_add_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

    cuda.synchronize()

    gpu_matrix_time = time.time() - start

    C = d_C.copy_to_host()

    print(f"CUDA Matrix Addition Time : {gpu_matrix_time:.6f} sec")

    # Correctness check
    print("Correctness:",
          np.allclose(C, A + B))

else:
    print("CUDA GPU not available.")


print("\n================ END OF PROGRAM =================")