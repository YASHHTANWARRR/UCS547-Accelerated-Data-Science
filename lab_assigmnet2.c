

// 4
#include <stdio.h>

// ================= DEVICE CODE (GPU) =================
__global__ void helloKernel()
{
    // Compute global thread ID
    int global_thread_id =
        blockIdx.x * blockDim.x + threadIdx.x;

    // Print from GPU thread
    printf("Hello from GPU thread %d\n",
           global_thread_id);
}

// ================= HOST CODE (CPU) =================
int main()
{
    printf("Launching CUDA Kernel...\n");

    // Launch kernel
    // <<<number_of_blocks, threads_per_block>>>
    helloKernel<<<1, 8>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    printf("Kernel execution completed.\n");

    return 0;
}

//5
#include <stdio.h>
#include <cuda_runtime.h>

// ================= DEVICE CODE =================
__global__ void printDeviceArray(int *d_array)
{
    int idx = threadIdx.x;

    printf("GPU Thread %d -> Value = %d\n",
           idx,
           d_array[idx]);
}

// ================= HOST CODE =================
int main()
{
    // Host array (CPU memory)
    int h_array[5] = {10, 20, 30, 40, 50};

    // Device pointer
    int *d_array;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_array,
               5 * sizeof(int));

    // Copy data from Host to Device
    cudaMemcpy(d_array,
               h_array,
               5 * sizeof(int),
               cudaMemcpyHostToDevice);

    printf("Launching Kernel...\n");

    // Launch kernel
    printDeviceArray<<<1, 5>>>(d_array);

    // Wait for GPU completion
    cudaDeviceSynchronize();

    // Copy data back to Host
    cudaMemcpy(h_array,
               d_array,
               5 * sizeof(int),
               cudaMemcpyDeviceToHost);

    printf("\nCopied Back to CPU:\n");

    for(int i = 0; i < 5; i++)
    {
        printf("h_array[%d] = %d\n",
               i,
               h_array[i]);
    }

    // Free GPU memory
    cudaFree(d_array);

    return 0;
}

//6
import time
import numpy as np

N = 10000000

# ================= LIST =================
lst = list(range(N))

start = time.time()

lst_squared = [x * x for x in lst]

end = time.time()

print("List Time:", end - start)

# ================= TUPLE =================
tpl = tuple(range(N))

start = time.time()

tpl_squared = tuple(x * x for x in tpl)

end = time.time()

print("Tuple Time:", end - start)

# ================= NUMPY ARRAY =================
arr = np.arange(N)

start = time.time()

arr_squared = arr * arr

end = time.time()

print("NumPy Time:", end - start)
