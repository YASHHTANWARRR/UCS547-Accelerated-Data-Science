```cpp
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// ================= GPU KERNEL =================
__global__ void vectorAdd(int *A, int *B, int *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N)
    {
        C[i] = A[i] + B[i];
    }
}

// ================= HOST CODE =================
int main()
{
    int h_A[N], h_B[N], h_C[N];

    int *d_A, *d_B, *d_C;

    // Initialize vectors
    for(int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy Host -> Device
    cudaMemcpy(d_A, h_A, N * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B, N * sizeof(int),
               cudaMemcpyHostToDevice);

    // Launch kernel
    vectorAdd<<<4, 256>>>(d_A, d_B, d_C);

    // Wait for GPU
    cudaDeviceSynchronize();

    // Copy Device -> Host
    cudaMemcpy(h_C, d_C, N * sizeof(int),
               cudaMemcpyDeviceToHost);

    // Print first 10 elements
    for(int i = 0; i < 10; i++)
    {
        printf("%d + %d = %d\n",
               h_A[i],
               h_B[i],
               h_C[i]);
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}