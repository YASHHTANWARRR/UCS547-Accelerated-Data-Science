#include <iostream>
#include <cuda_runtime.h>

#define N 5000000

__global__ void add(int *A,
                    int *B,
                    int *C)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if(i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    int *h_A, *h_B, *h_C;

    int *d_A, *d_B, *d_C;

    size_t size = N * sizeof(int);

    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    for(int i = 0; i < N; i++)
    {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size,
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B, size,
               cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    add<<<(N+255)/256,256>>>(d_A,
                             d_B,
                             d_C);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;

    cudaEventElapsedTime(&ms,
                         start,
                         stop);

    cout << "CUDA Time: "
         << ms
         << " ms\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}