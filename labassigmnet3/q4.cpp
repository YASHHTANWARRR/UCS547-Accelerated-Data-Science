#include <stdio.h>
#include <cuda_runtime.h>

#define N 16

// ================= GPU KERNEL =================
__global__ void matrixMul(int A[N][N],
                          int B[N][N],
                          int C[N][N])
{
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;

    for(int k = 0; k < N; k++)
    {
        sum += A[row][k] * B[k][col];
    }

    C[row][col] = sum;
}

// ================= HOST CODE =================
int main()
{
    int A[N][N], B[N][N], C[N][N];

    int (*d_A)[N], (*d_B)[N], (*d_C)[N];

    // Initialize matrices
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            A[i][j] = 1;
            B[i][j] = 2;
        }
    }

    // Allocate GPU memory
    cudaMalloc(&d_A, sizeof(A));
    cudaMalloc(&d_B, sizeof(B));
    cudaMalloc(&d_C, sizeof(C));

    // Copy Host -> Device
    cudaMemcpy(d_A, A, sizeof(A),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, B, sizeof(B),
               cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads(N, N);

    matrixMul<<<1, threads>>>(d_A,
                              d_B,
                              d_C);

    cudaDeviceSynchronize();

    // Copy Device -> Host
    cudaMemcpy(C, d_C, sizeof(C),
               cudaMemcpyDeviceToHost);

    // Print sample output
    printf("C[0][0] = %d\n", C[0][0]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}