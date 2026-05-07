#include <iostream>
#include <vector>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

using namespace std;
using namespace chrono;

#define N 1024

int main()
{
    // ================= CPU =================
    vector<int> h_A(N);
    vector<int> h_B(N);

    for(int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i;
    }

    auto start_cpu = high_resolution_clock::now();

    long long cpu_result = 0;

    for(int i = 0; i < N; i++)
    {
        cpu_result += h_A[i] * h_B[i];
    }

    auto end_cpu = high_resolution_clock::now();

    // ================= GPU THRUST =================
    thrust::device_vector<int> d_A = h_A;
    thrust::device_vector<int> d_B = h_B;

    auto start_gpu = high_resolution_clock::now();

    int gpu_result =
        thrust::inner_product(d_A.begin(),
                              d_A.end(),
                              d_B.begin(),
                              0);

    auto end_gpu = high_resolution_clock::now();

    // ================= OUTPUT =================
    cout << "CPU Result: "
         << cpu_result << endl;

    cout << "GPU Result: "
         << gpu_result << endl;

    auto cpu_time =
        duration_cast<microseconds>(
            end_cpu - start_cpu);

    auto gpu_time =
        duration_cast<microseconds>(
            end_gpu - start_gpu);

    cout << "CPU Time: "
         << cpu_time.count()
         << " microseconds\n";

    cout << "GPU Time: "
         << gpu_time.count()
         << " microseconds\n";

    return 0;
}