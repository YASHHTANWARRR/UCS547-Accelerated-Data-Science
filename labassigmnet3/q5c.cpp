#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace std;

int main()
{
    int N = 5000000;

    thrust::device_vector<int> A(N, 1);
    thrust::device_vector<int> B(N, 2);
    thrust::device_vector<int> C(N);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    thrust::transform(A.begin(),
                      A.end(),
                      B.begin(),
                      C.begin(),
                      thrust::plus<int>());

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;

    cudaEventElapsedTime(&ms,
                         start,
                         stop);

    cout << "Thrust Time: "
         << ms
         << " ms\n";

    return 0;
}