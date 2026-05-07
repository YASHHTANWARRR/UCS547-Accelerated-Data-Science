#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

using namespace std;

int main()
{
    const int N = 1024;

    thrust::device_vector<int> A(N);
    thrust::device_vector<int> B(N);
    thrust::device_vector<int> C(N);

    // Initialize
    for(int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = i * 2;
    }

    // Vector addition
    thrust::transform(A.begin(),
                      A.end(),
                      B.begin(),
                      C.begin(),
                      thrust::plus<int>());

    // Print first 10 elements
    for(int i = 0; i < 10; i++)
    {
        cout << A[i] << " + "
             << B[i] << " = "
             << C[i] << endl;
    }

    return 0;
}