#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace chrono;

int main()
{
    int N = 5000000;

    vector<int> A(N, 1);
    vector<int> B(N, 2);
    vector<int> C(N);

    auto start =
        high_resolution_clock::now();

    for(int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }

    auto end =
        high_resolution_clock::now();

    auto duration =
        duration_cast<milliseconds>(
            end - start);

    cout << "CPU Time: "
         << duration.count()
         << " ms\n";

    return 0;
}