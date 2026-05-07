#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

using namespace std;

int main()
{
    thrust::device_vector<int> vec(10);

    for(int i = 0; i < 10; i++)
    {
        vec[i] = i + 1;
    }

    int sum =
        thrust::reduce(vec.begin(),
                       vec.end(),
                       0,
                       thrust::plus<int>());

    cout << "Sum = "
         << sum << endl;

    return 0;
}