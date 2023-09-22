#include "mul.h"

__global__ void VecAdd(const float* A, const float* B, float* C)
{
    unsigned int i{blockIdx.x*blockDim.x + threadIdx.x};
    C[i] = A[i] + B[i];
}
