#include "mul.h"
#include "mem.h"
#include <iostream>
#include <vector>

int main(){
	unsigned int N{256*1024*1024};

    // local data
	std::vector<float> A(N, 1.0);
	std::vector<float> B(N, 2.0);
	std::vector<float> C(N, -99.0);

    // allocate device data
    float *d_A{CheckedCudaMalloc<float>(N)};
    float *d_B{CheckedCudaMalloc<float>(N)};
    float *d_C{CheckedCudaMalloc<float>(N)};

    // copy local values to device
    CheckedCopyToDevice(d_A, A);
    CheckedCopyToDevice(d_B, B);

    // both threads and blocks are up to 3d indexed
    // max number of threads per thread block is 1024
    // here we use 1-d for both, with 256 threads per block
    // also ignores padding issues, i.e. assumes N is divisible by 256
    dim3 threadsPerBlock{256, 1, 1};
    dim3 numBlocks{N/threadsPerBlock.x, 1, 1};

    // launch kernel
	VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // copy values back from device
    CheckedCopyToHost(C, d_C);

    // free device data
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // print a couple of elements of output array
    std::cout << C[0] << " ... " << C[N/2] << " ... " << C[N-1] << std::endl;
}