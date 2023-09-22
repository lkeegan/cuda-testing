#include "mul.h"
#include <iostream>
#include <vector>

int main(){
	unsigned int N{256*1024*1024*2};
    std::size_t size = N * sizeof(float);

    // local data
	std::vector<float> A(N, 1.0);
	std::vector<float> B(N, 2.0);
	std::vector<float> C(N, -99.0);

    // allocate device data
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);

    // copy local values to device (count is in bytes!)
    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    // both threads and blocks can be 1-2,2-d or 3-d indexed
    // max number of threads per thread block is 1024
    // here we use 1-d for both, with 256 threads per block
    // also ignores padding issues, i.e. assumes N is divisible by 256
    dim3 threadsPerBlock{256, 1, 1};
    dim3 numBlocks{N/threadsPerBlock.x, 1, 1};
	VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // copy device values to local (count is in bytes!)
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // free device data
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // print a couple of elements of output array
    std::cout << C[0] << " ... " << C[N/2] << " ... " << C[N-1] << std::endl;
}