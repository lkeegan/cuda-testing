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
    float *d_A{SafeCudaMalloc<float>(N)};
    float *d_B{SafeCudaMalloc<float>(N)};
    float *d_C{SafeCudaMalloc<float>(N)};

    // copy local values to device (count is in bytes!)
    cudaMemcpy(d_A, A.data(), A.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), B.size()*sizeof(float), cudaMemcpyHostToDevice);

    // both threads and blocks are up to 3d indexed
    // max number of threads per thread block is 1024
    // here we use 1-d for both, with 256 threads per block
    // also ignores padding issues, i.e. assumes N is divisible by 256
    dim3 threadsPerBlock{256, 1, 1};
    dim3 numBlocks{N/threadsPerBlock.x, 1, 1};
	VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // copy values back from device (count is in bytes!)
    cudaMemcpy(C.data(), d_C, C.size()*sizeof(float), cudaMemcpyDeviceToHost);

    // free device data
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // print a couple of elements of output array
    std::cout << C[0] << " ... " << C[N/2] << " ... " << C[N-1] << std::endl;
}