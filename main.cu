#include "mul.h"
#include <iostream>
#include <vector>

int main(){
	unsigned int N{1024*1024*1024};
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

    // invoke kernel
	VecAdd<<<1, N>>>(d_A, d_B, d_C);

    // copy device values to local (count is in bytes!)
    cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // free device data
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    exit(0);
    // print results
	for (const auto& c : C){
    	std::cout << c << std::endl;
	}
}