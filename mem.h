#pragma once
#include <stdexcept>
#include <vector>

template<typename T>
T *CheckedCudaMalloc(std::size_t n) {
    T *a{nullptr};
    std::size_t sz{sizeof(T) * n};
    if (cudaError err{cudaMalloc(&a, sz)}; err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return a;
}

template<typename T>
void CheckedCopyToDevice(T* dest, const std::vector<T>& src) {
    std::size_t count{sizeof(T) * src.size()}; // count in bytes
    if (cudaError err{cudaMemcpy(dest, src.data(), count, cudaMemcpyHostToDevice)}; err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template<typename T>
void CheckedCopyToHost(std::vector<T>& dest, const T* src) {
    // assumes vector is already correctly sized!
    std::size_t count{sizeof(T) * dest.size()}; // count in bytes
    if (cudaError err{cudaMemcpy(dest.data(), src, count, cudaMemcpyDeviceToHost)}; err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}