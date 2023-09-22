#pragma once
#include <stdexcept>

template<typename T>
T *SafeCudaMalloc(std::size_t n) {
    T *a{nullptr};
    std::size_t sz{sizeof(a) * n};
    if (cudaError err{cudaMalloc(&a, sz)}; err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return a;
}