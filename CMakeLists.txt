cmake_minimum_required(VERSION 3.17...3.27)

project(CUDA_TESTING LANGUAGES CUDA CXX)

add_library(mem mem.cu mem.h)

add_library(mul mul.cu mul.h)

add_executable(main main.cu)

target_link_libraries(main PRIVATE mul)