# Links

## To build and run

Usual CMake, need nvcc installed:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
ninja
./main
```

## To debug

- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#using-compute-sanitizer)
  - `compute-sanitizer --leak-check full ./main`
  - `compute-sanitizer --tool memcheck ./main`
  - `compute-sanitizer --tool racecheck ./main`
  - `compute-sanitizer --tool initcheck ./main`
  - `compute-sanitizer --tool synccheck ./main`
  - https://github.com/NVIDIA/compute-sanitizer-samples


## general

- https://docs.nvidia.com/cuda/cuda-c-programming-guide
- https://github.com/NVIDIA-developer-blog/code-samples

## cmake

- https://on-demand.gputechconf.com/gtc/2017/presentation/S7438-robert-maynard-build-systems-combining-cuda-and-machine-learning.pdf
- https://developer.nvidia.com/blog/building-cuda-applications-cmake/
