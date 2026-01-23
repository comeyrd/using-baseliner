cmake_minimum_required(VERSION 3.15)
project(MyBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  baseliner
  GIT_REPOSITORY https://github.com/comeyrd/gpu-kernel-baseliner.git
  GIT_TAG        v0.1
)

FetchContent_MakeAvailable(baseliner)

add_executable(my_benchmark 
    main.cu
    MatMul.cu
)

target_link_libraries(my_benchmark PRIVATE baseliner::baseliner)