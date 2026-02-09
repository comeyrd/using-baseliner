cmake_minimum_required(VERSION 3.15)
project(MyBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  baseliner
  GIT_REPOSITORY https://github.com/comeyrd/gpu-kernel-baseliner.git
  GIT_TAG        v0.2.1 
)
FetchContent_MakeAvailable(baseliner)
include(${baseliner_SOURCE_DIR}/cmake/utils.cmake)

#TEMPORARY FIX AS THROUGH PRESETS IT DOES NOT WORK
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
FetchContent_MakeAvailable(baseliner)
set(sources main.cu MatMul.cu)
add_executable(my_benchmark)
target_sources(my_benchmark PRIVATE ${sources})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

baseliner_enable_git_version(my_benchmark)

target_link_libraries(my_benchmark PRIVATE baseliner::baseliner)
