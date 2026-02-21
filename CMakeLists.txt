cmake_minimum_required(VERSION 3.18)
project(chroma-radiance LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Auto-detect GPU architecture, fallback to common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# Main executable
add_executable(chroma-radiance main.cu)

target_include_directories(chroma-radiance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# cuBLAS
find_package(CUDAToolkit REQUIRED)
target_link_libraries(chroma-radiance PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cudart)

# Optimization flags
target_compile_options(chroma-radiance PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
        -Xcompiler=-O3
    >
)

# Large model - need to allow large object files
target_compile_options(chroma-radiance PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
    >
)
