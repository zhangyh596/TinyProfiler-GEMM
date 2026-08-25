cmake_minimum_required(VERSION 3.16)
project(TinyProfilerGEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gemm main.cpp)

# 编译加速: 激活 CPU 本机向量化指令集 (AVX2/FMA) 与 FMA 乘加融合。
# -O3 显式给出, 保证即使不指定 CMAKE_BUILD_TYPE=Release 也保持优化。
if(MSVC)
    target_compile_options(gemm PRIVATE /O2 /arch:AVX2 /fp:fast)
else()
    target_compile_options(gemm PRIVATE -O3 -march=native -ffast-math)
endif()

# OpenMP 多线程 (Phase 7)。默认开启; 找不到 OpenMP 时自动退化为单线程。
option(GEMM_OPENMP "Enable OpenMP multithreading (Phase 7)" ON)
if(GEMM_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gemm PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
