cmake_minimum_required(VERSION 3.25)
project(gemm_opt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # compile_commands.json for clangd / the IDE

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GEMM_NATIVE "Tune for the build machine's CPU (-march=native / -mcpu=native)" ON)
set(GEMM_SANITIZER "" CACHE STRING "Sanitizer to build with: '', 'address', or 'undefined'")
option(GEMM_USE_BLAS "Build the vendor-BLAS reference kernel if a BLAS is found" ON)
option(GEMM_USE_OPENMP "Build the multithreaded kernel if OpenMP is found (macOS: brew install libomp)" ON)
option(GEMM_USE_CUDA "Build the CUDA kernels if nvcc is found (Colab / NVIDIA machines)" ON)
option(GEMM_VECTORIZE "Allow compiler auto-vectorization (OFF = experiment: isolate loop-order gains from SIMD)" ON)

# ---- flags applied to every C++ target ---------------------------------------
# $<COMPILE_LANGUAGE:CXX> keeps host-compiler flags away from nvcc, which rejects them.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

if(GEMM_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native HAVE_MARCH_NATIVE)
    check_cxx_compiler_flag(-mcpu=native  HAVE_MCPU_NATIVE)
    if(HAVE_MARCH_NATIVE)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)      # x86: enables AVX2/AVX-512 if the CPU has them
    elseif(HAVE_MCPU_NATIVE)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mcpu=native>)       # Apple clang on arm64
    endif()
endif()

if(NOT GEMM_VECTORIZE)
    # clang: loop vectorizer + SLP (straight-line) vectorizer. gcc spelling is -fno-tree-vectorize.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-vectorize> $<$<COMPILE_LANGUAGE:CXX>:-fno-slp-vectorize>)
endif()

if(GEMM_SANITIZER STREQUAL "address")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
    add_link_options(-fsanitize=address)
elseif(GEMM_SANITIZER STREQUAL "undefined")
    # no-recover: UB aborts with nonzero exit instead of printing and carrying on
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined> $<$<COMPILE_LANGUAGE:CXX>:-fno-sanitize-recover=all> $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
    add_link_options(-fsanitize=undefined)
elseif(NOT GEMM_SANITIZER STREQUAL "")
    message(FATAL_ERROR "GEMM_SANITIZER must be '', 'address', or 'undefined'")
endif()

# ---- kernels -----------------------------------------------------------------
add_library(gemm STATIC
    src/kernels/naive_ijk.cpp
    src/kernels/reorder_ijk.cpp
    src/kernels/blocked.cpp
    src/kernels/simd.cpp
)
target_include_directories(gemm PUBLIC include)

if(GEMM_USE_OPENMP)
    # Apple clang has no bundled OpenMP runtime; point FindOpenMP at Homebrew's libomp.
    if(APPLE AND NOT DEFINED OpenMP_ROOT)
        execute_process(COMMAND brew --prefix libomp
                        OUTPUT_VARIABLE _gemm_libomp OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_gemm_libomp AND EXISTS "${_gemm_libomp}/lib")
            set(OpenMP_ROOT "${_gemm_libomp}")
        endif()
    endif()
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gemm PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(gemm PUBLIC GEMM_HAVE_OPENMP)
        message(STATUS "OpenMP kernel: ON")
    else()
        message(STATUS "OpenMP kernel: OFF (OpenMP not found)")
    endif()
endif()

if(GEMM_USE_BLAS)
    if(APPLE)
        set(BLA_VENDOR Apple)       # Accelerate
    else()
        set(BLA_VENDOR OpenBLAS)
    endif()
    find_package(BLAS)
    if(BLAS_FOUND)
        target_sources(gemm PRIVATE src/kernels/blas.cpp)
        target_link_libraries(gemm PUBLIC BLAS::BLAS)
        target_compile_definitions(gemm PUBLIC GEMM_HAVE_BLAS)
        message(STATUS "BLAS reference kernel: ON (${BLA_VENDOR})")
    else()
        message(STATUS "BLAS reference kernel: OFF (no ${BLA_VENDOR} found)")
    endif()
endif()

# ---- benchmark ---------------------------------------------------------------
add_executable(bench src/bench.cpp)
target_link_libraries(bench PRIVATE gemm)

# FMA throughput microbenchmark: the real per-core ceiling for GEMM (no kernels, no memory)
add_executable(peak src/peak.cpp)

# ---- tests -------------------------------------------------------------------
enable_testing()

foreach(t test_alloc test_check test_naive test_kernels)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE gemm)
    # Release adds -DNDEBUG, which deletes assert(). Tests must keep their asserts.
    target_compile_options(${t} PRIVATE -UNDEBUG)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# ---- CUDA (Stage 5) ----------------------------------------------------------
# Built only where nvcc exists (Colab, NVIDIA machines); skipped quietly on the Mac.
if(GEMM_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)   # compile for the GPU in this machine (T4 = sm_75)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)

        add_library(gemm_cuda STATIC
            cuda/cublas_ref.cu
            cuda/kernels/naive.cu
            cuda/kernels/coalesced.cu
            cuda/kernels/smem.cu
        )
        target_include_directories(gemm_cuda PUBLIC cuda)
        target_link_libraries(gemm_cuda PUBLIC CUDA::cudart CUDA::cublas)
        # -lineinfo: maps profiler (Nsight Compute) results back to source lines; no runtime cost
        target_compile_options(gemm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

        add_executable(bench_cuda cuda/bench_cuda.cpp)
        target_link_libraries(bench_cuda PRIVATE gemm gemm_cuda)

        add_executable(test_cuda cuda/test_cuda.cpp)
        target_link_libraries(test_cuda PRIVATE gemm gemm_cuda)
        target_compile_options(test_cuda PRIVATE -UNDEBUG)
        add_test(NAME test_cuda COMMAND test_cuda)
        message(STATUS "CUDA kernels: ON (arch ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "CUDA kernels: OFF (no nvcc found)")
    endif()
endif()
