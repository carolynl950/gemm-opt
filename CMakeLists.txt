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
option(GEMM_VECTORIZE "Allow compiler auto-vectorization (OFF = experiment: isolate loop-order gains from SIMD)" ON)

# ---- flags applied to every target -------------------------------------------
add_compile_options(-Wall -Wextra)

if(GEMM_NATIVE)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native HAVE_MARCH_NATIVE)
    check_cxx_compiler_flag(-mcpu=native  HAVE_MCPU_NATIVE)
    if(HAVE_MARCH_NATIVE)
        add_compile_options(-march=native)      # x86: enables AVX2/AVX-512 if the CPU has them
    elseif(HAVE_MCPU_NATIVE)
        add_compile_options(-mcpu=native)       # Apple clang on arm64
    endif()
endif()

if(NOT GEMM_VECTORIZE)
    # clang: loop vectorizer + SLP (straight-line) vectorizer. gcc spelling is -fno-tree-vectorize.
    add_compile_options(-fno-vectorize -fno-slp-vectorize)
endif()

if(GEMM_SANITIZER STREQUAL "address")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
elseif(GEMM_SANITIZER STREQUAL "undefined")
    # no-recover: UB aborts with nonzero exit instead of printing and carrying on
    add_compile_options(-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
elseif(NOT GEMM_SANITIZER STREQUAL "")
    message(FATAL_ERROR "GEMM_SANITIZER must be '', 'address', or 'undefined'")
endif()

# ---- kernels -----------------------------------------------------------------
add_library(gemm STATIC
    src/kernels/naive_ijk.cpp
    src/kernels/reorder_ijk.cpp
)
target_include_directories(gemm PUBLIC include)

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

# ---- tests -------------------------------------------------------------------
enable_testing()

foreach(t test_alloc test_check test_naive test_kernels)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE gemm)
    # Release adds -DNDEBUG, which deletes assert(). Tests must keep their asserts.
    target_compile_options(${t} PRIVATE -UNDEBUG)
    add_test(NAME ${t} COMMAND ${t})
endforeach()
