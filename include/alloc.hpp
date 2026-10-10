#pragma once

#include <cstdlib> 
#include <cstdio>
#include <cstdint>

// gets index of flat array since 2D matrix doesnt exist in memory
// ld = row width
inline int idx(int row, int col, int ld) {
    return row * ld + col;
}

inline float* alloc_matrix(int rows, int cols) {
    void* p = nullptr;
    if (posix_memalign(&p, 32, rows * cols * sizeof(float))) {
        return nullptr;
    }
    return static_cast<float*>(p);
}

inline void free_matrix(float* p) {
    free(p);
}