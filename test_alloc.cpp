// test_alloc.cpp
#include "alloc.hpp"
#include <cassert>
#include <iostream>

int main() {
    const int R = 4, C = 4;
    float* m = alloc_matrix(R, C);
    assert(m != nullptr);

    // 32-byte alignment check
    assert(reinterpret_cast<uintptr_t>(m) % 32 == 0);

    // TODO: fill m so that element (i,j) holds the value i*10 + j,
    //       using idx() — not raw arithmetic
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            m[idx(r, c, C)] = static_cast<float>(r * 10 + c);
        }
    }
    // TODO: print the 4x4 as a grid and eyeball that row 2 reads 20 21 22 23
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            printf("%5.1f", m[idx(r, c, C)]);
        }
        printf("\n");
    }

    free_matrix(m);
    printf("ok\n");
    return 0;
}