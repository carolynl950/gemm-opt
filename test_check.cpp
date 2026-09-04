#include "check.hpp"
#include "alloc.hpp"

const int R = 4, C = 4;

void fill(float* m) {
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) {
            int i = idx(r, c, C);
            m[i] = static_cast<float>(r * C + c);
        }
    }
}

// identical matrices -> expect matrices_match to return true
bool t1() {
    // TODO: allocate two, fill both, compare, free, return the result
    float* ref = alloc_matrix(R, C);
    float* test = alloc_matrix(R, C);
    fill(ref);
    fill(test);
    bool ok = matrices_match(ref, test, R, C, C, C);
    free_matrix(ref);
    free_matrix(test);
    return ok;
}

// one element corrupted -> expect matrices_match to return false
bool t2() {
    // TODO: allocate two, fill both, damage one element of test,
    //       compare, free, return the result
    float* ref = alloc_matrix(R, C);
    float* test = alloc_matrix(R, C);
    fill(ref);
    fill(test);
    test[idx(0, 0, C)] += 100.0f;
    bool ok = matrices_match(ref, test, R, C, C, C);
    free_matrix(ref);
    free_matrix(test);
    return ok;

}

// same sum in a different order -> expect matrices_match to return true
bool t3() {
    const int K = 10000;
    float vals[K];
    for (int i = 0; i < K; i++) {
        vals[i] = (i <= K/2 ? 1e6f : 1.0f);
    }
    float fwd = 0;
    float bwd = 0;
    for (int i = 0; i < K; i++) {
        fwd += vals[i];
        bwd += vals[K - i - 1];
    }
    printf("fwd %.10g, bwd %.10g\n", fwd, bwd);
    float* ref = alloc_matrix(1, 1);
    float* test = alloc_matrix(1, 1);
    ref[idx(0, 0, 1)] = fwd;
    test[idx(0, 0, 1)] = bwd;
    bool ok = matrices_match(ref, test, 1, 1, 1, 1);
    free_matrix(ref);
    free_matrix(test);
    return ok;
}

int main() {
    printf("test 1 (identical):   %s\n", t1()  ? "PASS" : "FAIL");
    printf("test 2 (corrupted):   %s\n", !t2()   ? "PASS" : "FAIL");
    printf("test 3 (reordered):   %s\n", t3()  ? "PASS" : "FAIL");
    return 0;
}