CXX = clang++ 
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

all: test_alloc test_check naive_ijk

naive_ijk: src/naive_ijk.cpp alloc.hpp check.hpp
	$(CXX) $(CXXFLAGS) -I. src/naive_ijk.cpp -o naive_ijk

test_alloc: test_alloc.cpp alloc.hpp
	$(CXX) $(CXXFLAGS) test_alloc.cpp -o test_alloc

test_check: test_check.cpp check.hpp
	$(CXX) $(CXXFLAGS) test_check.cpp -o test_check

clean:
	rm -f test_alloc test_check naive_ijk