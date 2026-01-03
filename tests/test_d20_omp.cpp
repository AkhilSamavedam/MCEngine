#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <chrono>
#include <ostream>

using namespace mc;

struct D20Kernel {
    #pragma omp declare simd
    __attribute__((always_inline))
    double operator()(RNGView& rng) const {
        const double u = rng.next_u01();
        return static_cast<int>(u * 20.0) + 1;
    }
};

inline double rolls_per_second(uint64_t n_rolls,
                                std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end)
{
    const std::chrono::duration<double> elapsed = end - start;
    return static_cast<double>(n_rolls) / elapsed.count();
}


int main() {
    constexpr uint64_t N = 1'000'000'000;

    constexpr MCProblem<D20Kernel> problem(N);

    using clock = std::chrono::steady_clock;

    auto t0 = clock::now();
    double mean = mc::run(problem);
    auto t1 = clock::now();

    double rps = rolls_per_second(N, t0, t1);

    std::cout << "Rolls/sec = " << rps << "\n";
    std::cout << "Mean      = " << mean << "\n";

    return 0;
}