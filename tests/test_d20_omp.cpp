#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <rng/splitmix64.h>
#include <rng/config.h>
#include <ostream>

using namespace mc;

struct D20Kernel {
    __attribute__((always_inline))
    double operator()(const RNGState& rng) const {
        const uint64_t x = rng.index ^ rng.seed * 0x9e3779b97f4a7c15ULL;
        const uint64_t r = splitmix64(x);

        const double u = u01_from_u64(r);
        return static_cast<int>(u * 20.0) + 1;
    }
};

int main() {
    constexpr uint64_t N = 1'000'000'000;

    constexpr MCProblem<D20Kernel> problem(N);

    BASE_SEED = 777;

    const double mean = run(problem);

    std::cout << "mean = " << mean << std::endl;

    return 0;
}