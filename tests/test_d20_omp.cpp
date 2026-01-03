#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <ostream>

struct D20Kernel {
    double operator()(mc::RNGState& rng) const {
        rng.counter *= 6364136223846793005ULL;
        const double u = (rng.counter >> 11) * (1.0 / (1ULL << 53));
        const int roll = static_cast<int>(u * 20.) + 1;
        return roll;
    }
};

int main() {
    constexpr uint64_t N = 100'000'000;

    constexpr mc::MCProblem<D20Kernel> problem(N);

    const double mean = mc::run(problem);

    std::cout << "mean = " << mean << std::endl;

    return 0;
}