#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <mc_backend.hpp>
#include <ostream>

namespace mc {
    struct RNGState;
}

struct D20Kernel {
    double operator()(mc::RNGState& rng) const {
        rng.counter *= 6364136223846793005ULL;
        const double u = (rng.counter >> 11) * (1.0 / (1ULL << 53));
        const int roll = static_cast<int>(u * 20.) + 1;
        std::cout << "roll = " << roll << std::endl;
        return roll;
    }
};

int main() {
    constexpr uint64_t N = 10;

    constexpr mc::MCProblem problem = {D20Kernel{}, N};

    const double mean = mc::run(problem);

    std::cout << "mean = " << mean << std::endl;

}