#include <mc_engine.hpp>
#include <cmath>
#include <chrono>
#include <iostream>

using namespace mc;

int main() {
    constexpr uint64_t N = 10'000'000;
    constexpr uint32_t steps = 256;
    constexpr double dt = 0.01;

    auto brownian_kernel = [] MC (mc::RNGView& rng) -> double {
        double x = 0.0;
        for (uint32_t i = 0; i < steps; ++i) {
            const double z = rng.next_normal();
            x += z * ::sqrt(dt);
        }
        return x; // payoff = terminal state
    };

    const Problem problem(brownian_kernel, N);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const double mean = mc::run(problem);
    const auto t1 = clock::now();

    const std::chrono::duration<double> elapsed = t1 - t0;
    const double paths_per_sec = static_cast<double>(N) / elapsed.count();

    std::cout << "Brownian motion (simple kernel)\n";
    std::cout << "Paths/sec = " << paths_per_sec << "\n";
    std::cout << "Mean      = " << mean << "\n\n";
}
