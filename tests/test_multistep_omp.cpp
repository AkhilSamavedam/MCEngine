#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <mc_path.h>
#include <chrono>
#include <ostream>
#include <cmath>

using namespace mc;

MC_STATE(BrownianState,
    double x;
)

MC_STEP_KERNEL(BrownianStep, (BrownianState& st, RNGView& rng, double dt),
    const double z = rng.next_normal();
    st.x += z * ::sqrt(dt);
)

MC_PAYOFF_KERNEL(BrownianPayoff, (const BrownianState& st),
    return st.x;
)

inline double rolls_per_second(uint64_t n_rolls,
                                std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end)
{
    const std::chrono::duration<double> elapsed = end - start;
    return static_cast<double>(n_rolls) / elapsed.count();
}

int main() {
    constexpr uint64_t N = 50'000'000;
    constexpr uint32_t steps = 64;
    constexpr double dt = 0.1;

    const BrownianState init{0.0};
    const MCPathProblem problem(
        init,
        BrownianStep{},
        BrownianPayoff{},
        N,
        steps,
        dt
    );

    using clock = std::chrono::steady_clock;

    auto t0 = clock::now();
    const double mean = mc::run_paths(problem);
    auto t1 = clock::now();

    const double rps = rolls_per_second(N, t0, t1);

    std::cout << "Brownian motion (OMP)\n";
    std::cout << "Rolls/sec = " << rps << "\n";
    std::cout << "Mean      = " << mean << "\n\n";

    return 0;
}
