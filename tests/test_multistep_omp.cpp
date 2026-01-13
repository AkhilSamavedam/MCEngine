#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <mc_path.h>
#include <chrono>
#include <ostream>

using namespace mc;

MC_STATE(RandomWalkState,
    double x;
)

MC_STEP_KERNEL(RandomWalkStep, (RandomWalkState& st, RNGView& rng, double dt),
    const double u = rng.next_u01();
    const double step = (u - 0.5) * dt;
    st.x += step;
)

MC_PAYOFF_KERNEL(RandomWalkPayoff, (const RandomWalkState& st),
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

    const RandomWalkState init{0.0};
    const MCPathProblem problem(
        init,
        RandomWalkStep{},
        RandomWalkPayoff{},
        N,
        steps,
        dt
    );

    using clock = std::chrono::steady_clock;

    auto t0 = clock::now();
    const double mean = mc::run_paths(problem, mc::OMPBackend{});
    auto t1 = clock::now();

    const double rps = rolls_per_second(N, t0, t1);

    std::cout << "Random walk (OMP)\n";
    std::cout << "Rolls/sec = " << rps << "\n";
    std::cout << "Mean      = " << mean << "\n\n";

    return 0;
}
