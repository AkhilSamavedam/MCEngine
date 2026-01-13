#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <chrono>
#include <ostream>

using namespace mc;

MC_KERNEL(DiceUntil6Kernel, (MC_RNG(rng)),
    int sum = 0;
    while (true) {
        const int roll = rng.next_int(1, 6);
        sum += roll;
        if (roll == 6) {
            break;
        }
    }
    return static_cast<double>(sum);
)

inline double rolls_per_second(uint64_t n_rolls,
                                std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end)
{
    const std::chrono::duration<double> elapsed = end - start;
    return static_cast<double>(n_rolls) / elapsed.count();
}

int main() {
    constexpr uint64_t N = 100'000'000;
    const MCProblem<DiceUntil6Kernel> problem(N);

    using clock = std::chrono::steady_clock;

    auto t0 = clock::now();
    const double mean = mc::run(problem);
    auto t1 = clock::now();

    const double rps = rolls_per_second(N, t0, t1);

    std::cout << "Dice until 6 (OMP)\n";
    std::cout << "Rolls/sec = " << rps << "\n";
    std::cout << "Mean      = " << mean << "\n\n";

    return 0;
}
