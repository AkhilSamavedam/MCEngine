#include <cstdint>
#include <iostream>
#include <mc_engine.hpp>
#include <chrono>
#include <ostream>

using namespace mc;

const auto D20Kernel1 = [](uint32_t rnd) -> double {
    const double u = u01(rnd);
    return static_cast<int>(u * 20.0) + 1;
};

const auto D20Kernel2 = [](uint32_t rnd, uint32_t rnd2) -> double {
    (void)rnd2;
    const double u = u01(rnd);
    return static_cast<int>(u * 20.0) + 1;
};

const auto D20Kernel3 = [](uint32_t rnd, uint32_t rnd2, uint32_t rnd3) -> double {
    (void)rnd2;
    (void)rnd3;
    const double u = u01(rnd);
    return static_cast<int>(u * 20.0) + 1;
};

const auto D20Kernel4 = [](uint32_t rnd, uint32_t rnd2, uint32_t rnd3, uint32_t rnd4) -> double {
    (void)rnd2;
    (void)rnd3;
    (void)rnd4;
    const double u = u01(rnd);
    return static_cast<int>(u * 20.0) + 1;
};

const auto D20KernelDyn = [](mc::RNGView& rng) -> double {
    const double u = rng.next_u01();
    return static_cast<int>(u * 20.0) + 1;
};

inline double rolls_per_second(uint64_t n_rolls,
                                std::chrono::steady_clock::time_point start,
                                std::chrono::steady_clock::time_point end)
{
    const std::chrono::duration<double> elapsed = end - start;
    return static_cast<double>(n_rolls) / elapsed.count();
}

template <typename Kernel>
void measure_mc_e(const Kernel& kernel, const uint64_t N) {
    const Problem problem(kernel, N);

    using clock = std::chrono::steady_clock;

    auto t0 = clock::now();
    double mean = mc::run(problem);
    auto t1 = clock::now();

    double rps = rolls_per_second(N, t0, t1);

    std::cout << "Rolls/sec = " << rps << "\n";
    std::cout << "Mean      = " << mean << "\n\n";
}

int main() {
    constexpr uint64_t N = 100'000'000;

    std::cout << "D20 1 Random" << std::endl;
    measure_mc_e(D20Kernel1, N);
    std::cout << "D20 2 Random" << std::endl;
    measure_mc_e(D20Kernel2, N);
    std::cout << "D20 3 Random" << std::endl;
    measure_mc_e(D20Kernel3, N);
    std::cout << "D20 4 Random" << std::endl;
    measure_mc_e(D20Kernel4, N);
    std::cout << "D20 Dynamic Random" << std::endl;
    measure_mc_e(D20KernelDyn, N);

    return 0;
}
