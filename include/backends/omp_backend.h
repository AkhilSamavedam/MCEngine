#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_kernel.h>

#include <rng/config.h>
#include <rng/splitmix64.h>
#include <rng/philox.h>

namespace mc {

    template <typename Problem>
    double run(const Problem& problem, const OMPBackend&) {
        constexpr uint64_t BATCH = 4;  // Philox4x32
        double sum = 0.0;

        #pragma omp parallel reduction(+:sum)
        {
            const uint64_t tid =
                static_cast<uint64_t>(omp_get_thread_num());

            // Per-thread RNG stream (key)
            const uint64_t seed = splitmix64(BASE_SEED ^ tid);

            const uint64_t n_main = problem.n_paths / BATCH * BATCH;

            #pragma omp for schedule(static)
            for (uint64_t i = 0; i <= n_main; i += BATCH) {
                // One Philox call → 4 randoms
                const auto r = philox10(i, seed);

                sum += problem.kernel(r.c[0]);
                sum += problem.kernel(r.c[1]);
                sum += problem.kernel(r.c[2]);
                sum += problem.kernel(r.c[3]);
            }

            // Tail handling
            #pragma omp for schedule(static)
            for (uint64_t i = n_main; i < problem.n_paths; ++i) {
                const auto r = philox10(i, seed);
                sum += problem.kernel(r.c[0]);
            }
        }

        return sum / static_cast<double>(problem.n_paths);
    }

} // namespace mc

