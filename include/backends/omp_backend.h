#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_kernel.h>

#include <rng/config.h>
#include <rng/splitmix64.h>

namespace mc {
    template <typename Problem>
    double run(const Problem& problem, const OMPBackend&) {
        double sum = 0.0;

        #pragma omp parallel reduction(+:sum)
        {
            const uint64_t tid = static_cast<uint64_t>(omp_get_thread_num());

            const uint64_t stream = splitmix64(BASE_SEED ^ tid);

            #pragma omp for simd schedule(static)
            for (uint64_t i = 0; i < problem.n_paths; ++i) {
                RNGState rng{i, stream};
                sum += problem.kernel(rng);
            }
        }

        return sum / static_cast<double>(problem.n_paths);
    }

}
