#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_kernel.h>

namespace mc {
    template <typename Problem>
    double run(const Problem& problem, const OMPBackend&) {
        double sum = 0.0;

        #pragma omp parallel reduction(+:sum)
        {
            const uint64_t tid = omp_get_thread_num();

            #pragma omp for simd schedule(static)
            for (uint64_t i = 0; i < problem.n_paths; ++i) {
                RNGState rng{i, tid};
                sum += problem.kernel(rng);
            }
        }
        return sum / problem.n_paths;
    }
}
