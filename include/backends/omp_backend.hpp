#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_engine.hpp>

namespace mc {
    uint64_t thread_seed() {
        return static_cast<uint64_t>(omp_get_thread_num());
    }

    template <typename Problem>
    double run(const OMPBackend&, const Problem& problem) {
        double sum = 0.0;
        #pragma omp parallel
        {
            double local_sum = 0.0;
            RNGState rng(0, thread_seed());
            #pragma omp for
            for (uint64_t i = 0; i < problem.n_paths; i++) {
                rng.counter = i;
                local_sum += problem.kernel(rng);
            }
            #pragma omp atomic
            sum += local_sum;
        }
    }
}