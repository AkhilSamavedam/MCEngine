#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_kernel.h>
#include <mc_path.h>
#include <rng/splitmix64.h>

namespace mc {

    template <typename Problem>
    double run_paths(const Problem& problem, const OMPBackend&) {
        double sum = 0.0;

        #pragma omp parallel reduction(+:sum)
        {
            const uint64_t tid = static_cast<uint64_t>(omp_get_thread_num());
            const uint64_t stream = splitmix64(BASE_SEED ^ tid);

            #pragma omp for schedule(static)
            for (uint64_t i = 0; i < problem.n_paths; ++i) {
                RNGView rng(i, stream, 0);
                auto state = problem.initial_state;
                for (uint32_t step = 0; step < problem.n_steps; ++step) {
                    problem.step_kernel(state, rng, problem.dt);
                }
                sum += problem.payoff_kernel(state);
            }
        }

        return sum / static_cast<double>(problem.n_paths);
    }

} // namespace mc
