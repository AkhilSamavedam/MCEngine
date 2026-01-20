#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_kernel.h>
#include <mc_path.h>
#include <rng/splitmix64.h>

namespace mc {

    struct OmpPathLaunchTable {
        using LaunchFn = double (*)(const void* problem, uint64_t base_seed);
        LaunchFn launch = nullptr;
    };

    double run_omp_paths_erased(
        const void* problem,
        uint64_t base_seed,
        const OmpPathLaunchTable* table
    );

    namespace detail {
        template <typename Problem>
        double run_paths_problem(const Problem& problem, uint64_t base_seed) {
            double sum = 0.0;

            #pragma omp parallel reduction(+:sum)
            {
                const uint64_t tid = static_cast<uint64_t>(omp_get_thread_num());
                const uint64_t stream = splitmix64(base_seed ^ tid);

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

        template <typename Problem>
        double launch_paths_omp_erased(const void* problem_ptr, uint64_t base_seed) {
            const auto* problem = static_cast<const Problem*>(problem_ptr);
            return run_paths_problem(*problem, base_seed);
        }
    } // namespace detail

    template <typename Problem>
    inline OmpPathLaunchTable make_omp_path_launch_table_for() {
        OmpPathLaunchTable table{};
        table.launch = &detail::launch_paths_omp_erased<Problem>;
        return table;
    }

    template <typename Problem>
    double run_paths(const Problem& problem, const OMPBackend&) {
        const OmpPathLaunchTable table = make_omp_path_launch_table_for<Problem>();
        return run_omp_paths_erased(&problem, BASE_SEED, &table);
    }

} // namespace mc
