#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_kernel.h>

#include <rng/config.h>
#include <rng/splitmix64.h>
#include <rng/philox.h>

#include <rng/rng_arity.h>

namespace mc {

    template <typename Problem>
    double run(const Problem& problem, const OMPBackend&) {

        using Kernel = decltype(problem.kernel);
        constexpr int ARITY = rng_arity<Kernel>();

        double sum = 0.0;

        #pragma omp parallel reduction(+:sum)
        {
            const uint64_t tid = static_cast<uint64_t>(omp_get_thread_num());

            const uint64_t stream = splitmix64(BASE_SEED ^ tid);

            // -----------------------------
            // Fixed-arity fast paths
            // -----------------------------
            if constexpr (ARITY == 1) {
                const uint64_t n_main = (problem.n_paths / 4) * 4;

                #pragma omp for schedule(static)
                for (uint64_t i = 0; i < n_main; i += 4) {
                    const auto r = philox10(i, stream);
                    sum += problem.kernel(r.c[0]);
                    sum += problem.kernel(r.c[1]);
                    sum += problem.kernel(r.c[2]);
                    sum += problem.kernel(r.c[3]);
                }

                #pragma omp for schedule(static)
                for (uint64_t i = n_main; i < problem.n_paths; ++i) {
                    sum += problem.kernel(philox10(i, stream).c[0]);
                }
            }

            else if constexpr (ARITY == 2) {
                const uint64_t n_main = (problem.n_paths / 2) * 2;

                #pragma omp for schedule(static)
                for (uint64_t i = 0; i < n_main; i += 2) {
                    const auto r = philox10(i / 2, stream);
                    sum += problem.kernel(r.c[0], r.c[1]);
                    sum += problem.kernel(r.c[2], r.c[3]);
                }

                #pragma omp for schedule(static)
                for (uint64_t i = n_main; i < problem.n_paths; ++i) {
                    const auto r = philox10(i, stream);
                    sum += problem.kernel(r.c[0], r.c[1]);
                }
            }

            else if constexpr (ARITY == 3) {
                #pragma omp for schedule(static)
                for (uint64_t i = 0; i < problem.n_paths; ++i) {
                    const auto r = philox10(i, stream);
                    sum += problem.kernel(r.c[0], r.c[1], r.c[2]);
                }
            }

            else if constexpr (ARITY == 4) {
                #pragma omp for schedule(static)
                for (uint64_t i = 0; i < problem.n_paths; ++i) {
                    const auto r = philox10(i, stream);
                    sum += problem.kernel(r.c[0], r.c[1], r.c[2], r.c[3]);
                }
            }

            // -----------------------------
            // Dynamic RNG path
            // -----------------------------
            else { // ARITY == -1
                #pragma omp for schedule(static)
                for (uint64_t i = 0; i < problem.n_paths; ++i) {
                    RNGView rng(i, stream, 0);
                    sum += problem.kernel(rng);
                }
            }
        }

        return sum / static_cast<double>(problem.n_paths);
    }


} // namespace mc


