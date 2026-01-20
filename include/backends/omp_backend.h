#pragma once

#include <omp.h>
#include <cstdint>
#include <mc_kernel.h>

#include <rng/config.h>
#include <rng/splitmix64.h>
#include <rng/philox.h>

#include <rng/rng_arity.h>

namespace mc {
    struct OmpLaunchTable {
        using LaunchFn = double (*)(const void* kernel, uint64_t n_paths, uint64_t base_seed);

        LaunchFn launch_arity1 = nullptr;
        LaunchFn launch_arity2 = nullptr;
        LaunchFn launch_arity3 = nullptr;
        LaunchFn launch_arity4 = nullptr;
        LaunchFn launch_dynamic = nullptr;
    };

    double run_omp_erased(
        const void* kernel,
        uint64_t n_paths,
        uint64_t base_seed,
        int arity,
        const OmpLaunchTable* table
    );

    namespace detail {
        template <typename Kernel>
        double run_omp_kernel(const Kernel& kernel, uint64_t n_paths, uint64_t base_seed) {
            constexpr int ARITY = rng_arity<Kernel>();

            double sum = 0.0;

            #pragma omp parallel reduction(+:sum)
            {
                const uint64_t tid = static_cast<uint64_t>(omp_get_thread_num());
                const uint64_t stream = splitmix64(base_seed ^ tid);

                // -----------------------------
                // Fixed-arity fast paths
                // -----------------------------
                if constexpr (ARITY == 1) {
                    const uint64_t n_main = (n_paths / 4) * 4;

                    #pragma omp for schedule(static)
                    for (uint64_t i = 0; i < n_main; i += 4) {
                        const auto r = philox10(i, stream);
                        sum += kernel(r.c[0]);
                        sum += kernel(r.c[1]);
                        sum += kernel(r.c[2]);
                        sum += kernel(r.c[3]);
                    }

                    #pragma omp for schedule(static)
                    for (uint64_t i = n_main; i < n_paths; ++i) {
                        sum += kernel(philox10(i, stream).c[0]);
                    }
                }

                else if constexpr (ARITY == 2) {
                    const uint64_t n_main = (n_paths / 2) * 2;

                    #pragma omp for schedule(static)
                    for (uint64_t i = 0; i < n_main; i += 2) {
                        const auto r = philox10(i / 2, stream);
                        sum += kernel(r.c[0], r.c[1]);
                        sum += kernel(r.c[2], r.c[3]);
                    }

                    #pragma omp for schedule(static)
                    for (uint64_t i = n_main; i < n_paths; ++i) {
                        const auto r = philox10(i, stream);
                        sum += kernel(r.c[0], r.c[1]);
                    }
                }

                else if constexpr (ARITY == 3) {
                    #pragma omp for schedule(static)
                    for (uint64_t i = 0; i < n_paths; ++i) {
                        const auto r = philox10(i, stream);
                        sum += kernel(r.c[0], r.c[1], r.c[2]);
                    }
                }

                else if constexpr (ARITY == 4) {
                    #pragma omp for schedule(static)
                    for (uint64_t i = 0; i < n_paths; ++i) {
                        const auto r = philox10(i, stream);
                        sum += kernel(r.c[0], r.c[1], r.c[2], r.c[3]);
                    }
                }

                // -----------------------------
                // Dynamic RNG path
                // -----------------------------
                else { // ARITY == -1
                    #pragma omp for schedule(static)
                    for (uint64_t i = 0; i < n_paths; ++i) {
                        RNGView rng(i, stream, 0);
                        sum += kernel(rng);
                    }
                }
            }

            return sum / static_cast<double>(n_paths);
        }

        template <typename Kernel>
        double launch_kernel_omp_erased(
            const void* kernel_ptr,
            uint64_t n_paths,
            uint64_t base_seed
        ) {
            const auto* kernel = static_cast<const Kernel*>(kernel_ptr);
            return run_omp_kernel(*kernel, n_paths, base_seed);
        }
    } // namespace detail

    template <int Arity, typename Kernel>
    inline OmpLaunchTable make_omp_launch_table_for() {
        static_assert(
            Arity == rng_arity<Kernel>(),
            "Arity does not match kernel RNG arity"
        );

        OmpLaunchTable table{};
        if constexpr (Arity == 1) {
            table.launch_arity1 = &detail::launch_kernel_omp_erased<Kernel>;
        }
        else if constexpr (Arity == 2) {
            table.launch_arity2 = &detail::launch_kernel_omp_erased<Kernel>;
        }
        else if constexpr (Arity == 3) {
            table.launch_arity3 = &detail::launch_kernel_omp_erased<Kernel>;
        }
        else if constexpr (Arity == 4) {
            table.launch_arity4 = &detail::launch_kernel_omp_erased<Kernel>;
        }
        else {
            table.launch_dynamic = &detail::launch_kernel_omp_erased<Kernel>;
        }
        return table;
    }

    template <typename Problem>
    double run(const Problem& problem, const OMPBackend&) {
        using Kernel = decltype(problem.kernel);
        constexpr int arity = rng_arity<Kernel>();

        const OmpLaunchTable table = make_omp_launch_table_for<arity, Kernel>();
        return run_omp_erased(
            &problem.kernel,
            problem.n_paths,
            BASE_SEED,
            arity,
            &table
        );
    }

} // namespace mc

