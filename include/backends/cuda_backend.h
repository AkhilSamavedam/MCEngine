#pragma once

#include <cstdint>
#include "rng/rng_arity.h"

namespace mc {
    struct CudaLaunchTable {
        using LaunchFn = void (*)(
            const void* kernel,
            uint64_t n_paths,
            uint64_t base_seed,
            double* d_out,
            int grid,
            int block
        );

        LaunchFn launch_arity1;
        LaunchFn launch_arity2;
        LaunchFn launch_arity3;
        LaunchFn launch_arity4;
        LaunchFn launch_dynamic;
    };

    // Non-templated entry point (type-erased launch table)
    double run_cuda_erased(
        const void* problem,
        const void* kernel,
        uint64_t n_paths,
        int arity,
        const CudaLaunchTable* table
    );
} // namespace mc

#ifdef __CUDACC__
#include "backends/cuda_backend_impl.cuh"
#endif

namespace mc {

    template <typename Problem>
    double run_cuda(const Problem& problem) {
        #ifdef __CUDACC__
        using Kernel = decltype(problem.kernel);
        constexpr int arity = rng_arity<Kernel>();
        const CudaLaunchTable table = make_cuda_launch_table_for<arity, Kernel>();
        return run_cuda_erased(
            &problem,
            &problem.kernel,
            problem.n_paths,
            arity,
            &table
        );
        #else
        static_assert(
            sizeof(Problem) == 0,
            "CUDA backend requires NVCC compilation for the calling translation unit."
        );
        return 0.0;
        #endif
    }

} // namespace mc
