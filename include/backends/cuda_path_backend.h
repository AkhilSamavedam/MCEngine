#pragma once

#ifdef __CUDACC__
#include "backends/cuda_path_backend_impl.cuh"
#endif

#include <mc_kernel.h>
#include <mc_path.h>

namespace mc {

    template <typename Problem>
    double run_paths(const Problem& problem, const CUDABackend&) {
        #ifdef __CUDACC__
        return run_paths_cuda(problem);
        #else
        static_assert(
            sizeof(Problem) == 0,
            "CUDA path backend requires NVCC compilation for the calling translation unit."
        );
        return 0.0;
        #endif
    }

} // namespace mc
