#pragma once

#include "backends/omp_backend.h"
#include "backends/cuda_backend.h"

namespace mc {

    #if defined(MCENGINE_ENABLE_CUDA) && MCENGINE_ENABLE_CUDA && defined(__CUDACC__)
    template <typename Problem>
    double run(const Problem& problem, CUDABackend) {
        return run_cuda(problem);
    }
    #endif

    template <typename Problem>
    double run(const Problem& problem) {
        #if defined(MCENGINE_ENABLE_CUDA) && MCENGINE_ENABLE_CUDA && defined(__CUDACC__)
        return run(problem, CUDABackend{});
        #else
        return run(problem, OMPBackend{});
        #endif
    }

} // namespace mc
