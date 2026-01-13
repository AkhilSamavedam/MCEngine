#pragma once

#include "backends/omp_backend.h"
#include "backends/omp_path_backend.h"
#include "backends/cuda_backend.h"
#include "backends/cuda_path_backend.h"

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

    template <typename Problem>
    double run_paths(const Problem& problem) {
        #if defined(MCENGINE_ENABLE_CUDA) && MCENGINE_ENABLE_CUDA && defined(__CUDACC__)
        return run_paths(problem, CUDABackend{});
        #else
        return run_paths(problem, OMPBackend{});
        #endif
    }

} // namespace mc
