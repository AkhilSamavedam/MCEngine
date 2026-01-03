#pragma once

#include "backends/omp_backend.h"
#include "backends/cuda_backend.cuh"

namespace mc {

    #if defined(MCENGINE_ENABLE_CUDA)
        using  DefaultBackend = CUDABackend;
    #else
        using DefaultBackend = OMPBackend;
    #endif

    template <typename Problem, typename Backend = DefaultBackend>
    double run(const Problem& problem) {
        return run(problem, Backend());
    }
}