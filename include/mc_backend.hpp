#pragma once


#include "backends/omp_backend.hpp"
#include "backends/cuda_backend.cuh"

namespace mc {
    #if defined(MCENGINE_ENABLE_CUDA)
        using  DefaultBackend = CUDABackend;
    #else
        using DefaultBackend = OMPBackend;
    #endif

    template <typename Problem>
    double run(const Problem& problem) {
        return run(DefaultBackend{}, problem);
    }
}