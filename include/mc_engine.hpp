#pragma once

#include "mc_kernel.h"
#include "mc_path.h"

#include "backends/omp_backend.h"
#include "backends/omp_path_backend.h"
#include "backends/cuda_backend.h"
#include "backends/cuda_path_backend.h"

namespace mc {

    template <typename State, typename StepKernel, typename PayoffKernel>
    double run(const MCPathProblem<State, StepKernel, PayoffKernel>& problem, OMPBackend) {
        return run_paths(problem, OMPBackend{});
    }

    #if defined(MCENGINE_ENABLE_CUDA) && MCENGINE_ENABLE_CUDA && defined(__CUDACC__)
    template <typename Problem>
    double run(const Problem& problem, CUDABackend) {
        return run_cuda(problem);
    }

    template <typename State, typename StepKernel, typename PayoffKernel>
    double run(const MCPathProblem<State, StepKernel, PayoffKernel>& problem, CUDABackend) {
        return run_paths(problem, CUDABackend{});
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

    template <typename State, typename StepKernel, typename PayoffKernel>
    double run(const MCPathProblem<State, StepKernel, PayoffKernel>& problem) {
        return run_paths(problem);
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
