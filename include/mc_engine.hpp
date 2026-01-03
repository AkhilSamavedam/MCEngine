#pragma once

#include <cstdint>

namespace mc {
    struct RNGState {
        uint64_t counter;
        uint64_t seed;
    };

    // Kernel form
    /*
    struct Kernel {
        double operator()(RNGState&) const;
    };
    */
    template <typename Kernel>
    struct MCProblem {
        Kernel kernel;
        uint64_t n_paths;
    };

    struct OMPBackend {};
    struct CUDABackend {};
}