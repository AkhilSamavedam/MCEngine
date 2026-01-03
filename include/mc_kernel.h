#pragma once

#include <cstdint>

namespace mc {
    struct RNGState {
        uint64_t index;
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

        constexpr explicit MCProblem(const uint64_t n_paths) : kernel(), n_paths(n_paths) {}

        constexpr MCProblem(Kernel k, const uint64_t n_paths) : kernel(k), n_paths(n_paths) {}

        constexpr MCProblem(const MCProblem&) = delete;

    };

    template <typename Kernel>
    MCProblem(Kernel, uint64_t) -> MCProblem<Kernel>;

    struct OMPBackend {};
    struct CUDABackend {};
}