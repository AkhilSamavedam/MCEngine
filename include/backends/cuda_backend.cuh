#pragma once

#include <cstdint>
#include <stdexcept>


namespace mc {
    template <typename Problem>
    double run(const Problem& problem, const CUDABackend&) {
        throw std::logic_error("Running backend does not support CUDA");
    }
}
