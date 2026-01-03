#pragma once

#include <cstdint>
#include <stdexcept>


namespace mc {
    template <typename Problem>
    double run(const CUDABackend&, const Problem& problem) {
        throw std::logic_error("Running backend does not support CUDA");
    }
}
