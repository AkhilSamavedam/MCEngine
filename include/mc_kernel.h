#pragma once

#include <cstdint>
#include "rng/philox.h"

namespace mc {


    /*
     * RNGView
     *
     * Deterministic, dimension-aware RNG cursor.
     * Portable across CPU and CUDA without CUDA-only syntax.
     */
    struct RNGView {
        // ---- RNG coordinates ----
        uint64_t counter;   // path id
        uint64_t stream;    // RNG stream / seed
        uint32_t dim;       // dimension index

        // ---- cached Philox block ----
        Philox4x32 block;
        uint32_t cached_block;

        // Sentinel value (no <limits>)
        static constexpr uint32_t INVALID_BLOCK = 0xFFFFFFFFu;

        // ---- constructor ----
        MC_HOST_DEVICE MC_FORCEINLINE
        RNGView(uint64_t path,
                uint64_t stream_,
                uint32_t start_dim = 0)
            : counter(path)
            , stream(stream_)
            , dim(start_dim)
            , cached_block(INVALID_BLOCK)
        {}

        // ---- next raw 32-bit random ----
        MC_HOST_DEVICE MC_FORCEINLINE
        uint32_t next_u32() {
            const uint32_t block_id = dim >> 2;   // dim / 4
            const uint32_t lane     = dim & 3;    // dim % 4

            if (block_id != cached_block) {
                block = philox10(counter + block_id, stream);
                cached_block = block_id;
            }

            ++dim;
            return block.c[lane];
        }

        // ---- uniform in [0,1) ----
        MC_HOST_DEVICE MC_FORCEINLINE
        double next_u01() {
            return u01(next_u32());
        }

        // ---- bounded integer helpers ----
        MC_HOST_DEVICE MC_FORCEINLINE
        uint32_t next_u32_bounded(uint32_t n) {
            return next_u32() % n;  // modulo bias OK for MC
        }

        MC_HOST_DEVICE MC_FORCEINLINE
        int next_int(int lo, int hi) {
            const uint32_t span = static_cast<uint32_t>(hi - lo + 1);
            return lo + static_cast<int>(next_u32_bounded(span));
        }
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

#define MC_U32(name) uint32_t name
#define MC_RNG(name) mc::RNGView& name

#if !defined(__CUDACC__) && defined(_OPENMP)
    #define MC_KERNEL_OMP_PRAGMA _Pragma("omp declare simd")
#else
    #define MC_KERNEL_OMP_PRAGMA
#endif

#define MC_KERNEL(name, ...) \
    struct name { \
        MC_KERNEL_OMP_PRAGMA \
        MC_KERNEL_OP double operator()(__VA_ARGS__) const

#define MC_KERNEL_END \
    };
