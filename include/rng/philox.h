#pragma once
#include <cstdint>
#include <mc_portability.h>

// Philox4x32-10 (10 rounds as per Random123 / cuRAND)

namespace mc {

    struct Philox4x32 {
        uint32_t c[4];
        uint32_t k[2];
    };

    __attribute__((always_inline))
    static constexpr uint32_t mulhi(const uint32_t a, const uint32_t b) {
        return (static_cast<uint64_t>(a) * static_cast<uint64_t>(b)) >> 32;
    }

    __attribute__((always_inline))
    static Philox4x32 philox_round(const Philox4x32 x) {
        constexpr uint32_t M0 = 0xD2511F53;
        constexpr uint32_t M1 = 0xCD9E8D57;
        constexpr uint32_t W0 = 0x9E3779B9;
        constexpr uint32_t W1 = 0xBB67AE85;

        const uint32_t hi0 = mulhi(M0, x.c[0]);
        const uint32_t lo0 = M0 * x.c[0];

        const uint32_t hi1 = mulhi(M1, x.c[2]);
        const uint32_t lo1 = M1 * x.c[2];

        Philox4x32 out;
        out.c[0] = hi1 ^ x.c[1] ^ x.k[0];
        out.c[1] = lo1;
        out.c[2] = hi0 ^ x.c[3] ^ x.k[1];
        out.c[3] = lo0;

        out.k[0] = x.k[0] + W0;
        out.k[1] = x.k[1] + W1;

        return out;
    }

    __attribute__((always_inline))
    MC_HOST_DEVICE static Philox4x32 philox10(const uint64_t counter, const uint64_t seed) {
        Philox4x32 x{};
        x.c[0] = static_cast<uint32_t>(counter);
        x.c[1] = static_cast<uint32_t>(counter >> 32);
        x.c[2] = 0;
        x.c[3] = 0;

        x.k[0] = static_cast<uint32_t>(seed);
        x.k[1] = static_cast<uint32_t>(seed >> 32);

        #pragma unroll
        for (int i = 0; i < 10; ++i)
            x = philox_round(x);

        return x;
    }

    __attribute__((always_inline))
    MC_HOST_DEVICE static constexpr double u01(const uint32_t x) {
        return (x >> 8) * (1.0 / (1u << 24));
    }
} // namespace mc