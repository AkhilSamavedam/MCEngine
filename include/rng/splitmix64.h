#pragma once

#include <cstdint>

namespace mc {

    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ x >> 30) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ x >> 27) * 0x94d049bb133111ebULL;
        return x ^ x >> 31;
    }

    // Convert to uniform double in [0, 1)
    static double u01_from_u64(const uint64_t x) {
        return (x >> 11) * (1.0 / (1ULL << 53));
    }

}
