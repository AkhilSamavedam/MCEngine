#pragma once

// Detect CUDA compilation
#if defined(__CUDACC__)
    #define MC_HOST_DEVICE __host__ __device__
    #define MC_FORCEINLINE __forceinline__
#else
    #define MC_HOST_DEVICE
    #define MC_FORCEINLINE inline
#endif
