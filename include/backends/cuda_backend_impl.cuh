#pragma once

#ifdef __CUDACC__   // <-- critical
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <mc_kernel.h>
#include <rng/splitmix64.h>
#include <rng/philox.h>

namespace mc {

    __device__ __forceinline__
    uint64_t global_thread_id() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ __forceinline__
    uint64_t global_stride() {
        return gridDim.x * blockDim.x;
    }

    template <typename Kernel>
    __global__ void mc_kernel_arity1(Kernel kernel, uint64_t n_paths, uint64_t base_seed, double* out) {
        const uint64_t tid = global_thread_id();
        const uint64_t stride = global_stride();

        double local_sum = 0.0;
        const uint64_t stream = splitmix64(base_seed ^ tid);

        const uint64_t n_main = (n_paths / 4) * 4;

        for (uint64_t i = tid * 4; i < n_main; i += stride * 4) {
            const auto r = philox10(i / 4, stream);
            local_sum += kernel(r.c[0]);
            local_sum += kernel(r.c[1]);
            local_sum += kernel(r.c[2]);
            local_sum += kernel(r.c[3]);
        }

        for (uint64_t i = n_main + tid; i < n_paths; i += stride) {
            const auto r = philox10(i, stream);
            local_sum += kernel(r.c[0]);
        }

        out[tid] = local_sum;
    }

    template <typename Kernel>
    __global__ void mc_kernel_arity2(Kernel kernel, uint64_t n_paths, uint64_t base_seed, double* out){
        const uint64_t tid = global_thread_id();
        const uint64_t stride = global_stride();

        double local_sum = 0.0;
        const uint64_t stream = splitmix64(base_seed ^ tid);

        const uint64_t n_main = (n_paths / 2) * 2;

        for (uint64_t i = tid * 2; i < n_main; i += stride * 2) {
            const auto r = philox10(i / 2, stream);
            local_sum += kernel(r.c[0], r.c[1]);
            local_sum += kernel(r.c[2], r.c[3]);
        }

        for (uint64_t i = n_main + tid; i < n_paths; i += stride) {
            const auto r = philox10(i, stream);
            local_sum += kernel(r.c[0], r.c[1]);
        }

        out[tid] = local_sum;
    }

    template <typename Kernel>
    __global__ void mc_kernel_arity3(Kernel kernel, uint64_t n_paths, uint64_t base_seed, double* out){
        const uint64_t tid = global_thread_id();
        const uint64_t stride = global_stride();

        double local_sum = 0.0;
        const uint64_t stream = splitmix64(base_seed ^ tid);

        for (uint64_t i = tid; i < n_paths; i += stride) {
            const auto r = philox10(i, stream);
            local_sum += kernel(r.c[0], r.c[1], r.c[2]);
        }

        out[tid] = local_sum;
    }

    template <typename Kernel>
    __global__ void mc_kernel_arity4(Kernel kernel, uint64_t n_paths, uint64_t base_seed, double* out){
        const uint64_t tid = global_thread_id();
        const uint64_t stride = global_stride();

        double local_sum = 0.0;
        const uint64_t stream = splitmix64(base_seed ^ tid);

        for (uint64_t i = tid; i < n_paths; i += stride) {
            const auto r = philox10(i, stream);
            local_sum += kernel(r.c[0], r.c[1], r.c[2], r.c[3]);
        }

        out[tid] = local_sum;
    }

    template <typename Kernel>
    __global__ void mc_kernel_dynamic(Kernel kernel, uint64_t n_paths, uint64_t base_seed, double* out)
    {
        const uint64_t tid = global_thread_id();
        const uint64_t stride = global_stride();

        double local_sum = 0.0;
        const uint64_t stream = splitmix64(base_seed ^ tid);

        #if defined(MCENGINE_USE_CURAND) && MCENGINE_USE_CURAND
        RNGView rng(tid, stream, 0);
        for (uint64_t i = tid; i < n_paths; i += stride) {
            rng.counter = i;
            local_sum += kernel(rng);
        }
        #else
        for (uint64_t i = tid; i < n_paths; i += stride) {
            RNGView rng(i, stream, 0);
            local_sum += kernel(rng);
        }
        #endif

        out[tid] = local_sum;
    }

    template <typename Kernel>
    inline void launch_kernel_arity1_erased(
        const void* kernel_ptr,
        uint64_t n_paths,
        uint64_t base_seed,
        double* d_out,
        int grid,
        int block
    ) {
        const auto* kernel = static_cast<const Kernel*>(kernel_ptr);
        mc_kernel_arity1<<<grid, block>>>(*kernel, n_paths, base_seed, d_out);
    }

    template <typename Kernel>
    inline void launch_kernel_arity2_erased(
        const void* kernel_ptr,
        uint64_t n_paths,
        uint64_t base_seed,
        double* d_out,
        int grid,
        int block
    ) {
        const auto* kernel = static_cast<const Kernel*>(kernel_ptr);
        mc_kernel_arity2<<<grid, block>>>(*kernel, n_paths, base_seed, d_out);
    }

    template <typename Kernel>
    inline void launch_kernel_arity3_erased(
        const void* kernel_ptr,
        uint64_t n_paths,
        uint64_t base_seed,
        double* d_out,
        int grid,
        int block
    ) {
        const auto* kernel = static_cast<const Kernel*>(kernel_ptr);
        mc_kernel_arity3<<<grid, block>>>(*kernel, n_paths, base_seed, d_out);
    }

    template <typename Kernel>
    inline void launch_kernel_arity4_erased(
        const void* kernel_ptr,
        uint64_t n_paths,
        uint64_t base_seed,
        double* d_out,
        int grid,
        int block
    ) {
        const auto* kernel = static_cast<const Kernel*>(kernel_ptr);
        mc_kernel_arity4<<<grid, block>>>(*kernel, n_paths, base_seed, d_out);
    }

    template <typename Kernel>
    inline void launch_kernel_dynamic_erased(
        const void* kernel_ptr,
        uint64_t n_paths,
        uint64_t base_seed,
        double* d_out,
        int grid,
        int block
    ) {
        const auto* kernel = static_cast<const Kernel*>(kernel_ptr);
        mc_kernel_dynamic<<<grid, block>>>(*kernel, n_paths, base_seed, d_out);
    }

    template <int Arity, typename Kernel>
    inline CudaLaunchTable make_cuda_launch_table_for() {
        CudaLaunchTable table{};
        if constexpr (Arity == 1) {
            table.launch_arity1 = &launch_kernel_arity1_erased<Kernel>;
        } 
        else if constexpr (Arity == 2) {
            table.launch_arity2 = &launch_kernel_arity2_erased<Kernel>;
        } 
        else if constexpr (Arity == 3) {
            table.launch_arity3 = &launch_kernel_arity3_erased<Kernel>;
        } 
        else if constexpr (Arity == 4) {
            table.launch_arity4 = &launch_kernel_arity4_erased<Kernel>;
        } 
        else {
            table.launch_dynamic = &launch_kernel_dynamic_erased<Kernel>;
        }
        return table;
    }

} // namespace mc

#endif // __CUDACC__
