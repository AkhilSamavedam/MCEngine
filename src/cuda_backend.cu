#include <backends/cuda_backend.h>
#include <cuda_runtime.h>
#include <rng/config.h>

#include <utility>
#include <vector>

namespace mc {

    struct LaunchConfig {
        int block;
        int grid;
    };

    static LaunchConfig select_launch_config(uint64_t n_paths) {
        int block = 256;
        int grid = 1024;

        int device = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
            const int sm_count = prop.multiProcessorCount;
            const int blocks_per_sm = 4;
            grid = sm_count * blocks_per_sm;

            const uint64_t max_grid = (n_paths + static_cast<uint64_t>(block) - 1) /
                                      static_cast<uint64_t>(block);
            if (max_grid > 0 && grid > static_cast<int>(max_grid)) {
                grid = static_cast<int>(max_grid);
            }
            if (grid < 1) {
                grid = 1;
            }
        }

        return {block, grid};
    }

    inline __device__ double warp_reduce_sum(double val) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    __global__ void reduce_sum_kernel(const double* input, double* output, size_t n) {
        const unsigned int tid = threadIdx.x;
        const unsigned int idx = blockIdx.x * blockDim.x * 2 + tid;

        double sum = 0.0;
        if (idx < n) {
            sum = input[idx];
            const unsigned int idx2 = idx + blockDim.x;
            if (idx2 < n) {
                sum += input[idx2];
            }
        }

        sum = warp_reduce_sum(sum);

        __shared__ double warp_sums[32];
        const unsigned int lane = tid & (warpSize - 1);
        const unsigned int warp_id = tid / warpSize;
        if (lane == 0) {
            warp_sums[warp_id] = sum;
        }
        __syncthreads();

        double block_sum = 0.0;
        if (warp_id == 0) {
            block_sum = (lane < (blockDim.x + warpSize - 1) / warpSize)
                ? warp_sums[lane]
                : 0.0;
            block_sum = warp_reduce_sum(block_sum);
            if (lane == 0) {
                output[blockIdx.x] = block_sum;
            }
        }
    }

    double run_cuda_erased(
        const void* problem_ptr,
        const void* kernel_ptr,
        uint64_t n_paths,
        int arity,
        const CudaLaunchTable* table
    )
    {
        (void)problem_ptr;

        const LaunchConfig cfg = select_launch_config(n_paths);
        const int block = cfg.block;
        const int grid = cfg.grid;
        const size_t n_threads = static_cast<size_t>(block) * static_cast<size_t>(grid);

        double* d_out = nullptr;
        double* d_tmp = nullptr;
        cudaMalloc(&d_out, n_threads * sizeof(double));
        cudaMalloc(&d_tmp, n_threads * sizeof(double));

        switch (arity) {
            case 1:
                table->launch_arity1(kernel_ptr, n_paths, BASE_SEED, d_out, grid, block);
                break;
            case 2:
                table->launch_arity2(kernel_ptr, n_paths, BASE_SEED, d_out, grid, block);
                break;
            case 3:
                table->launch_arity3(kernel_ptr, n_paths, BASE_SEED, d_out, grid, block);
                break;
            case 4:
                table->launch_arity4(kernel_ptr, n_paths, BASE_SEED, d_out, grid, block);
                break;
            default:
                table->launch_dynamic(kernel_ptr, n_paths, BASE_SEED, d_out, grid, block);
        }

        cudaDeviceSynchronize();

        size_t n = n_threads;
        double* d_in = d_out;
        double* d_reduce_out = d_tmp;
        while (n > 1) {
            const int reduce_grid = static_cast<int>((n + block * 2 - 1) / (block * 2));
            reduce_sum_kernel<<<reduce_grid, block, block * sizeof(double)>>>(d_in, d_reduce_out, n);
            n = static_cast<size_t>(reduce_grid);
            std::swap(d_in, d_reduce_out);
        }

        double sum = 0.0;
        cudaMemcpy(&sum, d_in, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        cudaFree(d_tmp);

        return sum / static_cast<double>(n_paths);
    }

}
