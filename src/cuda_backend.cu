#include <backends/cuda_backend.h>
#include <cuda_runtime.h>
#include <rng/config.h>

#include <utility>
#include <vector>

namespace mc {

    __global__ void reduce_sum_kernel(const double* input, double* output, size_t n) {
        extern __shared__ double sdata[];
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

        sdata[tid] = sum;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
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

        const int block = 256;
        const int grid = 1024;
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
