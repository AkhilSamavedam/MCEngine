#include <backends/cuda_backend.h>
#include <cuda_runtime.h>
#include <rng/config.h>

#include <vector>

namespace mc {

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
        cudaMalloc(&d_out, n_threads * sizeof(double));

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

        std::vector<double> h_out(n_threads);
        cudaMemcpy(h_out.data(), d_out, n_threads * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (const double v : h_out) sum += v;

        cudaFree(d_out);

        return sum / static_cast<double>(n_paths);
    }

}
