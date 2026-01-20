#include <backends/omp_backend.h>
#include <backends/omp_path_backend.h>

namespace mc {
    double run_omp_erased(
        const void* kernel,
        uint64_t n_paths,
        uint64_t base_seed,
        int arity,
        const OmpLaunchTable* table
    ) {
        switch (arity) {
            case 1:
                return table->launch_arity1(kernel, n_paths, base_seed);
            case 2:
                return table->launch_arity2(kernel, n_paths, base_seed);
            case 3:
                return table->launch_arity3(kernel, n_paths, base_seed);
            case 4:
                return table->launch_arity4(kernel, n_paths, base_seed);
            default:
                return table->launch_dynamic(kernel, n_paths, base_seed);
        }
    }

    double run_omp_paths_erased(
        const void* problem,
        uint64_t base_seed,
        const OmpPathLaunchTable* table
    ) {
        return table->launch(problem, base_seed);
    }
} // namespace mc
