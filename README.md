# MCEngine

High-performance Monte Carlo engine with deterministic RNG, OpenMP CPU backend, and optional CUDA backend. Kernels can be written either as fixed-arity functions (1-4 randoms per path) or as dynamic RNG streams via `RNGView` for complex path logic.

## Overview

MCEngine is a C++20 library for running Monte Carlo simulations across many independent paths. It provides:

- An annotation-based kernel definition style using `MC` for CUDA compatibility.
- A path-based API (`PathProblem`) for multi-step simulations.
- Deterministic, dimension-aware RNG (`RNGView`) based on Philox4x32-10.
- Backend selection between OpenMP (CPU) and CUDA (GPU).

## Core concepts

### 1) Single-step problems: `Problem`

A single-step problem is a kernel that maps random inputs to a `double` payoff. MCEngine infers the RNG mode from the kernel signature:

- Fixed arity (fast path): 1-4 `uint32_t` arguments.
- Dynamic RNG: a single `mc::RNGView&` argument for arbitrary random draws.

Example (fixed-arity, 1 random per path):

```cpp
#include <mc_engine.hpp>

using namespace mc;

auto k = [](uint32_t rnd) -> double {
    const double u = u01(rnd);
    return static_cast<int>(u * 20.0) + 1;
};

int main() {
    const Problem problem(k, 100'000'000);
    const double mean = mc::run(problem);
    (void)mean;
}
```

Example (dynamic RNG):

```cpp
#include <mc_engine.hpp>

using namespace mc;

auto k = [](mc::RNGView& rng) -> double {
    int sum = 0;
    while (true) {
        const int roll = rng.next_int(1, 6);
        sum += roll;
        if (roll == 6) {
            break;
        }
    }
    return static_cast<double>(sum);
};

int main() {
    const Problem problem(k, 10'000'000);
    const double mean = mc::run(problem);
    (void)mean;
}
```

Note: For CUDA builds, lambdas must be device-callable. Use `MC` on the lambda (or `__host__ __device__`) and compile with NVCC extended lambdas. Function pointers are not supported as CUDA kernels; use lambdas or functors.

### 2) Path-based problems: `PathProblem`

A path-based problem evolves a state over multiple steps and computes a payoff at the end:

```cpp
#include <mc_engine.hpp>

using namespace mc;

struct BrownianState {
    double x;
};

auto step = [](BrownianState& st, RNGView& rng, double dt) {
    const double z = rng.next_normal();
    st.x += z * ::sqrt(dt);
};

auto payoff = [](const BrownianState& st) -> double {
    return st.x;
};

int main() {
    const BrownianState init{0.0};
    const PathProblem problem(
        init,
        step,
        payoff,
        50'000'000,
        64,
        0.1
    );

    const double mean = mc::run(problem);
    (void)mean;
}
```

## RNG model and determinism

- The engine uses Philox4x32-10 for CPU RNG.
- Each path gets a deterministic RNG stream based on `(path_id, stream_seed)`.
- `RNGView` tracks a dimension counter and caches Philox blocks for efficient sequential draws.
- On CUDA, `RNGView` is backed by cuRAND Philox state and uses `curand()` and `curand_normal2_double()` for draws.
- OpenMP backend seeds each thread with `splitmix64(BASE_SEED ^ tid)`.
- `mc::BASE_SEED` is defined in `include/rng/config.h` and can be overridden by the user.

Important details:

- Fixed-arity kernels (1-4 `uint32_t`) are fastest and vectorize well on CPU.
- Dynamic RNG (`RNGView`) supports `next_u32`, `next_u01`, `next_normal`, and `next_int`.
- `next_u32_bounded` uses modulo reduction; bias is accepted for Monte Carlo use.

## Backends

### OpenMP (CPU)

Default when CUDA is not enabled or not compiled with NVCC. Implements fast paths for arities 1-4 and a dynamic RNG path for `RNGView`.

### CUDA (GPU)

- Available when `MCENGINE_ENABLE_CUDA=1` and the translation unit is compiled with NVCC.
- Uses cuRAND Philox (`curandStatePhilox4_32_10_t`) for device-side RNG and `curand_normal2_double` for normals.
- Dispatches to CUDA kernels generated in `include/backends/cuda_backend_impl.cuh`.

The `mc::run()` and `mc::run_paths()` helpers select the backend automatically:

- CUDA if `MCENGINE_ENABLE_CUDA` is on **and** the calling translation unit is compiled with NVCC.
- Otherwise OpenMP.

## Type erasure architecture

MCEngine exposes stable, non-templated entry points for both single-step and path-based execution, while keeping templated kernels for performance. The flow is:

1. At compile time, the kernel or problem type is inspected to determine RNG arity and the correct execution path.
2. A launch table is built for that concrete type. For CUDA this is a `CudaLaunchTable` with device kernel launchers; for OpenMP this is `OmpLaunchTable` and `OmpPathLaunchTable`.
3. The runtime entry points (`run_cuda_erased`, `run_omp_erased`, `run_omp_paths_erased`) receive a `void*` to the concrete kernel/problem, plus the arity and launch table, and perform a single dispatch to the correct function.

This design keeps dispatch overhead to a single switch per run, avoids per-path runtime checks, and provides a shared-library friendly ABI boundary while still allowing fully inlined kernel code in the templated launchers.

## Build

Requirements:

- CMake >= 4.1.
- C++20 compiler with OpenMP.
- CUDA toolkit (only required when `MCENGINE_ENABLE_CUDA=ON`).

Typical build:

```bash
cmake -S . -B build -DMCENGINE_ENABLE_CUDA=ON
cmake --build build -j
```

CPU-only build:

```bash
cmake -S . -B build -DMCENGINE_ENABLE_CUDA=OFF
cmake --build build -j
```

## Tests and examples

The `tests/` folder contains example programs (they are also registered as CTest tests). They are heavy by default (tens to hundreds of millions of paths).

```bash
cmake -S . -B build -DMCENGINE_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build
```

Example programs:

- `tests/test_d20_omp.cpp`: fixed-arity kernel benchmarks.
- `tests/test_dice_until_6_omp.cpp`: dynamic RNG kernel with a loop.
- `tests/test_multistep_omp.cpp`: path-based Brownian motion.
- `tests/test_*_cuda.cpp`: CUDA equivalents.

## API reference (quick)

- `mc::Problem<Kernel>`: single-step Monte Carlo problem.
- `mc::PathProblem<State, StepKernel, PayoffKernel>`: multi-step path simulation.
- `mc::run(problem)`: run with auto-selected backend.
- `mc::run(problem, mc::OMPBackend{})`: force OpenMP.
- `mc::run(problem, mc::CUDABackend{})`: force CUDA (NVCC only).
- `mc::run_paths(problem)`: run path-based problem with auto backend (explicit).
- `mc::run(problem)`: runs `Problem` or `PathProblem` with auto backend.

Kernel macros (optional, defined in `mc_kernel.h` and `mc_path.h`, and included by `mc_engine.hpp`):

- `MC_KERNEL(Name, (args...), body...)`
- `MC_STATE(Name, body...)`
- `MC_STEP_KERNEL(Name, (args...), body...)`
- `MC_PAYOFF_KERNEL(Name, (args...), body...)`

Argument helpers:

- `MC_U32(name)` for fixed-arity kernels.
- `MC_RNG(name)` for `RNGView&` kernels.

## Repo layout

- `include/`: public headers and kernel macros.
- `src/`: backend implementations (`omp_backend.cpp`, `cuda_backend.cu`).
- `tests/`: example programs and CTest registrations.

## Notes

- A kernel must take either 1-4 `uint32_t` arguments or a single `mc::RNGView&`; other signatures are rejected at compile time.
- CUDA backends require the translation unit that calls `mc::run` or `mc::run_paths` to be compiled with NVCC.
- When `MCENGINE_ENABLE_CUDA=OFF`, you can build without the CUDA toolkit installed.
