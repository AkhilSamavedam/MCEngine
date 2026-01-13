#pragma once

#include <cstdint>
#include "mc_kernel.h"

namespace mc {

    template <typename State, typename StepKernel, typename PayoffKernel>
    struct MCPathProblem {
        State initial_state;
        StepKernel step_kernel;
        PayoffKernel payoff_kernel;
        uint64_t n_paths;
        uint32_t n_steps;
        double dt;

        MCPathProblem(State init,
                      StepKernel step,
                      PayoffKernel payoff,
                      uint64_t n_paths,
                      uint32_t n_steps,
                      double dt)
            : initial_state(init)
            , step_kernel(step)
            , payoff_kernel(payoff)
            , n_paths(n_paths)
            , n_steps(n_steps)
            , dt(dt)
        {}

        MCPathProblem(const MCPathProblem&) = delete;
    };

    template <typename State, typename StepKernel, typename PayoffKernel>
    MCPathProblem(State, StepKernel, PayoffKernel, uint64_t, uint32_t, double)
        -> MCPathProblem<State, StepKernel, PayoffKernel>;

} // namespace mc

#define MC_STATE(name, ...) \
    struct name { \
        __VA_ARGS__ \
    };

#define MC_STEP_KERNEL(name, args, ...) \
    struct name { \
        MC_KERNEL_OP void operator() args const { \
            __VA_ARGS__ \
        } \
    };

#define MC_PAYOFF_KERNEL(name, args, ...) \
    struct name { \
        MC_KERNEL_OP double operator() args const { \
            __VA_ARGS__ \
        } \
    };
