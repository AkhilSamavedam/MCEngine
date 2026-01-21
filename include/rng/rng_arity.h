#pragma once

#include <type_traits>
#include <tuple>
#include <cstdint>
#include <utility>

// Forward declare RNGView so kernels can reference it
namespace mc {
    struct RNGView;
}

/* ============================================================
 *  function_traits: extract operator() signature
 * ============================================================ */

template <typename T>
struct function_traits;

// const-qualified operator()
template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const> {
    using return_type = R;
    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t I>
    using arg = std::tuple_element_t<I, std::tuple<Args...>>;
};

// non-const operator() (allow it, but discourage it)
template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...)> {
    using return_type = R;
    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t I>
    using arg = std::tuple_element_t<I, std::tuple<Args...>>;
};

// free function pointer
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
    using return_type = R;
    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t I>
    using arg = std::tuple_element_t<I, std::tuple<Args...>>;
};

// free function type
template <typename R, typename... Args>
struct function_traits<R(Args...)> {
    using return_type = R;
    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t I>
    using arg = std::tuple_element_t<I, std::tuple<Args...>>;
};

/* ============================================================
 *  RNGView detection
 * ============================================================ */

template <typename Kernel, typename Enable = void>
struct kernel_traits {
    using type = function_traits<Kernel>;
};

template <typename Kernel>
struct kernel_traits<Kernel, std::enable_if_t<std::is_class_v<Kernel>>> {
    using type = function_traits<decltype(&Kernel::operator())>;
};

template <typename Kernel>
using kernel_traits_t = typename kernel_traits<Kernel>::type;

template <typename Kernel>
constexpr bool uses_rngview() {
    using traits = kernel_traits_t<Kernel>;

    if constexpr (traits::arity == 1) {
        using A0 = typename traits::template arg<0>;
        return std::is_same_v<
            std::remove_cvref_t<A0>,
            mc::RNGView
        >;
    }
    else {
        return false;
    }
}

/* ============================================================
 *  RNG arity inference
 *
 *  Return values:
 *    1..4  : fixed RNG arity
 *    -1    : dynamic (RNGView&)
 * ============================================================ */

template <typename Kernel>
constexpr int rng_arity() {
    #if defined(__CUDACC__)
    static_assert(
        !std::is_pointer_v<Kernel>,
        "CUDA kernels must be lambdas/functors, not function pointers."
    );
    #endif

    using traits = kernel_traits_t<Kernel>;

    static_assert(
        traits::arity > 0,
        "Kernel must take at least one argument"
    );

    // Dynamic RNG path
    if constexpr (uses_rngview<Kernel>()) {
        return -1;
    }
    else {
        // All arguments must be uint32_t
        constexpr bool all_u32 =
            ([]<std::size_t... I>(std::index_sequence<I...>) {
                return (
                    std::is_same_v<
                        std::remove_cvref_t<typename traits::template arg<I>>,
                        uint32_t
                    > && ...
                );
            })(std::make_index_sequence<traits::arity>{});

        static_assert(
            all_u32,
            "Kernel must take either:"
            "\n  - uint32_t arguments (fixed RNG arity)"
            "\n  - OR a single mc::RNGView& argument (dynamic RNG)"
        );

        static_assert(
            traits::arity <= 4,
            "Fixed RNG arity > 4 not supported; "
            "increase batching or use RNGView"
        );

        return static_cast<int>(traits::arity);
    }
}
