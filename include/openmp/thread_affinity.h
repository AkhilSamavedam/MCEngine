#pragma once

#include <sched.h>
#include <unistd.h>

#if defined(__linux__)
inline int mc_detect_physical_cores() {
    // Conservative: assume SMT=2 if present
    const int logical = sysconf(_SC_NPROCESSORS_ONLN);
    return logical >= 2 ? logical / 2 : logical;
}

inline void mc_pin_thread(int core_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core_id, &set);
    sched_setaffinity(0, sizeof(set), &set);
}
#else
inline int mc_detect_physical_cores() { return 1; }
inline void mc_pin_thread(int) {}
#endif