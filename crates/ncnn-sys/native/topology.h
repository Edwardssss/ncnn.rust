#pragma once
#include <stddef.h>
#include <stdint.h>
#define NCNN_RS_UNSUPPORTED (-102)
#ifdef __cplusplus
extern "C"
{
#endif
    int ncnn_rs_os_affinity_supported(void);
    size_t ncnn_rs_os_processors(int current_thread, uint64_t* ids, size_t capacity);
    int ncnn_rs_os_physical_count(void);
    int ncnn_rs_thread_affinity_set(const uint64_t* ids, size_t count);
    int ncnn_rs_cpuset_capacity(void);
#ifdef __cplusplus
}
#endif
