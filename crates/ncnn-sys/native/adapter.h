#ifndef NCNN_RUST_ADAPTER_H
#define NCNN_RUST_ADAPTER_H
#include <stdio.h>
#include "c_api.h"
#include "cpu_adapter.h"
#include "topology.h"
#include "image_adapter.h"
#ifdef __cplusplus
extern "C"
{
#endif
    int ncnn_rs_cpu_count(void);
#if NCNN_VULKAN
    int ncnn_rs_gpu_initialize(void);
    int ncnn_rs_gpu_count(void);
    int ncnn_rs_gpu_default(void);
    const char* ncnn_rs_gpu_name(int index);
    unsigned int ncnn_rs_gpu_property(int index, int property);
    int ncnn_rs_gpu_valid(int index);
    int ncnn_rs_net_vulkan(ncnn_net_t net, int index);
    int ncnn_rs_reference_vulkan_add(int index, const float* a, const float* b, int count, float* out);
    void* ncnn_rs_gpu_allocator_create(int index);
    void ncnn_rs_gpu_allocator_destroy(void* allocator);
    void* ncnn_rs_gpu_upload(void* allocator, ncnn_mat_t mat);
    void ncnn_rs_gpu_tensor_destroy(void* tensor);
    int ncnn_rs_gpu_download(void* allocator, void* tensor, ncnn_mat_t* out);
    size_t ncnn_rs_gpu_tensor_bytes(void* tensor);
    int ncnn_rs_gpu_buffer_info(void* tensor, unsigned long long* buffer, size_t* offset, size_t* capacity, unsigned int* access, unsigned int* stage);
    void* ncnn_rs_gpu_command_create(void* allocator, ncnn_net_t net);
    void ncnn_rs_gpu_command_destroy(void* command);
    int ncnn_rs_gpu_command_input(void* command, const char* name, void* tensor);
    void* ncnn_rs_gpu_command_extract(void* command, const char* name);
    int ncnn_rs_gpu_command_wait(void* command);
    int ncnn_rs_gpu_cache_save(int index, unsigned char** data, size_t* size);
    int ncnn_rs_gpu_cache_load(int index, const unsigned char* data, size_t size);
    void ncnn_rs_gpu_cache_free(unsigned char* data);
    size_t ncnn_rs_gpu_heap_budget(int index);
    unsigned long long ncnn_rs_gpu_device_handle(int index);
    void* ncnn_rs_gpu_import(void* allocator, unsigned long long buffer, size_t offset, size_t capacity, int count, unsigned int access, unsigned int stage);
#endif
    int ncnn_rs_option_set(ncnn_option_t opt, int key, int value);
    int ncnn_rs_option_get(ncnn_option_t opt, int key, int* value);
    int ncnn_rs_mat_copy_f32(const float* data, int dims, const int* shape, ncnn_mat_t* out);
    int ncnn_rs_mat_clone(ncnn_mat_t mat, ncnn_mat_t* out);
    int ncnn_rs_mat_normalized(ncnn_mat_t mat, const float* mean, const float* norm, ncnn_mat_t* out);
    int ncnn_rs_mat_read_f32(ncnn_mat_t mat, float* out, size_t count);
    int ncnn_rs_mat_reshape(ncnn_mat_t mat, int dims, const int* shape, int independent, ncnn_mat_t* out);
    int ncnn_rs_load_bytes(ncnn_net_t net, const unsigned char* data, size_t size, int model);
    typedef ncnn_mat_t (*ncnn_rs_model_load_t)(void* userdata, int dims, int w, int h, int d, int c, int type);
    ncnn_modelbin_t ncnn_rs_modelbin_callbacks(ncnn_rs_model_load_t load, void* userdata);
    int ncnn_rs_layer_load_model(ncnn_layer_t layer, ncnn_modelbin_t model);
    ncnn_datareader_t ncnn_rs_reader_memory(const unsigned char* data, size_t size);
    size_t ncnn_rs_reader_reference(ncnn_datareader_t reader, size_t size, const void** out);
    ncnn_datareader_t ncnn_rs_reader_create(const unsigned char* data, size_t size);
    ncnn_modelbin_t ncnn_rs_modelbin_from_mats(const ncnn_mat_t* mats, int count);
    int ncnn_rs_reference_add(const float* a, const float* b, int count, float* out);
#ifdef __cplusplus
}
#endif
#endif
