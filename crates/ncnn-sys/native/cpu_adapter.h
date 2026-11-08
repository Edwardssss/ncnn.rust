#ifndef NCNN_RUST_CPU_H
#define NCNN_RUST_CPU_H
#include "c_api.h"
#ifdef __cplusplus
extern "C"
{
#endif
    /* Advanced calls use initialized native handles and caller-managed dependencies. */
    void* ncnn_rs_fast_malloc(size_t size);
    void ncnn_rs_fast_free(void* data);
    int ncnn_rs_mat_clone_with(ncnn_mat_t mat, ncnn_allocator_t allocator, ncnn_mat_t* out);
    int ncnn_rs_mat_reshape_with(ncnn_mat_t mat, int dims, const int* shape, ncnn_allocator_t allocator, ncnn_mat_t* out);
    int ncnn_rs_mat_info(ncnn_mat_t mat, size_t* total, int* bits);
    int ncnn_rs_mat_shape(ncnn_mat_t mat, ncnn_mat_t* out);
    int ncnn_rs_mat_external_batch(void* data, int dims, const int* shape, int batch, size_t elemsize, int packing, ncnn_mat_t* out);
    int ncnn_rs_mat_shallow(ncnn_mat_t mat, ncnn_mat_t* out);
    int ncnn_rs_mat_fill(ncnn_mat_t mat, float value, int integer);
    ncnn_paramdict_t ncnn_rs_param_clone(ncnn_paramdict_t pd);
    int ncnn_rs_net_resize_graph(ncnn_net_t net, int blobs, int layers);
    int ncnn_rs_net_set_blob(ncnn_net_t net, int index, int producer, int consumer, const int* shape);
    int ncnn_rs_net_set_layer(ncnn_net_t net, int index, ncnn_layer_t replacement);
    int ncnn_rs_layer_set_edges(ncnn_layer_t layer, int output, const int* indexes, int count);
    int ncnn_rs_layer_set_shape(ncnn_layer_t layer, int output, int index, const int* shape);
    int ncnn_rs_mat_view(ncnn_mat_t mat, int axis, int start, int count, ncnn_mat_t* out);
    void* ncnn_rs_layer_userdata(ncnn_layer_t layer);
    void ncnn_rs_layer_set_userdata(ncnn_layer_t layer, void* data);
    int ncnn_rs_layer_shape(ncnn_layer_t layer, int output, int index, int* shape);
    int ncnn_rs_net_blob(ncnn_net_t net, int index, int* producer, int* consumer, int* shape);
    int ncnn_rs_register_layer(ncnn_net_t net, const char* name, int index, ncnn_layer_creator_t creator, ncnn_layer_destroyer_t destroyer, void* userdata);
    void ncnn_rs_extractor_clear(ncnn_extractor_t ex);
    ncnn_extractor_t ncnn_rs_extractor_clone(ncnn_extractor_t ex);
    ncnn_layer_t ncnn_rs_layer_create(int index, int backend);
    ncnn_mat_t ncnn_rs_modelbin_load4(ncnn_modelbin_t model, int w, int h, int d, int c, int type);
    int ncnn_rs_net_custom_index(ncnn_net_t net, const char* name);
    int ncnn_rs_net_layer_count(ncnn_net_t net);
    void ncnn_rs_borrowed_layer_release(ncnn_layer_t layer);
    ncnn_layer_t ncnn_rs_net_layer(ncnn_net_t net, int index);
    int ncnn_rs_cpu_query(int key);
    int ncnn_rs_runtime_get(int key);
    int ncnn_rs_runtime_set(int key, int value);
    void* ncnn_rs_cpuset_create(void);
    void ncnn_rs_cpuset_destroy(void* mask);
    int ncnn_rs_cpuset_update(void* mask, int cpu, int enable);
    int ncnn_rs_cpuset_contains(const void* mask, int cpu);
    int ncnn_rs_cpuset_count(const void* mask);
    int ncnn_rs_cpuset_affinity(void* mask, int powersave);
    int ncnn_rs_cpuset_apply(const void* mask);
    int ncnn_rs_pool_configure(ncnn_allocator_t pool, int unlocked, float ratio, size_t threshold);
    void ncnn_rs_pool_clear(ncnn_allocator_t pool, int unlocked);
    int ncnn_rs_param_set_string(ncnn_paramdict_t pd, int id, const unsigned char* data, size_t size);
    size_t ncnn_rs_param_get_string(ncnn_paramdict_t pd, int id, unsigned char* out, size_t capacity);
    int ncnn_rs_extract(ncnn_extractor_t ex, int index, int type, ncnn_mat_t* out);
    void ncnn_rs_extractor_lightmode(ncnn_extractor_t ex, int enabled);
    void ncnn_rs_extractor_allocator(ncnn_extractor_t ex, int kind, ncnn_allocator_t allocator);
    void ncnn_rs_extractor_cache_hint(ncnn_extractor_t ex, int length);
    int ncnn_rs_layer_flag(ncnn_layer_t layer, int key, int value, int write);
    int ncnn_rs_tensor_transform(ncnn_mat_t src, ncnn_mat_t* out, int operation, int width, int height, ncnn_mat_t scale, ncnn_mat_t bias, ncnn_option_t options);
    int ncnn_rs_requantize(ncnn_mat_t src, ncnn_mat_t* out, ncnn_mat_t scale_in, ncnn_mat_t scale_out, ncnn_mat_t bias, int activation, ncnn_mat_t parameters, ncnn_option_t options);
    unsigned short ncnn_rs_f32_to_f16(float value);
    float ncnn_rs_f16_to_f32(unsigned short value);
    unsigned short ncnn_rs_f32_to_bf16(float value);
    float ncnn_rs_bf16_to_f32(unsigned short value);
    unsigned char ncnn_rs_f16_to_f8(unsigned short value, int brain);
    unsigned short ncnn_rs_f8_to_f16(unsigned char value, int brain);
#ifdef __cplusplus
}
#endif
#endif
