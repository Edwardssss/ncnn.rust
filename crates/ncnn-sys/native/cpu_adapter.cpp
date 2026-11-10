#include <stdio.h>
#include "cpu_adapter.h"
#include "topology.h"
#include "cpu.h"
#include "net.h"
#include "paramdict.h"
#include <cstring>
#include <new>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include "modelbin.h"
extern "C" void ncnn_rs_borrowed_layer_release(ncnn_layer_t layer)
{
    free(layer);
}

#define GUARD try
#define RECOVER                   \
    catch (const std::bad_alloc&) \
    {                             \
        return -100;              \
    }                             \
    catch (...)                   \
    {                             \
        return -101;              \
    }
extern "C" void* ncnn_rs_fast_malloc(size_t size)
{
    return ncnn::fastMalloc(size);
}
extern "C" void ncnn_rs_fast_free(void* data)
{
    ncnn::fastFree(data);
}
extern "C" int ncnn_rs_mat_clone_with(ncnn_mat_t mat, ncnn_allocator_t allocator, ncnn_mat_t* out)
{
    GUARD
    {
        ncnn::Mat m = ((ncnn::Mat*)mat)->clone(allocator ? (ncnn::Allocator*)allocator->pthis : 0);
        if (m.empty())
            return -100;
        *out = (ncnn_mat_t) new ncnn::Mat(m);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_mat_reshape_with(ncnn_mat_t mat, int dims, const int* s, ncnn_allocator_t allocator, ncnn_mat_t* out)
{
    GUARD
    {
        ncnn::Allocator* a = allocator ? (ncnn::Allocator*)allocator->pthis : 0;
        ncnn::Mat& src = *(ncnn::Mat*)mat;
        ncnn::Mat m;
        if (dims == 1)
            m = src.reshape(s[0], a);
        if (dims == 2)
            m = src.reshape(s[0], s[1], a);
        if (dims == 3)
            m = src.reshape(s[0], s[1], s[2], a);
        if (dims == 4)
            m = src.reshape(s[0], s[1], s[2], s[3], a);
        if (m.empty())
            return -1;
        *out = (ncnn_mat_t) new ncnn::Mat(m);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_mat_info(ncnn_mat_t mat, size_t* total, int* bits)
{
    ncnn::Mat& m = *(ncnn::Mat*)mat;
    *total = m.total();
    *bits = m.elembits();
    return m.empty();
}
extern "C" int ncnn_rs_mat_shape(ncnn_mat_t mat, ncnn_mat_t* out)
{
    GUARD
    {
        *out = (ncnn_mat_t) new ncnn::Mat(((ncnn::Mat*)mat)->shape());
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_mat_external_batch(void* data, int dims, const int* s, int batch, size_t elemsize, int packing, ncnn_mat_t* out)
{
    if (dims < 1 || dims > 4 || batch < 1 || packing < 1 || !elemsize)
        return -1;
    for (int i = 0; i < dims; i++)
        if (s[i] <= 0)
            return -1;
    GUARD
    {
        ncnn::Mat m;
        if (dims == 1)
            m = ncnn::Mat(s[0], data, elemsize, packing, batch);
        if (dims == 2)
            m = ncnn::Mat(s[0], s[1], data, elemsize, packing, batch);
        if (dims == 3)
            m = ncnn::Mat(s[0], s[1], s[2], data, elemsize, packing, batch);
        if (dims == 4)
            m = ncnn::Mat(s[0], s[1], s[2], s[3], data, elemsize, packing, batch);
        *out = (ncnn_mat_t) new ncnn::Mat(m);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_mat_shallow(ncnn_mat_t mat, ncnn_mat_t* out)
{
    GUARD
    {
        *out = (ncnn_mat_t) new ncnn::Mat(*(ncnn::Mat*)mat);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_mat_fill(ncnn_mat_t mat, float value, int integer)
{
    if (integer && (!std::isfinite(value) || (double)value < std::numeric_limits<int>::min() || (double)value > std::numeric_limits<int>::max()))
        return -1;
    if (integer)
        ((ncnn::Mat*)mat)->fill((int)value);
    else
        ((ncnn::Mat*)mat)->fill(value);
    return 0;
}
extern "C" ncnn_paramdict_t ncnn_rs_param_clone(ncnn_paramdict_t pd)
{
    try
    {
        return (ncnn_paramdict_t) new ncnn::ParamDict(*(ncnn::ParamDict*)pd);
    }
    catch (...)
    {
        return 0;
    }
}
static ncnn::Mat shape_value(const int* s)
{
    ncnn::Mat m;
    m.dims = s[0];
    m.w = s[1];
    m.h = s[2];
    m.d = s[3];
    m.c = s[4];
    m.n = s[5];
    return m;
}
extern "C" int ncnn_rs_net_resize_graph(ncnn_net_t net, int blobs, int layers)
{
    if (blobs < 0 || layers < 0)
        return -1;
    GUARD
    {
        ncnn::Net& n = *(ncnn::Net*)net->pthis;
        n.mutable_blobs().resize(blobs);
        n.mutable_layers().resize(layers, 0);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_net_set_blob(ncnn_net_t net, int index, int producer, int consumer, const int* shape)
{
    std::vector<ncnn::Blob>& blobs = ((ncnn::Net*)net->pthis)->mutable_blobs();
    if (index < 0 || (size_t)index >= blobs.size())
        return -1;
    blobs[index].producer = producer;
    blobs[index].consumer = consumer;
    blobs[index].shape = shape_value(shape);
    return 0;
}
extern "C" int ncnn_rs_net_set_layer(ncnn_net_t net, int index, ncnn_layer_t replacement)
{
    std::vector<ncnn::Layer*>& layers = ((ncnn::Net*)net->pthis)->mutable_layers();
    if (index < 0 || (size_t)index >= layers.size())
        return -1;
    layers[index] = (ncnn::Layer*)replacement->pthis;
    return 0;
}
extern "C" int ncnn_rs_layer_set_edges(ncnn_layer_t layer, int output, const int* indexes, int count)
{
    if (count < 0)
        return -1;
    GUARD
    {
        ncnn::Layer& l = *(ncnn::Layer*)layer->pthis;
        (output ? l.tops : l.bottoms).assign(indexes, indexes + count);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_layer_set_shape(ncnn_layer_t layer, int output, int index, const int* shape)
{
    ncnn::Layer& l = *(ncnn::Layer*)layer->pthis;
    std::vector<ncnn::Mat>& shapes = output ? l.top_shapes : l.bottom_shapes;
    if (index < 0 || (size_t)index >= shapes.size())
        return -1;
    shapes[index] = shape_value(shape);
    return 0;
}
static void copy_shape(const ncnn::Mat& m, int* s)
{
    s[0] = m.dims;
    s[1] = m.w;
    s[2] = m.h;
    s[3] = m.d;
    s[4] = m.c;
    s[5] = m.n;
}
extern "C" void* ncnn_rs_layer_userdata(ncnn_layer_t layer)
{
    return ((ncnn::Layer*)layer->pthis)->userdata;
}
extern "C" void ncnn_rs_layer_set_userdata(ncnn_layer_t layer, void* data)
{
    ((ncnn::Layer*)layer->pthis)->userdata = data;
}
extern "C" int ncnn_rs_layer_shape(ncnn_layer_t layer, int output, int index, int* shape)
{
    ncnn::Layer& l = *(ncnn::Layer*)layer->pthis;
    const std::vector<ncnn::Mat>& shapes = output ? l.top_shapes : l.bottom_shapes;
    if (index < 0 || (size_t)index >= shapes.size())
        return -1;
    copy_shape(shapes[index], shape);
    return 0;
}
extern "C" int ncnn_rs_net_blob(ncnn_net_t net, int index, int* producer, int* consumer, int* shape)
{
    const std::vector<ncnn::Blob>& blobs = ((ncnn::Net*)net->pthis)->blobs();
    if (index < 0 || (size_t)index >= blobs.size())
        return -1;
    const ncnn::Blob& b = blobs[index];
    *producer = b.producer;
    *consumer = b.consumer;
    copy_shape(b.shape, shape);
    return 0;
}
extern "C" int ncnn_rs_mat_view(ncnn_mat_t mat, int axis, int start, int count, ncnn_mat_t* out)
{
    GUARD
    {
        ncnn::Mat& m = *(ncnn::Mat*)mat;
        int size = axis == 0 ? m.w : axis == 1 ? m.h
                                 : axis == 2   ? m.d
                                 : axis == 3   ? m.c
                                 : axis == 4   ? m.n
                                               : 0;
        if (start < 0 || count <= 0 || start > size || count > size - start)
            return -1;
        // Each axis needs the extent it slices to exist; axis 0 slices a run of the first row, and
        // axis 4 slices the batch, whose count is independent of the rank.
        if ((axis == 1 && m.dims < 2) || ((axis == 2 || axis == 3) && m.dims < 3))
            return -1;
        ncnn::Mat view;
        switch (axis)
        {
        case 0:
            view = m.range(start, count);
            break;
        case 1:
            view = m.row_range(start, count);
            break;
        case 2:
            view = m.depth_range(start, count);
            break;
        case 3:
            view = m.channel_range(start, count);
            break;
        case 4:
            view = m.batch_range(start, count);
            break;
        default:
            return -1;
        }
        *out = (ncnn_mat_t) new ncnn::Mat(view);
        return 0;
    }
    RECOVER
}
struct CallbackFunctions
{
    decltype(((ncnn_layer_t)0)->forward_1) one;
    decltype(((ncnn_layer_t)0)->forward_n) many;
};
struct CallbackRegistry
{
    std::mutex mutex;
    std::map<ncnn_layer_t, CallbackFunctions> functions;
};
static CallbackRegistry& callback_registry()
{
    static CallbackRegistry registry;
    return registry;
}

static int complete_outputs(ncnn_mat_t* outputs, int count, int status)
{
    for (int i = 0; i < count; i++)
    {
        if (outputs[i])
            continue;
        // The upstream C layer dereferences every output even when a callback fails.
        outputs[i] = (ncnn_mat_t) new (std::nothrow) ncnn::Mat;
        if (!outputs[i])
            std::abort();
        if (status == 0)
            status = -100;
    }
    return status;
}
static int checked_forward_one(ncnn_layer_t layer, ncnn_mat_t input, ncnn_mat_t* output, ncnn_option_t opt)
{
    int result = -101;
    try
    {
        CallbackFunctions callbacks;
        {
            CallbackRegistry& registry = callback_registry();
            std::lock_guard<std::mutex> lock(registry.mutex);
            callbacks = registry.functions.at(layer);
        }
        if (callbacks.one)
            result = callbacks.one(layer, input, output, opt);
    }
    catch (...)
    {
    }
    return complete_outputs(output, 1, result);
}
static int checked_forward_many(ncnn_layer_t layer, const ncnn_mat_t* inputs, int count, ncnn_mat_t* outputs, int output_count, ncnn_option_t opt)
{
    int result = -101;
    try
    {
        CallbackFunctions callbacks;
        {
            CallbackRegistry& registry = callback_registry();
            std::lock_guard<std::mutex> lock(registry.mutex);
            callbacks = registry.functions.at(layer);
        }
        if (callbacks.many)
            result = callbacks.many(layer, inputs, count, outputs, output_count, opt);
    }
    catch (...)
    {
    }
    return complete_outputs(outputs, output_count, result);
}
static ncnn::Layer* registered_create(void* userdata)
{
    ncnn_net_custom_layer_factory_t f = (ncnn_net_custom_layer_factory_t)userdata;
    ncnn_layer_t handle = f->creator(f->userdata);
    if (!handle)
        return 0;
    try
    {
        CallbackRegistry& registry = callback_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.functions.emplace(handle, CallbackFunctions{handle->forward_1, handle->forward_n});
    }
    catch (...)
    {
        f->destroyer(handle, f->userdata);
        return 0;
    }
    handle->forward_1 = checked_forward_one;
    handle->forward_n = checked_forward_many;
    ncnn::Layer* layer = (ncnn::Layer*)handle->pthis;
    layer->userdata = handle;
    return layer;
}
static void registered_destroy(ncnn::Layer* layer, void* userdata)
{
    ncnn_net_custom_layer_factory_t f = (ncnn_net_custom_layer_factory_t)userdata;
    ncnn_layer_t handle = (ncnn_layer_t)layer->userdata;
    {
        CallbackRegistry& registry = callback_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        std::map<ncnn_layer_t, CallbackFunctions>::iterator found = registry.functions.find(handle);
        if (found != registry.functions.end())
        {
            handle->forward_1 = found->second.one;
            handle->forward_n = found->second.many;
            registry.functions.erase(found);
        }
    }
    f->destroyer(handle, f->userdata);
}
extern "C" int ncnn_rs_register_layer(ncnn_net_t net, const char* name, int index, ncnn_layer_creator_t creator, ncnn_layer_destroyer_t destroyer, void* userdata)
{
    if (!creator || !destroyer)
        return -1;
    ncnn_net_custom_layer_factory_t f = (ncnn_net_custom_layer_factory_t)calloc(1, sizeof(*f));
    if (!f)
        return -100;
    f->creator = creator;
    f->destroyer = destroyer;
    f->userdata = userdata;
    try
    {
        ncnn::Net* n = (ncnn::Net*)net->pthis;
        int result = name ? n->register_custom_layer(name, registered_create, registered_destroy, f) : n->register_custom_layer(index, registered_create, registered_destroy, f);
        if (result)
        {
            free(f);
            return result;
        }
        f->next = net->custom_layer_factory;
        net->custom_layer_factory = f;
        return 0;
    }
    catch (...)
    {
        free(f);
        return -101;
    }
}
extern "C" void ncnn_rs_extractor_clear(ncnn_extractor_t ex)
{
    ((ncnn::Extractor*)ex)->clear();
}
extern "C" ncnn_extractor_t ncnn_rs_extractor_clone(ncnn_extractor_t ex)
{
    try
    {
        return (ncnn_extractor_t) new ncnn::Extractor(*(ncnn::Extractor*)ex);
    }
    catch (...)
    {
        return 0;
    }
}
extern "C" ncnn_layer_t ncnn_rs_layer_create(int index, int backend)
{
    ncnn_layer_t handle = ncnn_layer_create_by_typeindex(index);
    if (!handle)
        return 0;
    try
    {
        ncnn::Layer* replacement = backend == 0 ? ncnn::create_layer_naive(index) : ncnn::create_layer_cpu(index);
        if (!replacement)
        {
            ncnn_layer_destroy(handle);
            return 0;
        }
        delete (ncnn::Layer*)handle->pthis;
        handle->pthis = replacement;
        return handle;
    }
    catch (...)
    {
        ncnn_layer_destroy(handle);
        return 0;
    }
}
extern "C" ncnn_mat_t ncnn_rs_modelbin_load4(ncnn_modelbin_t model, int w, int h, int d, int c, int type)
{
    try
    {
        return (ncnn_mat_t) new ncnn::Mat(((ncnn::ModelBin*)model->pthis)->load(w, h, d, c, type));
    }
    catch (...)
    {
        return 0;
    }
}
extern "C" int ncnn_rs_net_custom_index(ncnn_net_t net, const char* name)
{
    return ((ncnn::Net*)net->pthis)->custom_layer_to_index(name);
}
extern "C" int ncnn_rs_net_layer_count(ncnn_net_t net)
{
    return (int)((ncnn::Net*)net->pthis)->layers().size();
}
extern "C" ncnn_layer_t ncnn_rs_net_layer(ncnn_net_t net, int index)
{
    const std::vector<ncnn::Layer*>& layers = ((ncnn::Net*)net->pthis)->layers();
    if (index < 0 || (size_t)index >= layers.size())
        return 0;
    ncnn_layer_t handle = ncnn_layer_create_by_typeindex(0);
    if (!handle)
        return 0;
    delete (ncnn::Layer*)handle->pthis;
    handle->pthis = layers[index];
    return handle;
}

extern "C" int ncnn_rs_cpu_query(int key)
{
    GUARD
    {
        switch (key)
        {
#define QUERY(k, f) \
    case k:         \
        return ncnn::f()
            QUERY(0, get_cpu_count);
            QUERY(1, get_little_cpu_count);
            QUERY(2, get_big_cpu_count);
            QUERY(3, get_physical_cpu_count);
            QUERY(4, get_physical_little_cpu_count);
            QUERY(5, get_physical_big_cpu_count);
            QUERY(6, get_cpu_level2_cache_size);
            QUERY(7, get_cpu_level3_cache_size);
            QUERY(8, cpu_support_x86_avx);
            QUERY(9, cpu_support_x86_fma);
            QUERY(10, cpu_support_x86_fma4);
            QUERY(11, cpu_support_x86_xop);
            QUERY(12, cpu_support_x86_f16c);
            QUERY(13, cpu_support_x86_avx2);
            QUERY(14, cpu_support_x86_avx_vnni);
            QUERY(15, cpu_support_x86_avx_vnni_int8);
            QUERY(16, cpu_support_x86_avx_vnni_int16);
            QUERY(17, cpu_support_x86_avx_ne_convert);
            QUERY(18, cpu_support_x86_avx512);
            QUERY(19, cpu_support_x86_avx512_vnni);
            QUERY(20, cpu_support_x86_avx512_bf16);
            QUERY(21, cpu_support_x86_avx512_fp16);
            QUERY(22, cpu_support_arm_edsp);
            QUERY(23, cpu_support_arm_neon);
            QUERY(24, cpu_support_arm_vfpv4);
            QUERY(25, cpu_support_arm_asimdhp);
            QUERY(26, cpu_support_arm_cpuid);
            QUERY(27, cpu_support_arm_asimddp);
            QUERY(28, cpu_support_arm_asimdfhm);
            QUERY(29, cpu_support_arm_bf16);
            QUERY(30, cpu_support_arm_i8mm);
            QUERY(31, cpu_support_arm_sve);
            QUERY(32, cpu_support_arm_sve2);
            QUERY(33, cpu_support_arm_svebf16);
            QUERY(34, cpu_support_arm_svei8mm);
            QUERY(35, cpu_support_arm_svef32mm);
#undef QUERY
        default:
            return -1;
        }
    }
    RECOVER
}
extern "C" int ncnn_rs_runtime_get(int key)
{
    GUARD
    {
        switch (key)
        {
        case 0:
            return ncnn::get_cpu_powersave();
        case 1:
            return ncnn::get_omp_num_threads();
        case 2:
            return ncnn::get_omp_dynamic();
        case 3:
            return ncnn::get_omp_thread_num();
        case 4:
            return ncnn::get_kmp_blocktime();
        case 5:
            return ncnn::get_flush_denormals();
        default:
            return -1;
        }
    }
    RECOVER
}
extern "C" int ncnn_rs_runtime_set(int key, int value)
{
    GUARD
    {
        switch (key)
        {
        case 0:
            return value >= 0 && value <= 2 ? ncnn::set_cpu_powersave(value) : -1;
        case 1:
            if (value < 1)
                return -1;
            ncnn::set_omp_num_threads(value);
            return 0;
        case 2:
            if (value != 0 && value != 1)
                return -1;
            ncnn::set_omp_dynamic(value);
            return 0;
        case 4:
            if (value < 0)
                return -1;
            ncnn::set_kmp_blocktime(value);
            return 0;
        case 5:
            return value >= 0 && value <= 3 ? ncnn::set_flush_denormals(value) : -1;
        default:
            return -1;
        }
    }
    RECOVER
}
static int mask_capacity()
{
#if defined(_WIN32)
    return sizeof(ULONG_PTR) * 8;
#elif defined(__linux__)
    return CPU_SETSIZE;
#elif defined(__APPLE__)
    return sizeof(unsigned int) * 8;
#else
    return 0;
#endif
}
extern "C" int ncnn_rs_cpuset_capacity()
{
    return mask_capacity();
}
extern "C" void* ncnn_rs_cpuset_create()
{
    try
    {
        return new ncnn::CpuSet;
    }
    catch (...)
    {
        return 0;
    }
}
extern "C" void ncnn_rs_cpuset_destroy(void* mask)
{
    delete (ncnn::CpuSet*)mask;
}
extern "C" int ncnn_rs_cpuset_update(void* mask, int cpu, int enable)
{
    if (cpu < 0 || cpu >= mask_capacity())
        return -1;
    if (enable)
        ((ncnn::CpuSet*)mask)->enable(cpu);
    else
        ((ncnn::CpuSet*)mask)->disable(cpu);
    return 0;
}
extern "C" int ncnn_rs_cpuset_contains(const void* mask, int cpu)
{
    if (cpu < 0 || cpu >= mask_capacity())
        return -1;
    return ((const ncnn::CpuSet*)mask)->is_enabled(cpu);
}
extern "C" int ncnn_rs_cpuset_count(const void* mask)
{
    return ((const ncnn::CpuSet*)mask)->num_enabled();
}
extern "C" int ncnn_rs_cpuset_affinity(void* mask, int powersave)
{
    if (!mask || powersave < 0 || powersave > 2)
        return -1;
    GUARD
    {
        *(ncnn::CpuSet*)mask = ncnn::get_cpu_thread_affinity_mask(powersave);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_cpuset_apply(const void* mask)
{
#if defined(_WIN32) || defined(__linux__)
    if (!mask || !ncnn_rs_cpuset_count(mask))
        return -1;
    GUARD
    {
        return ncnn::set_cpu_thread_affinity(*(const ncnn::CpuSet*)mask);
    }
    RECOVER
#else
    (void)mask;
    return NCNN_RS_UNSUPPORTED;
#endif
}
extern "C" int ncnn_rs_pool_configure(ncnn_allocator_t pool, int unlocked, float ratio, size_t threshold)
{
    if (!(ratio >= 0.f && ratio <= 1.f))
        return -1;
    if (unlocked)
    {
        ncnn::UnlockedPoolAllocator* p = (ncnn::UnlockedPoolAllocator*)pool->pthis;
        p->set_size_compare_ratio(ratio);
        p->set_size_drop_threshold(threshold);
    }
    else
    {
        ncnn::PoolAllocator* p = (ncnn::PoolAllocator*)pool->pthis;
        p->set_size_compare_ratio(ratio);
        p->set_size_drop_threshold(threshold);
    }
    return 0;
}
extern "C" void ncnn_rs_pool_clear(ncnn_allocator_t pool, int unlocked)
{
    if (unlocked)
        ((ncnn::UnlockedPoolAllocator*)pool->pthis)->clear();
    else
        ((ncnn::PoolAllocator*)pool->pthis)->clear();
}
extern "C" int ncnn_rs_param_set_string(ncnn_paramdict_t pd, int id, const unsigned char* data, size_t size)
{
    if (id < 0 || id >= 32)
        return -1;
    GUARD
    {
        ((ncnn::ParamDict*)pd)->set(id, std::string((const char*)data, size));
        return 0;
    }
    RECOVER
}
extern "C" size_t ncnn_rs_param_get_string(ncnn_paramdict_t pd, int id, unsigned char* out, size_t capacity)
{
    if (id < 0 || id >= 32)
        return (size_t)-1;
    try
    {
        std::string s = ((ncnn::ParamDict*)pd)->get(id, std::string());
        if (capacity >= s.size() && !s.empty())
            memcpy(out, s.data(), s.size());
        return s.size();
    }
    catch (...)
    {
        return (size_t)-1;
    }
}
extern "C" int ncnn_rs_extract(ncnn_extractor_t ex, int index, int type, ncnn_mat_t* out)
{
    if (type != 0 && type != 1)
        return -1;
    GUARD
    {
        ncnn::Mat m;
        int ret = ((ncnn::Extractor*)ex)->extract(index, m, type);
        if (ret)
            return ret;
        *out = (ncnn_mat_t) new ncnn::Mat(m);
        return 0;
    }
    RECOVER
}
extern "C" void ncnn_rs_extractor_lightmode(ncnn_extractor_t ex, int enabled)
{
    ((ncnn::Extractor*)ex)->set_light_mode(enabled != 0);
}
extern "C" void ncnn_rs_extractor_allocator(ncnn_extractor_t ex, int kind, ncnn_allocator_t allocator)
{
    ncnn::Allocator* a = allocator ? (ncnn::Allocator*)allocator->pthis : 0;
    if (kind == 0)
        ((ncnn::Extractor*)ex)->set_blob_allocator(a);
    if (kind == 1)
        ((ncnn::Extractor*)ex)->set_workspace_allocator(a);
    if (kind == 2)
        ((ncnn::Extractor*)ex)->set_kvcache_allocator(a);
}
extern "C" void ncnn_rs_extractor_cache_hint(ncnn_extractor_t ex, int length)
{
    ((ncnn::Extractor*)ex)->set_kvcache_max_seqlen_hint(length);
}
extern "C" int ncnn_rs_layer_flag(ncnn_layer_t layer, int key, int value, int write)
{
    ncnn::Layer& l = *(ncnn::Layer*)layer->pthis;
    switch (key)
    {
    case 0:
        if (write)
            l.support_int8_storage = value;
        return l.support_int8_storage;
    case 1:
        if (write)
            l.support_batch = value;
        return l.support_batch;
    case 2:
        if (write)
            l.featmask = value;
        return l.featmask;
    default:
        return -1;
    }
}
extern "C" int ncnn_rs_tensor_transform(ncnn_mat_t src, ncnn_mat_t* out, int op, int w, int h, ncnn_mat_t scale, ncnn_mat_t bias, ncnn_option_t options)
{
    GUARD
    {
        const ncnn::Mat& m = *(ncnn::Mat*)src;
        ncnn::Mat dst;
        const ncnn::Option opt = options ? *(ncnn::Option*)options : ncnn::Option();
        switch (op)
        {
        case 0:
            ncnn::resize_nearest(m, dst, w, h, opt);
            break;
        case 1:
            ncnn::resize_bilinear(m, dst, w, h, opt);
            break;
        case 2:
            ncnn::resize_bicubic(m, dst, w, h, opt);
            break;
        case 3:
            ncnn::cast_float32_to_float16(m, dst, opt);
            break;
        case 4:
            ncnn::cast_float16_to_float32(m, dst, opt);
            break;
        case 5:
            ncnn::cast_int8_to_float32(m, dst, opt);
            break;
        case 6:
            ncnn::cast_float32_to_bfloat16(m, dst, opt);
            break;
        case 7:
            ncnn::cast_bfloat16_to_float32(m, dst, opt);
            break;
        case 8:
            ncnn::quantize_to_int8(m, dst, *(ncnn::Mat*)scale, opt);
            break;
        case 9:
            ncnn::dequantize_from_int32(m, dst, *(ncnn::Mat*)scale, *(ncnn::Mat*)bias, opt);
            break;
        default:
            return -1;
        }
        if (dst.empty())
            return -100;
        *out = (ncnn_mat_t) new ncnn::Mat(dst);
        return 0;
    }
    RECOVER
}
extern "C" int ncnn_rs_requantize(ncnn_mat_t src, ncnn_mat_t* out, ncnn_mat_t si, ncnn_mat_t so, ncnn_mat_t bias, int activation, ncnn_mat_t parameters, ncnn_option_t options)
{
    GUARD
    {
        ncnn::Mat dst;
        ncnn::requantize_from_int32_to_int8(*(ncnn::Mat*)src, dst, *(ncnn::Mat*)si, *(ncnn::Mat*)so, *(ncnn::Mat*)bias, activation, *(ncnn::Mat*)parameters, options ? *(ncnn::Option*)options : ncnn::Option());
        if (dst.empty())
            return -100;
        *out = (ncnn_mat_t) new ncnn::Mat(dst);
        return 0;
    }
    RECOVER
}
extern "C" unsigned short ncnn_rs_f32_to_f16(float v)
{
    return ncnn::float32_to_float16(v);
}
extern "C" float ncnn_rs_f16_to_f32(unsigned short v)
{
    return ncnn::float16_to_float32(v);
}
extern "C" unsigned short ncnn_rs_f32_to_bf16(float v)
{
    return ncnn::float32_to_bfloat16(v);
}
extern "C" float ncnn_rs_bf16_to_f32(unsigned short v)
{
    return ncnn::bfloat16_to_float32(v);
}
extern "C" unsigned char ncnn_rs_f16_to_f8(unsigned short v, int brain)
{
    return brain ? ncnn::float16_to_bfloat8(v) : ncnn::float16_to_float8(v);
}
extern "C" unsigned short ncnn_rs_f8_to_f16(unsigned char v, int brain)
{
    return brain ? ncnn::bfloat8_to_float16(v) : ncnn::float8_to_float16(v);
}
