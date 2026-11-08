#include "adapter.h"
#if NCNN_VULKAN
#include "gpu.h"
#include "net.h"
#include "command.h"
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace
{
std::mutex gpu_mutex;
}
extern "C"
{
    int ncnn_rs_gpu_initialize(void)
    {
        std::lock_guard<std::mutex> lock(gpu_mutex);
        static const int result = []
        {
            const int status = ncnn::create_gpu_instance();
            // The executable exit callback runs before shared-library and driver teardown.
            // Native cleanup is idempotent when ncnn's own exit callback runs later.
            if (status == 0)
                std::atexit(ncnn::destroy_gpu_instance);
            return status;
        }();
        return result;
    }
    int ncnn_rs_gpu_count(void)
    {
        return ncnn::get_gpu_count();
    }
    int ncnn_rs_gpu_default(void)
    {
        return ncnn::get_default_gpu_index();
    }
    const char* ncnn_rs_gpu_name(int index)
    {
        return ncnn::get_gpu_info(index).device_name();
    }
    unsigned int ncnn_rs_gpu_property(int index, int property)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(index);
        switch (property)
        {
        case 0:
            return info.api_version();
        case 1:
            return info.vendor_id();
        case 2:
            return info.device_id();
        default:
            return info.type();
        }
    }
    int ncnn_rs_gpu_valid(int index)
    {
        std::lock_guard<std::mutex> lock(gpu_mutex);
        if (index < 0 || index >= ncnn::get_gpu_count())
            return 0;
        const ncnn::VulkanDevice* device = ncnn::get_gpu_device(index);
        return device && device->is_valid();
    }
    int ncnn_rs_reference_vulkan_add(int index, const float* a, const float* b, int count, float* out)
    {
        if (!ncnn_rs_gpu_valid(index) || count < 1 || !a || !b || !out)
            return -1;
        ncnn::VkBlobAllocator blob(ncnn::get_gpu_device(index));
        ncnn::VkStagingAllocator staging(ncnn::get_gpu_device(index));
        ncnn::Net net;
        net.opt.blob_vkallocator = &blob;
        net.opt.workspace_vkallocator = &blob;
        net.opt.staging_vkallocator = &staging;
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.set_vulkan_device(index);
        if (net.load_param_mem("7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n") != 0)
            return -2;
        const unsigned char empty[4] = {};
        if (net.load_model(empty) != 0)
            return -3;
        ncnn::Mat left(count), right(count);
        std::memcpy(left.data, a, count * sizeof(float));
        std::memcpy(right.data, b, count * sizeof(float));
        ncnn::Extractor ex = net.create_extractor();
        if (ex.input("a", left) || ex.input("b", right))
            return -4;
        ncnn::VkCompute cmd(net.vulkan_device());
        ncnn::VkMat gpu_output;
        if (ex.extract("sum", gpu_output, cmd) || gpu_output.empty())
            return -5;
        ncnn::Mat result;
        cmd.record_download(gpu_output, result, net.opt);
        if (cmd.submit_and_wait() || result.total() * result.elempack != static_cast<size_t>(count))
            return -6;
        std::memcpy(out, result.data, count * sizeof(float));
        return 0;
    }
    int ncnn_rs_net_vulkan(ncnn_net_t net, int index)
    {
        if (!ncnn_rs_gpu_valid(index))
            return -1;
        ncnn_net_set_vulkan_device(net, index);
        ncnn_option_set_use_vulkan_compute(ncnn_net_get_option(net), 1);
        return 0;
    }
}
#endif

#if NCNN_VULKAN
#include "pipelinecache.h"
#include <vector>
#include <new>
#include <memory>
#include <limits>
struct RsGpuAllocator
{
    const ncnn::VulkanDevice* device;
    ncnn::VkBlobAllocator blob;
    ncnn::VkStagingAllocator staging;
    explicit RsGpuAllocator(int index) : device(ncnn::get_gpu_device(index)), blob(device), staging(device) {}
    ncnn::Option option()
    {
        ncnn::Option o;
        o.use_vulkan_compute = true;
        o.use_fp16_packed = false;
        o.use_fp16_storage = false;
        o.use_fp16_arithmetic = false;
        o.use_packing_layout = false;
        o.blob_vkallocator = &blob;
        o.workspace_vkallocator = &blob;
        o.staging_vkallocator = &staging;
        return o;
    }
};
struct RsGpuTensor
{
    ncnn::VkBufferMemory external;
    ncnn::VkMat mat;
};
struct RsGpuCommand
{
    RsGpuAllocator* allocator;
    ncnn::Option inference_option;
    ncnn::Extractor extractor;
    std::vector<ncnn::VkMat> retained;
    ncnn::VkCompute compute;
    RsGpuCommand(RsGpuAllocator* a, ncnn::Net* n) : allocator(a), extractor(n->create_extractor()), compute(a->device)
    {
        inference_option = n->opt;
        inference_option.blob_vkallocator = &a->blob;
        inference_option.workspace_vkallocator = &a->blob;
        inference_option.staging_vkallocator = &a->staging;
        extractor.set_blob_vkallocator(&a->blob);
        extractor.set_workspace_vkallocator(&a->blob);
        extractor.set_staging_vkallocator(&a->staging);
#if NCNN_BENCHMARK
        // External command extraction uses layer timestamp slots just like native extraction.
        if (compute.create_query_pool((uint32_t)n->layers().size() * 2) != 0)
            throw std::runtime_error("GPU timestamp query pool creation failed");
#endif
    }
};
extern "C"
{
    void* ncnn_rs_gpu_allocator_create(int index)
    {
        try
        {
            return new RsGpuAllocator(index);
        }
        catch (...)
        {
            return 0;
        }
    }
    void ncnn_rs_gpu_allocator_destroy(void* p)
    {
        try
        {
            delete (RsGpuAllocator*)p;
        }
        catch (...)
        {
        }
    }
    void ncnn_rs_gpu_tensor_destroy(void* p)
    {
        try
        {
            delete (RsGpuTensor*)p;
        }
        catch (...)
        {
        }
    }
    void* ncnn_rs_gpu_upload(void* p, ncnn_mat_t mat)
    {
        try
        {
            RsGpuAllocator* a = (RsGpuAllocator*)p;
            std::unique_ptr<RsGpuTensor> t(new RsGpuTensor);
            ncnn::VkCompute cmd(a->device);
            cmd.record_upload(*(ncnn::Mat*)mat, t->mat, a->option());
            if (cmd.submit_and_wait() != 0 || t->mat.empty())
            {
                return 0;
            }
            return t.release();
        }
        catch (...)
        {
            return 0;
        }
    }
    int ncnn_rs_gpu_download(void* p, void* tensor, ncnn_mat_t* out)
    {
        *out = 0;
        try
        {
            RsGpuAllocator* a = (RsGpuAllocator*)p;
            ncnn::Mat result;
            ncnn::VkMat detached;
            ncnn::VkCompute cmd(a->device);
            const RsGpuTensor* source = (RsGpuTensor*)tensor;
            if (source->mat.data == &source->external)
            {
                // UMA downloads map native allocations; imported descriptors have no mapped address.
                cmd.record_clone(source->mat, detached, a->option());
                if (detached.empty())
                    return -1;
                cmd.record_download(detached, result, a->option());
            }
            else
            {
                cmd.record_download(source->mat, result, a->option());
            }
            int code = cmd.submit_and_wait();
            if (code != 0)
                return code;
            if (result.empty())
                return -100;
            // record_download owns a host allocation and submit_and_wait completes its copy.
            *out = (ncnn_mat_t) new ncnn::Mat(result);
            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }
    size_t ncnn_rs_gpu_tensor_bytes(void* p)
    {
        try
        {
            return ((RsGpuTensor*)p)->mat.total() * ((RsGpuTensor*)p)->mat.elemsize;
        }
        catch (...)
        {
            return 0;
        }
    }
    int ncnn_rs_gpu_buffer_info(void* p, unsigned long long* buffer, size_t* offset, size_t* capacity, unsigned int* access, unsigned int* stage)
    {
        try
        {
            const ncnn::VkBufferMemory* m = ((RsGpuTensor*)p)->mat.data;
            if (!m)
                return -1;
            *buffer = (unsigned long long)m->buffer;
            *offset = m->offset;
            *capacity = m->capacity;
            *access = m->access_flags;
            *stage = m->stage_flags;
            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }
    void* ncnn_rs_gpu_command_create(void* p, ncnn_net_t net)
    {
        try
        {
            return new RsGpuCommand((RsGpuAllocator*)p, (ncnn::Net*)net->pthis);
        }
        catch (...)
        {
            return 0;
        }
    }
    void ncnn_rs_gpu_command_destroy(void* p)
    {
        try
        {
            delete (RsGpuCommand*)p;
        }
        catch (...)
        {
        }
    }
    int ncnn_rs_gpu_command_input(void* p, const char* name, void* t)
    {
        try
        {
            RsGpuCommand* c = (RsGpuCommand*)p;
            ncnn::VkMat copy;
            c->compute.record_clone(((RsGpuTensor*)t)->mat, copy, c->allocator->option());
            if (copy.empty())
                return -1;
            ncnn::VkMat converted;
            c->allocator->device->convert_packing(copy, converted, 1, c->inference_option.use_fp16_storage ? 2 : 1, c->compute, c->inference_option);
            c->retained.push_back(copy);
            c->retained.push_back(converted);
            return c->extractor.input(name, converted);
        }
        catch (...)
        {
            return -1;
        }
    }
    void* ncnn_rs_gpu_command_extract(void* p, const char* name)
    {
        try
        {
            RsGpuCommand* c = (RsGpuCommand*)p;
            std::unique_ptr<RsGpuTensor> t(new RsGpuTensor);
            ncnn::VkMat output;
            if (c->extractor.extract(name, output, c->compute) != 0 || output.empty())
            {
                return 0;
            }
            ncnn::VkMat fp32;
            c->allocator->device->convert_packing(output, fp32, 1, 1, c->compute, c->allocator->option());
            c->retained.push_back(output);
            c->retained.push_back(fp32);
            c->compute.record_clone(fp32, t->mat, c->allocator->option());
            if (t->mat.empty())
            {
                return 0;
            }
            return t.release();
        }
        catch (...)
        {
            return 0;
        }
    }
    int ncnn_rs_gpu_command_wait(void* p)
    {
        try
        {
            return ((RsGpuCommand*)p)->compute.submit_and_wait();
        }
        catch (...)
        {
            return -1;
        }
    }
    int ncnn_rs_gpu_cache_save(int index, unsigned char** data, size_t* size)
    {
        *data = 0;
        *size = 0;
        try
        {
            std::vector<unsigned char> bytes;
            int code = ncnn::get_gpu_device(index)->get_pipeline_cache()->save_cache(bytes);
            if (code != 0)
                return code;
            *data = (unsigned char*)malloc(bytes.size());
            if (!*data)
                return -1;
            memcpy(*data, bytes.data(), bytes.size());
            *size = bytes.size();
            return 0;
        }
        catch (...)
        {
            return -1;
        }
    }
    int ncnn_rs_gpu_cache_load(int index, const unsigned char* data, size_t size)
    {
        try
        {
            return ncnn::get_gpu_device(index)->get_pipeline_cache()->load_cache(data, size);
        }
        catch (...)
        {
            return -1;
        }
    }
    void ncnn_rs_gpu_cache_free(unsigned char* data)
    {
        free(data);
    }
    size_t ncnn_rs_gpu_heap_budget(int index)
    {
        try
        {
            return ncnn::get_gpu_device(index)->get_heap_budget();
        }
        catch (...)
        {
            return 0;
        }
    }
    unsigned long long ncnn_rs_gpu_device_handle(int index)
    {
        try
        {
            return (unsigned long long)ncnn::get_gpu_device(index)->vkdevice();
        }
        catch (...)
        {
            return 0;
        }
    }
    void* ncnn_rs_gpu_import(void* p, unsigned long long buffer, size_t offset, size_t capacity, int count, unsigned int access, unsigned int stage)
    {
        try
        {
            std::unique_ptr<RsGpuTensor> t(new RsGpuTensor);
            RsGpuAllocator* a = (RsGpuAllocator*)p;
            const size_t alignment = a->device->info.buffer_offset_alignment();
            if (!buffer || count <= 0 || (alignment && offset % alignment != 0) || capacity > std::numeric_limits<size_t>::max() - offset || (size_t)count > (std::numeric_limits<size_t>::max() - 15) / 4 || (((size_t)count * 4 + 15) & ~(size_t)15) > capacity || !stage)
                return 0;
            memset(&t->external, 0, sizeof(t->external));
            t->external.buffer = (VkBuffer)buffer;
            t->external.offset = offset;
            t->external.capacity = capacity;
            t->external.access_flags = access;
            t->external.stage_flags = stage;
            t->mat = ncnn::VkMat(count, &t->external, (size_t)4, &((RsGpuAllocator*)p)->blob);
            return t.release();
        }
        catch (...)
        {
            return 0;
        }
    }
}
#endif
