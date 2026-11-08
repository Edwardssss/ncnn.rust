#include <net.h>
#if NCNN_VULKAN
#include <gpu.h>
#include <command.h>
#endif

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static FILE* open_file(const std::filesystem::path& path)
{
#ifdef _WIN32
    return _wfopen(path.c_str(), L"rb");
#else
    return fopen(path.c_str(), "rb");
#endif
}

static void write_bytes(const void* data, size_t size)
{
    if (fwrite(data, 1, size, stdout) != size)
        throw std::runtime_error("Reference output write failed");
}

static void write_u32(uint32_t value)
{
    write_bytes(&value, sizeof(value));
}

static void write_u64(uint64_t value)
{
    write_bytes(&value, sizeof(value));
}

static int run(const std::filesystem::path& param_path, const std::filesystem::path& model_path,
               int width, int height, int threads, bool packing, int profile)
{
    std::unique_ptr<FILE, int (*)(FILE*)> param(open_file(param_path), fclose);
    std::unique_ptr<FILE, int (*)(FILE*)> model(open_file(model_path), fclose);
    if (!param || !model || width <= 0 || height <= 0 || threads <= 0)
        throw std::runtime_error("Invalid reference input");

    const bool vulkan = profile >= 10;
    if (vulkan) profile -= 10;
#if NCNN_VULKAN
    struct GpuLifetime { bool active; ~GpuLifetime() { if (active) ncnn::destroy_gpu_instance(); } } gpu_lifetime{vulkan};
    std::unique_ptr<ncnn::VkBlobAllocator> blob;
    std::unique_ptr<ncnn::VkStagingAllocator> staging;
#endif
    ncnn::Net net;
    if (vulkan)
    {
#if NCNN_VULKAN
        if (ncnn::create_gpu_instance() != 0 || ncnn::get_gpu_count() == 0)
            throw std::runtime_error("No usable Vulkan device");
        const int index = ncnn::get_default_gpu_index();
        net.set_vulkan_device(index);
        blob = std::make_unique<ncnn::VkBlobAllocator>(net.vulkan_device());
        staging = std::make_unique<ncnn::VkStagingAllocator>(net.vulkan_device());
        net.opt.blob_vkallocator = blob.get();
        net.opt.workspace_vkallocator = blob.get();
        net.opt.staging_vkallocator = staging.get();
        net.opt.use_vulkan_compute = true;
        fprintf(stderr, "Native Vulkan device=%s api=%u driver=%u profile=%d\n", ncnn::get_gpu_info(index).device_name(), ncnn::get_gpu_info(index).api_version(), ncnn::get_gpu_info(index).driver_version(), profile);
#else
        throw std::runtime_error("Native SDK lacks Vulkan");
#endif
    }
    net.opt.num_threads = threads;
    net.opt.lightmode = false;
    net.opt.use_packing_layout = packing;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = profile == 1;
    if (profile == 2)
    {
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = true;
    }
    if (profile == 3)
    {
        net.opt.use_winograd_convolution = false;
        net.opt.use_sgemm_convolution = false;
        net.opt.use_int8_inference = false;
        net.opt.use_local_pool_allocator = false;
        net.opt.openmp_blocktime = 0;
        net.opt.flush_denormals = 0;
        net.opt.use_winograd23_convolution = false;
        net.opt.use_winograd43_convolution = false;
        net.opt.use_winograd63_convolution = false;
        net.opt.use_mapped_model_loading = true;
        net.opt.kvcache_max_seqlen_hint = 32;
    }
    if (profile == 4) net.opt.lightmode = true;
    if (profile >= 5 && profile <= 7)
    {
        net.opt.use_sgemm_convolution = false;
        net.opt.use_winograd23_convolution = profile == 5;
        net.opt.use_winograd43_convolution = profile == 6;
        net.opt.use_winograd63_convolution = profile == 7;
        net.opt.flush_denormals = profile - 4;
    }
    write_u32(0x4e435053);
    for (int value : {net.opt.num_threads, (int)net.opt.lightmode, (int)net.opt.use_winograd_convolution,
                      (int)net.opt.use_sgemm_convolution, (int)net.opt.use_int8_inference, (int)net.opt.use_packing_layout,
                      (int)net.opt.use_fp16_packed, (int)net.opt.use_fp16_storage, (int)net.opt.use_fp16_arithmetic,
                      (int)net.opt.use_bf16_storage, (int)net.opt.use_local_pool_allocator, net.opt.openmp_blocktime,
                      (int)net.opt.flush_denormals, (int)net.opt.use_winograd23_convolution, (int)net.opt.use_winograd43_convolution,
                      (int)net.opt.use_winograd63_convolution, (int)net.opt.use_mapped_model_loading, net.opt.kvcache_max_seqlen_hint})
        write_u32((uint32_t)value);
    if (net.load_param(param.get()) != 0 || (profile == 3 ? net.load_model(model_path.c_str()) : net.load_model(model.get())) != 0)
        throw std::runtime_error("Native model loading failed");

    ncnn::Mat input(width, height, 3);
    if (input.empty() || net.input_names().size() != 1)
        throw std::runtime_error("Reference expects one RGB input");
    const size_t plane = (size_t)width * height;
    for (int q = 0; q < 3; q++)
    {
        float* values = input.channel(q);
        for (size_t i = 0; i < plane; i++)
            values[i] = ((int)(((q * plane + i) * 17 + 3) % 251) - 125) / 64.f;
    }
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input(net.input_names()[0], input) != 0)
        throw std::runtime_error("Native input failed");

    int probe_index = -1;
    int probe_bits = 0;
    for (const ncnn::Layer* layer : net.layers())
    {
        if ((layer->type != "Convolution" && layer->type != "Gemm" && layer->type != "LayerNorm" && layer->type != "MultiHeadAttention") || layer->tops.empty()) continue;
        probe_index = layer->tops[0];
        ncnn::Extractor probe = net.create_extractor();
        ncnn::Mat observed;
        if (probe.input(net.input_names()[0], input) != 0 || probe.extract(probe_index, observed, 1) != 0)
            throw std::runtime_error("Native precision probe failed");
        probe_bits = observed.elembits();
#if NCNN_VULKAN
        if (vulkan)
        {
            ncnn::Extractor gpu_probe = net.create_extractor();
            ncnn::VkCompute cmd(net.vulkan_device());
            ncnn::VkMat gpu_observed;
            if (gpu_probe.input(net.input_names()[0], input) != 0 || gpu_probe.extract(probe_index, gpu_observed, cmd) != 0 || gpu_observed.empty() || cmd.submit_and_wait() != 0)
                throw std::runtime_error("Native Vulkan intermediate execution failed");
            fprintf(stderr, "Native VkMat intermediate=%d storage_bits=%d\n", probe_index, gpu_observed.elembits());
        }
#endif
        break;
    }
    write_u32((uint32_t)probe_index);
    write_u32((uint32_t)probe_bits);
    write_u32((uint32_t)net.output_names().size());
    for (const char* name : net.output_names())
    {
        ncnn::Mat output;
        if (ex.extract(name, output) != 0 || output.empty() || output.elemsize != 4 || output.elempack != 1 || output.n != 1)
            throw std::runtime_error("Native extraction must produce unpacked FP32");
        const std::string output_name(name);
        write_u32((uint32_t)output_name.size());
        write_bytes(output_name.data(), output_name.size());
        for (int value : {output.dims, output.w, output.h, output.d, output.c, output.n, output.elempack})
            write_u32((uint32_t)value);
        write_u64(output.elemsize);
        write_u64(output.cstep);
        const size_t output_plane = (size_t)output.w * output.h * output.d;
        write_u64(output_plane * output.c);
        for (int q = 0; q < output.c; q++)
            write_bytes(output.channel(q), output_plane * sizeof(float));
    }
    return fflush(stdout) == 0 ? 0 : 1;
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv)
#else
int main(int argc, char** argv)
#endif
{
    try
    {
        if (argc != 8)
            throw std::runtime_error("Usage: ncnn_cpu_reference <param> <model> <width> <height> <threads> <packing> <profile>");
#ifdef _WIN32
        if (_setmode(_fileno(stdout), _O_BINARY) == -1)
            throw std::runtime_error("Binary stdout setup failed");
#endif
        return run(argv[1], argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]) != 0, std::stoi(argv[7]));
    }
    catch (const std::exception& error)
    {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
