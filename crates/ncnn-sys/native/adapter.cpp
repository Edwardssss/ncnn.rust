#include "adapter.h"
#include "cpu.h"
#include "net.h"
#include "datareader.h"
#include <cstring>
#include <new>
#include <vector>
#include <cstdlib>
#include "modelbin.h"

class OwnedModelBin : public ncnn::ModelBin
{
  public:
    OwnedModelBin(const ncnn_mat_t* mats, int count) : cursor(0)
    {
        for (int i = 0; i < count; i++)
            weights.push_back(*(ncnn::Mat*)mats[i]);
    }
    ncnn::Mat load(int, int) const
    {
        if (cursor >= weights.size())
            return ncnn::Mat();
        return weights[cursor++];
    }

  private:
    std::vector<ncnn::Mat> weights;
    mutable size_t cursor;
};
static ncnn_mat_t model_load1(ncnn_modelbin_t m, int w, int t)
{
    return (ncnn_mat_t) new ncnn::Mat(((ncnn::ModelBin*)m->pthis)->load(w, t));
}
static ncnn_mat_t model_load2(ncnn_modelbin_t m, int w, int h, int t)
{
    return (ncnn_mat_t) new ncnn::Mat(((ncnn::ModelBin*)m->pthis)->load(w, h, t));
}
static ncnn_mat_t model_load3(ncnn_modelbin_t m, int w, int h, int c, int t)
{
    return (ncnn_mat_t) new ncnn::Mat(((ncnn::ModelBin*)m->pthis)->load(w, h, c, t));
}
class CallbackModelBin : public ncnn::ModelBin
{
  public:
    CallbackModelBin(ncnn_rs_model_load_t f, void* u) : callback(f), userdata(u) {}
    ncnn::Mat invoke(int dims, int w, int h, int d, int c, int type) const
    {
        ncnn_mat_t handle = callback(userdata, dims, w, h, d, c, type);
        if (!handle)
            return ncnn::Mat();
        ncnn::Mat result = *(ncnn::Mat*)handle;
        ncnn_mat_destroy(handle);
        return result;
    }
    ncnn::Mat load(int w, int t) const
    {
        return invoke(1, w, 1, 1, 1, t);
    }
    ncnn::Mat load(int w, int h, int t) const
    {
        return invoke(2, w, h, 1, 1, t);
    }
    ncnn::Mat load(int w, int h, int c, int t) const
    {
        return invoke(3, w, h, 1, c, t);
    }
    ncnn::Mat load(int w, int h, int d, int c, int t) const
    {
        return invoke(4, w, h, d, c, t);
    }

  private:
    ncnn_rs_model_load_t callback;
    void* userdata;
};
extern "C" ncnn_modelbin_t ncnn_rs_modelbin_callbacks(ncnn_rs_model_load_t load, void* userdata)
{
    if (!load)
        return 0;
    ncnn_modelbin_t result = (ncnn_modelbin_t)calloc(1, sizeof(*result));
    if (!result)
        return 0;
    try
    {
        result->pthis = new CallbackModelBin(load, userdata);
    }
    catch (...)
    {
        free(result);
        return 0;
    }
    result->load_1d = model_load1;
    result->load_2d = model_load2;
    result->load_3d = model_load3;
    return result;
}
extern "C" ncnn_modelbin_t ncnn_rs_modelbin_from_mats(const ncnn_mat_t* mats, int count)
{
    ncnn_modelbin_t result = (ncnn_modelbin_t)calloc(1, sizeof(*result));
    if (!result)
        return 0;
    try
    {
        result->pthis = new OwnedModelBin(mats, count);
    }
    catch (...)
    {
        free(result);
        return 0;
    }
    result->load_1d = model_load1;
    result->load_2d = model_load2;
    result->load_3d = model_load3;
    return result;
}

#define NCNN_RS_TRY try
#define NCNN_RS_CATCH             \
    catch (const std::bad_alloc&) \
    {                             \
        return -100;              \
    }                             \
    catch (...)                   \
    {                             \
        return -101;              \
    }

extern "C" int ncnn_rs_layer_load_model(ncnn_layer_t layer, ncnn_modelbin_t model)
{
    NCNN_RS_TRY
    {
        return ((ncnn::Layer*)layer->pthis)->load_model(*(ncnn::ModelBin*)model->pthis);
    }
    NCNN_RS_CATCH
}

extern "C" int ncnn_rs_cpu_count(void)
{
    NCNN_RS_TRY
    {
        return ncnn::get_cpu_count();
    }
    NCNN_RS_CATCH
}

static int option_value(ncnn::Option& opt, int key, int& value, bool write)
{
#define FIELD(k, name)        \
    case k:                   \
        if (write)            \
            opt.name = value; \
        else                  \
            value = opt.name; \
        break
    switch (key)
    {
        FIELD(0, num_threads);
        FIELD(1, lightmode);
        FIELD(2, use_winograd_convolution);
        FIELD(3, use_sgemm_convolution);
        FIELD(4, use_int8_inference);
        FIELD(5, use_packing_layout);
        FIELD(6, use_fp16_packed);
        FIELD(7, use_fp16_storage);
        FIELD(8, use_fp16_arithmetic);
        FIELD(9, use_bf16_storage);
        FIELD(10, use_local_pool_allocator);
        FIELD(11, openmp_blocktime);
        FIELD(12, flush_denormals);
        FIELD(13, use_winograd23_convolution);
        FIELD(14, use_winograd43_convolution);
        FIELD(15, use_winograd63_convolution);
        FIELD(16, use_mapped_model_loading);
        FIELD(17, kvcache_max_seqlen_hint);
        FIELD(18, use_subgroup_ops);
        FIELD(19, use_int8_packed);
        FIELD(20, use_int8_storage);
        FIELD(21, use_int8_arithmetic);
        FIELD(22, use_bf16_packed);
        FIELD(23, use_tensor_storage);
        FIELD(24, use_weights_in_host_memory);
        FIELD(25, use_shader_local_memory);
        FIELD(26, use_cooperative_matrix);
        FIELD(27, use_a53_a55_optimized_kernel);
        FIELD(28, use_fp16_uniform);
        FIELD(29, use_int8_uniform);
        FIELD(30, use_int16_packed);
        FIELD(31, use_int16_storage);
    default:
        return -1;
    }
#undef FIELD
    return 0;
}
extern "C" int ncnn_rs_option_set(ncnn_option_t opt, int key, int value)
{
    return option_value(*(ncnn::Option*)opt, key, value, true);
}
extern "C" int ncnn_rs_option_get(ncnn_option_t opt, int key, int* value)
{
    return option_value(*(ncnn::Option*)opt, key, *value, false);
}

extern "C" int ncnn_rs_mat_copy_f32(const float* data, int dims, const int* s, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        ncnn::Mat m;
        if (dims == 1)
            m.create(s[0]);
        if (dims == 2)
            m.create(s[0], s[1]);
        if (dims == 3)
            m.create(s[0], s[1], s[2]);
        if (dims == 4)
            m.create(s[0], s[1], s[2], s[3]);
        if (m.empty())
            return -100;
        const size_t plane = (size_t)m.w * m.h * m.d;
        for (int q = 0; q < m.c; q++)
            memcpy(m.channel(q), data + q * plane, plane * sizeof(float));
        *out = (ncnn_mat_t) new ncnn::Mat(m);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_mat_clone(ncnn_mat_t mat, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        ncnn::Mat copy = ((ncnn::Mat*)mat)->clone();
        if (copy.empty())
            return -100;
        *out = (ncnn_mat_t) new ncnn::Mat(copy);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_mat_normalized(ncnn_mat_t mat, const float* mean, const float* norm, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        ncnn::Mat copy = ((ncnn::Mat*)mat)->clone();
        if (copy.empty())
            return -100;
        // Scale interprets dimensions below three as element/row coefficients.
        // A header-only channel view preserves the public per-channel contract.
        ncnn::Mat channels = copy;
        if (channels.dims < 3)
            channels.dims = 3;
        channels.substract_mean_normalize(mean, norm);
        *out = (ncnn_mat_t) new ncnn::Mat(copy);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_mat_read_f32(ncnn_mat_t mat, float* out, size_t count)
{
    if (!mat)
        return -1;
    const ncnn::Mat& m = *(ncnn::Mat*)mat;
    if (m.elemsize != 4 || m.elempack != 1 || m.n <= 0 || m.w <= 0 || m.h <= 0 || m.d <= 0 || m.c <= 0)
        return -1;
    size_t logical = 1;
    for (int extent : {m.w, m.h, m.d, m.c, m.n})
    {
        if ((size_t)extent > SIZE_MAX / logical)
            return -1;
        logical *= (size_t)extent;
    }
    if (count != logical || count > SIZE_MAX / sizeof(float) || !m.data || !out)
        return -1;
    const size_t plane = logical / (size_t)m.c / (size_t)m.n;
    if (m.cstep < plane || m.cstep > SIZE_MAX / (size_t)m.c || m.nstep < m.cstep * (size_t)m.c || m.nstep > SIZE_MAX / (size_t)m.n / sizeof(float))
        return -1;
    if (m.n == 1 && m.cstep == plane)
    {
        // Unpadded channels form one contiguous run.
        memcpy(out, m.data, logical * sizeof(float));
        return 0;
    }
    for (int b = 0; b < m.n; b++)
        for (int q = 0; q < m.c; q++)
            memcpy(out + ((size_t)b * m.c + q) * plane,
                   (const float*)m.data + (size_t)b * m.nstep + (size_t)q * m.cstep,
                   plane * sizeof(float));
    return 0;
}
static size_t rs_element_bytes(int kind)
{
    switch (kind)
    {
    case 0:
    case 6:
        return 4;
    case 1:
    case 2:
    case 5:
        return 2;
    case 3:
    case 4:
        return 1;
    default:
        return 0;
    }
}
static float rs_element_to_f32(const unsigned char* p, int kind)
{
    switch (kind)
    {
    case 0:
    {
        float v;
        memcpy(&v, p, sizeof(v));
        return v;
    }
    case 1:
    {
        unsigned short v;
        memcpy(&v, p, sizeof(v));
        return ncnn::float16_to_float32(v);
    }
    case 2:
    {
        unsigned short v;
        memcpy(&v, p, sizeof(v));
        return ncnn::bfloat16_to_float32(v);
    }
    case 3:
    {
        signed char v;
        memcpy(&v, p, sizeof(v));
        return (float)v;
    }
    case 4:
    {
        unsigned char v;
        memcpy(&v, p, sizeof(v));
        return (float)v;
    }
    case 5:
    {
        short v;
        memcpy(&v, p, sizeof(v));
        return (float)v;
    }
    case 6:
    {
        int v;
        memcpy(&v, p, sizeof(v));
        return (float)v;
    }
    default:
        return 0.f;
    }
}
extern "C" int ncnn_rs_mat_read_typed(ncnn_mat_t mat, int kind, float* out, size_t count)
{
    if (!mat || !out)
        return -1;
    const ncnn::Mat& m = *(ncnn::Mat*)mat;
    const size_t element_bytes = rs_element_bytes(kind);
    if (!element_bytes || m.elempack <= 0 || m.elemsize != element_bytes * (size_t)m.elempack)
        return -1;
    if (m.n <= 0 || m.w <= 0 || m.h <= 0 || m.d <= 0 || m.c <= 0 || !m.data)
        return -1;
    size_t pixels = 1;
    for (int extent : {m.w, m.h, m.d})
    {
        if ((size_t)extent > SIZE_MAX / pixels)
            return -1;
        pixels *= (size_t)extent;
    }
    const size_t per_channel = pixels * (size_t)m.elempack;
    if ((size_t)m.c > SIZE_MAX / per_channel || (size_t)m.n > SIZE_MAX / ((size_t)m.c * per_channel))
        return -1;
    if (count != (size_t)m.c * per_channel * (size_t)m.n)
        return -1;
    for (int b = 0; b < m.n; b++)
    {
        for (int q = 0; q < m.c; q++)
        {
            const unsigned char* base = (const unsigned char*)m.data + ((size_t)b * m.nstep + (size_t)q * m.cstep) * m.elemsize;
            float* dst = out + ((size_t)b * m.c + q) * per_channel;
            for (size_t i = 0; i < per_channel; i++)
                dst[i] = rs_element_to_f32(base + i * element_bytes, kind);
        }
    }
    return 0;
}
// Native extraction converts packed, FP16, BF16 and INT8 blobs unless the caller passes a
// non-zero type, which keeps the storage a layer wrote so Rust can read it by element kind.
extern "C" int ncnn_rs_extractor_extract_stored(ncnn_extractor_t ex, const char* name, int type, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        if (!ex || !name || !out)
            return -1;
        ncnn::Mat feat;
        int ret = ((ncnn::Extractor*)ex)->extract(name, feat, type);
        if (ret != 0 || feat.empty())
            return ret != 0 ? ret : -100;
        *out = (ncnn_mat_t) new ncnn::Mat(feat);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_extractor_extract_stored_index(ncnn_extractor_t ex, int index, int type, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        if (!ex || !out)
            return -1;
        ncnn::Mat feat;
        int ret = ((ncnn::Extractor*)ex)->extract(index, feat, type);
        if (ret != 0 || feat.empty())
            return ret != 0 ? ret : -100;
        *out = (ncnn_mat_t) new ncnn::Mat(feat);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_mat_reshape(ncnn_mat_t mat, int dims, const int* s, int independent, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        const ncnn::Mat& m = *(ncnn::Mat*)mat;
        ncnn::Mat result;
        if (dims == 1)
            result = m.reshape(s[0]);
        if (dims == 2)
            result = m.reshape(s[0], s[1]);
        if (dims == 3)
            result = m.reshape(s[0], s[1], s[2]);
        if (dims == 4)
            result = m.reshape(s[0], s[1], s[2], s[3]);
        if (result.empty())
            return -1;
        // Native padded reshapes already allocate with the default allocator.
        if (independent && result.data == m.data)
            result = result.clone();
        if (result.empty())
            return -100;
        *out = (ncnn_mat_t) new ncnn::Mat(result);
        return 0;
    }
    NCNN_RS_CATCH
}

class BoundedReader : public ncnn::DataReader
{
  public:
    BoundedReader(const unsigned char* p, size_t n) : data(p), remaining(n) {}
    size_t read(void* out, size_t size) const
    {
        if (size > remaining)
            return 0;
        if (size)
            memcpy(out, data, size);
        data += size;
        remaining -= size;
        return size;
    }

  private:
    mutable const unsigned char* data;
    mutable size_t remaining;
};
class OwnedReader : public ncnn::DataReader
{
  public:
    OwnedReader(const unsigned char* data, size_t size) : bytes(data, data + size), cursor(0) {}
    size_t read(void* out, size_t size) const
    {
        if (size > bytes.size() - cursor)
            return 0;
        if (size)
            memcpy(out, bytes.data() + cursor, size);
        cursor += size;
        return size;
    }

  private:
    std::vector<unsigned char> bytes;
    mutable size_t cursor;
};
static size_t reader_read(ncnn_datareader_t dr, void* out, size_t size)
{
    return ((ncnn::DataReader*)dr->pthis)->read(out, size);
}
static int reader_scan(ncnn_datareader_t, const char*, void*)
{
    return 0;
}
extern "C" ncnn_datareader_t ncnn_rs_reader_create(const unsigned char* data, size_t size)
{
    ncnn_datareader_t result = (ncnn_datareader_t)calloc(1, sizeof(*result));
    if (!result)
        return 0;
    try
    {
        result->pthis = new OwnedReader(data, size);
    }
    catch (...)
    {
        free(result);
        return 0;
    }
    result->read = reader_read;
    result->scan = reader_scan;
    return result;
}

class MemoryReader : public ncnn::DataReader
{
  public:
    MemoryReader(const unsigned char* data, size_t size) : storage(data, data + size), cursor(storage.data()), reader(cursor, size) {}
    int scan(const char* format, void* out) const
    {
        return reader.scan(format, out);
    }
    size_t read(void* out, size_t size) const
    {
        return reader.read(out, size);
    }
    size_t reference(size_t size, const void** out) const
    {
        return reader.reference(size, out);
    }

  private:
    std::vector<unsigned char> storage;
    mutable const unsigned char* cursor;
    ncnn::DataReaderFromMemory reader;
};
static int memory_scan(ncnn_datareader_t dr, const char* format, void* out)
{
    return ((ncnn::DataReader*)dr->pthis)->scan(format, out);
}
extern "C" ncnn_datareader_t ncnn_rs_reader_memory(const unsigned char* data, size_t size)
{
    ncnn_datareader_t result = (ncnn_datareader_t)calloc(1, sizeof(*result));
    if (!result)
        return 0;
    try
    {
        result->pthis = new MemoryReader(data, size);
    }
    catch (...)
    {
        free(result);
        return 0;
    }
    result->read = reader_read;
    result->scan = memory_scan;
    return result;
}
extern "C" size_t ncnn_rs_reader_reference(ncnn_datareader_t reader, size_t size, const void** out)
{
    return ((ncnn::DataReader*)reader->pthis)->reference(size, out);
}

extern "C" int ncnn_rs_load_bytes(ncnn_net_t net, const unsigned char* data, size_t size, int model)
{
    NCNN_RS_TRY
    {
        BoundedReader reader(data, size);
        ncnn::Net& n = *(ncnn::Net*)net->pthis;
        return model ? n.load_model(reader) : n.load_param_bin(reader);
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_reference_add(const float* a, const float* b, int count, float* out)
{
    NCNN_RS_TRY
    {
        ncnn::Net net;
        net.opt.num_threads = 1;
        int ret = net.load_param_mem("7767517\n3 3\nInput a 0 1 a\nInput b 0 1 b\nBinaryOp sum 2 1 a b sum 0=0\n");
        if (ret)
            return ret;
        BoundedReader reader((const unsigned char*)"", 0);
        if ((ret = net.load_model(reader)))
            return ret;
        ncnn::Mat x(count), y(count), z;
        memcpy(x.data, a, count * sizeof(float));
        memcpy(y.data, b, count * sizeof(float));
        ncnn::Extractor ex = net.create_extractor();
        if ((ret = ex.input("a", x)) || (ret = ex.input("b", y)) || (ret = ex.extract("sum", z)))
            return ret;
        memcpy(out, z.data, count * sizeof(float));
        return 0;
    }
    NCNN_RS_CATCH
}
