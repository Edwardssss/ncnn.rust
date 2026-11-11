#include "adapter.h"
#include "cpu.h"
#include "net.h"
#include "datareader.h"
#include <cstring>
#include <new>
#include <vector>
#include <cstdlib>
#include "modelbin.h"
#if _WIN32
#include <windows.h>
#endif

// Native loaders and readers take narrow paths on every platform, so a UTF-8 path is widened on
// Windows before it reaches them.
static FILE* rs_fopen_utf8(const char* path)
{
#if _WIN32
    const int len = MultiByteToWideChar(CP_UTF8, 0, path, -1, 0, 0);
    if (len <= 0)
        return 0;
    std::vector<wchar_t> wide((size_t)len);
    if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wide.data(), len) <= 0)
        return 0;
    return _wfopen(wide.data(), L"rb");
#else
    return fopen(path, "rb");
#endif
}

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
        if (!mat || !out)
            return -1;
        const ncnn::Mat& m = *(const ncnn::Mat*)mat;
        if (m.elemsize != 4 || m.elempack != 1 || m.n != 1 || m.c <= 0 || m.dims < 1)
            return -1;
        ncnn::Mat dst;
        if (m.dims == 1)
            dst.create(m.w, (size_t)4u, 1);
        if (m.dims == 2)
            dst.create(m.w, m.h, (size_t)4u, 1);
        if (m.dims == 3)
            dst.create(m.w, m.h, m.c, (size_t)4u, 1);
        if (m.dims == 4)
            dst.create(m.w, m.h, m.d, m.c, (size_t)4u, 1);
        if (dst.empty())
            return -100;
        const size_t plane = (size_t)m.w * m.h * m.d;
        // One pass over the fresh storage instead of a copy followed by a native layer, which is
        // the same scale and bias the native Scale path applies.
        for (int q = 0; q < m.c; q++)
        {
            const float* src = m.channel(q);
            float* dst_ptr = dst.channel(q);
            if (mean && norm)
            {
                const float scale = norm[q];
                const float offset = -mean[q] * norm[q];
                for (size_t i = 0; i < plane; i++)
                    dst_ptr[i] = src[i] * scale + offset;
            }
            else if (norm)
            {
                const float scale = norm[q];
                for (size_t i = 0; i < plane; i++)
                    dst_ptr[i] = src[i] * scale;
            }
            else if (mean)
            {
                const float offset = -mean[q];
                for (size_t i = 0; i < plane; i++)
                    dst_ptr[i] = src[i] + offset;
            }
            else
            {
                memcpy(dst_ptr, src, plane * sizeof(float));
            }
        }
        *out = (ncnn_mat_t) new ncnn::Mat(dst);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_mat_normalize_in_place(ncnn_mat_t mat, const float* mean, const float* norm)
{
    NCNN_RS_TRY
    {
        if (!mat)
            return -1;
        ncnn::Mat& m = *(ncnn::Mat*)mat;
        if (m.elemsize != 4 || m.elempack != 1 || m.n != 1 || m.c <= 0 || m.dims < 1)
            return -1;
        const size_t plane = (size_t)m.w * m.h * m.d;
        for (int q = 0; q < m.c; q++)
        {
            float* data = m.channel(q);
            if (mean && norm)
            {
                const float scale = norm[q];
                const float offset = -mean[q] * norm[q];
                for (size_t i = 0; i < plane; i++)
                    data[i] = data[i] * scale + offset;
            }
            else if (norm)
            {
                const float scale = norm[q];
                for (size_t i = 0; i < plane; i++)
                    data[i] *= scale;
            }
            else if (mean)
            {
                const float offset = -mean[q];
                for (size_t i = 0; i < plane; i++)
                    data[i] += offset;
            }
        }
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
// The counterpart of the typed reader: one element width per tensor, filled plane by plane so
// the channel padding ncnn reserves stays untouched.
extern "C" int ncnn_rs_mat_copy_typed(int kind, const unsigned char* data, size_t size, int dims, const int* s, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        const size_t element_bytes = rs_element_bytes(kind);
        if (!element_bytes || !out || (!data && size != 0) || dims < 1 || dims > 4)
            return -1;

        ncnn::Mat m;
        if (dims == 1)
            m.create(s[0], element_bytes, 1);
        if (dims == 2)
            m.create(s[0], s[1], element_bytes, 1);
        if (dims == 3)
            m.create(s[0], s[1], s[2], element_bytes, 1);
        if (dims == 4)
            m.create(s[0], s[1], s[2], s[3], element_bytes, 1);
        if (m.empty())
            return -100;

        const size_t plane_elements = (size_t)m.w * m.h * m.d;
        const size_t plane_bytes = plane_elements * element_bytes;
        if (plane_bytes == 0 || (size_t)m.c > SIZE_MAX / plane_bytes)
            return -1;
        if ((size_t)m.c * plane_bytes != size)
            return -1;

        for (int q = 0; q < m.c; q++)
            memcpy(m.channel(q), data + q * plane_bytes, plane_bytes);
        *out = (ncnn_mat_t) new ncnn::Mat(m);
        return 0;
    }
    NCNN_RS_CATCH
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
        if (ret != 0)
            return ret;
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
        if (ret != 0)
            return ret;
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

// Reads a file through stdio and closes it with the reader, so a stream larger than the addressable
// buffer never needs a Rust-side copy. ncnn's stdio reader supplies the scan and read semantics.
class FileReader : public ncnn::DataReaderFromStdio
{
  public:
    explicit FileReader(FILE* f) : ncnn::DataReaderFromStdio(f), fp(f) {}
    ~FileReader()
    {
        if (fp)
            fclose(fp);
    }

  private:
    FILE* fp;
    FileReader(const FileReader&);
    FileReader& operator=(const FileReader&);
};
extern "C" ncnn_datareader_t ncnn_rs_reader_file(const char* path)
{
    if (!path)
        return 0;
    FILE* fp = rs_fopen_utf8(path);
    if (!fp)
        return 0;
    ncnn_datareader_t result = (ncnn_datareader_t)calloc(1, sizeof(*result));
    if (!result)
    {
        fclose(fp);
        return 0;
    }
    try
    {
        result->pthis = new FileReader(fp);
    }
    catch (...)
    {
        fclose(fp);
        free(result);
        return 0;
    }
    result->read = reader_read;
    result->scan = memory_scan;
    return result;
}

extern "C" int ncnn_rs_net_load_file(ncnn_net_t net, const char* path, int model, int mapped)
{
    NCNN_RS_TRY
    {
        if (!net || !path)
            return -1;
        ncnn::Net& n = *(ncnn::Net*)net->pthis;
        const bool previous = n.opt.use_mapped_model_loading;
        n.opt.use_mapped_model_loading = mapped != 0;
        int ret = -1;
#if _WIN32
        const int len = MultiByteToWideChar(CP_UTF8, 0, path, -1, 0, 0);
        if (len > 0)
        {
            std::vector<wchar_t> wide((size_t)len);
            if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wide.data(), len) > 0)
                ret = model ? n.load_model(wide.data()) : n.load_param(wide.data());
        }
#else
        ret = model ? n.load_model(path) : n.load_param(path);
#endif
        n.opt.use_mapped_model_loading = previous;
        return ret;
    }
    NCNN_RS_CATCH
}

extern "C" int ncnn_rs_mat_copy_f32_batch(const float* data, int dims, const int* s, int batch, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        if (!data || !s || !out || batch < 1)
            return -1;
        ncnn::Mat m;
        if (dims == 1)
            m.create(s[0], (size_t)4u, 1, batch);
        if (dims == 2)
            m.create(s[0], s[1], (size_t)4u, 1, batch);
        if (dims == 3)
            m.create(s[0], s[1], s[2], (size_t)4u, 1, batch);
        if (dims == 4)
            m.create(s[0], s[1], s[2], s[3], (size_t)4u, 1, batch);
        if (m.empty())
            return -100;
        const size_t plane = (size_t)m.w * m.h * m.d;
        for (int b = 0; b < m.n; b++)
            for (int q = 0; q < m.c; q++)
                memcpy((float*)m.data + (size_t)b * m.nstep + (size_t)q * m.cstep, data + ((size_t)b * m.c + q) * plane, plane * sizeof(float));
        *out = (ncnn_mat_t) new ncnn::Mat(m);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_mat_pack(ncnn_mat_t mat, int elempack, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        if (!mat || !out || elempack < 1)
            return -1;
        const ncnn::Mat& m = *(const ncnn::Mat*)mat;
        if (m.elempack == elempack)
        {
            ncnn::Mat copy = m.clone();
            if (copy.empty())
                return -100;
            *out = (ncnn_mat_t) new ncnn::Mat(copy);
            return 0;
        }
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        ncnn::Mat packed;
        convert_packing(m, packed, elempack, opt);
        if (packed.empty())
            return -100;
        *out = (ncnn_mat_t) new ncnn::Mat(packed);
        return 0;
    }
    NCNN_RS_CATCH
}
extern "C" int ncnn_rs_mat_flatten(ncnn_mat_t mat, ncnn_mat_t* out)
{
    NCNN_RS_TRY
    {
        if (!mat || !out)
            return -1;
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        ncnn::Mat flat;
        ncnn::flatten(*(const ncnn::Mat*)mat, flat, opt);
        if (flat.empty())
            return -100;
        *out = (ncnn_mat_t) new ncnn::Mat(flat);
        return 0;
    }
    NCNN_RS_CATCH
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

// Graph description. The name pointers refer to strings owned by the network, so they stay valid
// until the network changes; callers copy what they need.
extern "C" int ncnn_rs_net_layer_info(ncnn_net_t net, int index, const char** name, const char** type,
                                      int* typeindex, int* bottom_count, int* top_count, int* flags)
{
    NCNN_RS_TRY
    {
        if (!net || !name || !type || !typeindex || !bottom_count || !top_count || !flags)
            return -1;
        const std::vector<ncnn::Layer*>& layers = ((ncnn::Net*)net->pthis)->layers();
        if (index < 0 || (size_t)index >= layers.size())
            return -1;
        const ncnn::Layer* layer = layers[(size_t)index];
        if (!layer)
            return -1;
        *name = layer->name.c_str();
        *type = layer->type.c_str();
        *typeindex = layer->typeindex;
        *bottom_count = (int)layer->bottoms.size();
        *top_count = (int)layer->tops.size();
        int bits = 0;
        if (layer->support_vulkan)
            bits |= 1 << 0;
        if (layer->support_vulkan_packing)
            bits |= 1 << 1;
        if (layer->support_packing)
            bits |= 1 << 2;
        if (layer->support_fp16_storage)
            bits |= 1 << 3;
        if (layer->support_bf16_storage)
            bits |= 1 << 4;
        if (layer->one_blob_only)
            bits |= 1 << 5;
        if (layer->support_inplace)
            bits |= 1 << 6;
        if (layer->support_any_packing)
            bits |= 1 << 7;
        if (layer->support_vulkan_any_packing)
            bits |= 1 << 8;
        if (layer->support_int8_storage)
            bits |= 1 << 9;
        if (layer->support_tensor_storage)
            bits |= 1 << 10;
        if (layer->support_batch)
            bits |= 1 << 11;
        *flags = bits;
        return 0;
    }
    NCNN_RS_CATCH
}

extern "C" int ncnn_rs_net_blob_count(ncnn_net_t net)
{
    NCNN_RS_TRY
    {
        if (!net)
            return -1;
        return (int)((ncnn::Net*)net->pthis)->blobs().size();
    }
    NCNN_RS_CATCH
}

extern "C" int ncnn_rs_net_blob_info(ncnn_net_t net, int index, const char** name, int* dims,
                                     int* w, int* h, int* d, int* c, int* producer, int* consumer)
{
    NCNN_RS_TRY
    {
        if (!net || !name || !dims || !w || !h || !d || !c || !producer || !consumer)
            return -1;
        const std::vector<ncnn::Blob>& blobs = ((ncnn::Net*)net->pthis)->blobs();
        if (index < 0 || (size_t)index >= blobs.size())
            return -1;
        const ncnn::Blob& blob = blobs[(size_t)index];
        *name = blob.name.c_str();
        *dims = blob.shape.dims;
        *w = blob.shape.dims ? blob.shape.w : 0;
        *h = blob.shape.dims ? blob.shape.h : 0;
        *d = blob.shape.dims ? blob.shape.d : 0;
        *c = blob.shape.dims ? blob.shape.c : 0;
        *producer = blob.producer;
        *consumer = blob.consumer;
        return 0;
    }
    NCNN_RS_CATCH
}

extern "C" int ncnn_rs_net_layer_shape(ncnn_net_t net, int index, int top, int blob_index,
                                       int* dims, int* w, int* h, int* d, int* c)
{
    NCNN_RS_TRY
    {
        if (!net || !dims || !w || !h || !d || !c)
            return -1;
        const std::vector<ncnn::Layer*>& layers = ((ncnn::Net*)net->pthis)->layers();
        if (index < 0 || (size_t)index >= layers.size())
            return -1;
        const ncnn::Layer* layer = layers[(size_t)index];
        if (!layer)
            return -1;
        const std::vector<ncnn::Mat>& shapes = top ? layer->top_shapes : layer->bottom_shapes;
        if (blob_index < 0)
            return -1;
        if ((size_t)blob_index >= shapes.size())
        {
            // A layer that recorded no shape reports zero extents rather than an error.
            *dims = 0;
            *w = 0;
            *h = 0;
            *d = 0;
            *c = 0;
            return 0;
        }
        const ncnn::Mat& shape = shapes[(size_t)blob_index];
        *dims = shape.dims;
        *w = shape.dims ? shape.w : 0;
        *h = shape.dims ? shape.h : 0;
        *d = shape.dims ? shape.d : 0;
        *c = shape.dims ? shape.c : 0;
        return 0;
    }
    NCNN_RS_CATCH
}
