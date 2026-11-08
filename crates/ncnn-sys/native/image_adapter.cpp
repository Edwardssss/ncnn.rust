#include "image_adapter.h"
#include "mat.h"
extern "C" void ncnn_rs_image_yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    ncnn::yuv420sp2rgb(yuv420sp, w, h, rgb);
}
extern "C" void ncnn_rs_image_yuv420sp2rgb_nv12(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    ncnn::yuv420sp2rgb_nv12(yuv420sp, w, h, rgb);
}
extern "C" void ncnn_rs_image_yuv420sp2rgb_half(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    ncnn::yuv420sp2rgb_half(yuv420sp, w, h, rgb);
}
extern "C" void ncnn_rs_image_resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    ncnn::resize_bilinear_c1(src, srcw, srch, dst, w, h);
}
extern "C" void ncnn_rs_image_resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    ncnn::resize_bilinear_c2(src, srcw, srch, dst, w, h);
}
extern "C" void ncnn_rs_image_resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    ncnn::resize_bilinear_c3(src, srcw, srch, dst, w, h);
}
extern "C" void ncnn_rs_image_resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    ncnn::resize_bilinear_c4(src, srcw, srch, dst, w, h);
}
extern "C" void ncnn_rs_image_resize_bilinear_c1_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    ncnn::resize_bilinear_c1(src, srcw, srch, srcstride, dst, w, h, stride);
}
extern "C" void ncnn_rs_image_resize_bilinear_c2_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    ncnn::resize_bilinear_c2(src, srcw, srch, srcstride, dst, w, h, stride);
}
extern "C" void ncnn_rs_image_resize_bilinear_c3_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    ncnn::resize_bilinear_c3(src, srcw, srch, srcstride, dst, w, h, stride);
}
extern "C" void ncnn_rs_image_resize_bilinear_c4_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    ncnn::resize_bilinear_c4(src, srcw, srch, srcstride, dst, w, h, stride);
}
extern "C" void ncnn_rs_image_resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    ncnn::resize_bilinear_yuv420sp(src, srcw, srch, dst, w, h);
}
extern "C" void ncnn_rs_image_kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    ncnn::kanna_rotate_c1(src, srcw, srch, dst, w, h, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    ncnn::kanna_rotate_c2(src, srcw, srch, dst, w, h, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    ncnn::kanna_rotate_c3(src, srcw, srch, dst, w, h, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    ncnn::kanna_rotate_c4(src, srcw, srch, dst, w, h, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_c1_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    ncnn::kanna_rotate_c1(src, srcw, srch, srcstride, dst, w, h, stride, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_c2_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    ncnn::kanna_rotate_c2(src, srcw, srch, srcstride, dst, w, h, stride, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_c3_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    ncnn::kanna_rotate_c3(src, srcw, srch, srcstride, dst, w, h, stride, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_c4_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    ncnn::kanna_rotate_c4(src, srcw, srch, srcstride, dst, w, h, stride, type);
}
extern "C" void ncnn_rs_image_kanna_rotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    ncnn::kanna_rotate_yuv420sp(src, srcw, srch, dst, w, h, type);
}
extern "C" void ncnn_rs_image_get_rotation_matrix(float angle, float scale, float dx, float dy, float* tm)
{
    ncnn::get_rotation_matrix(angle, scale, dx, dy, tm);
}
extern "C" void ncnn_rs_image_get_affine_transform(const float* points_from, const float* points_to, int num_point, float* tm)
{
    ncnn::get_affine_transform(points_from, points_to, num_point, tm);
}
extern "C" void ncnn_rs_image_invert_affine_transform(const float* tm, float* tm_inv)
{
    ncnn::invert_affine_transform(tm, tm_inv);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c1(src, srcw, srch, dst, w, h, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c2(src, srcw, srch, dst, w, h, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c3(src, srcw, srch, dst, w, h, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c4(src, srcw, srch, dst, w, h, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c1_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c1(src, srcw, srch, srcstride, dst, w, h, stride, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c2_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c2(src, srcw, srch, srcstride, dst, w, h, stride, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c3_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c3(src, srcw, srch, srcstride, dst, w, h, stride, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_c4_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_c4(src, srcw, srch, srcstride, dst, w, h, stride, tm, type, v);
}
extern "C" void ncnn_rs_image_warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    ncnn::warpaffine_bilinear_yuv420sp(src, srcw, srch, dst, w, h, tm, type, v);
}
extern "C" void ncnn_rs_image_draw_rectangle_c1(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c1(pixels, w, h, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_c2(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c2(pixels, w, h, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_c3(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c3(pixels, w, h, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_c4(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c4(pixels, w, h, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_c1_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c1(pixels, w, h, stride, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_c2_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c2(pixels, w, h, stride, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_c3_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c3(pixels, w, h, stride, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_c4_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_c4(pixels, w, h, stride, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_rectangle_yuv420sp(unsigned char* yuv420sp, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness)
{
    ncnn::draw_rectangle_yuv420sp(yuv420sp, w, h, rx, ry, rw, rh, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c1(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c1(pixels, w, h, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c2(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c2(pixels, w, h, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c3(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c3(pixels, w, h, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c4(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c4(pixels, w, h, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c1_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c1(pixels, w, h, stride, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c2_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c2(pixels, w, h, stride, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c3_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c3(pixels, w, h, stride, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_c4_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_c4(pixels, w, h, stride, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_circle_yuv420sp(unsigned char* yuv420sp, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness)
{
    ncnn::draw_circle_yuv420sp(yuv420sp, w, h, cx, cy, radius, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c1(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c1(pixels, w, h, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c2(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c2(pixels, w, h, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c3(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c3(pixels, w, h, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c4(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c4(pixels, w, h, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c1_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c1(pixels, w, h, stride, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c2_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c2(pixels, w, h, stride, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c3_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c3(pixels, w, h, stride, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_c4_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_c4(pixels, w, h, stride, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_draw_line_yuv420sp(unsigned char* yuv420sp, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness)
{
    ncnn::draw_line_yuv420sp(yuv420sp, w, h, x0, y0, x1, y1, color, thickness);
}
extern "C" void ncnn_rs_image_get_text_drawing_size(const char* text, int fontpixelsize, int* w, int* h)
{
    ncnn::get_text_drawing_size(text, fontpixelsize, w, h);
}
extern "C" void ncnn_rs_image_draw_text_c1(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c1(pixels, w, h, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_c2(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c2(pixels, w, h, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_c3(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c3(pixels, w, h, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_c4(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c4(pixels, w, h, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_c1_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c1(pixels, w, h, stride, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_c2_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c2(pixels, w, h, stride, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_c3_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c3(pixels, w, h, stride, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_c4_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_c4(pixels, w, h, stride, text, x, y, fontpixelsize, color);
}
extern "C" void ncnn_rs_image_draw_text_yuv420sp(unsigned char* yuv420sp, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color)
{
    ncnn::draw_text_yuv420sp(yuv420sp, w, h, text, x, y, fontpixelsize, color);
}
