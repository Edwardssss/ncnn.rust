#ifndef NCNN_RUST_IMAGE_H
#define NCNN_RUST_IMAGE_H
#ifdef __cplusplus
extern "C"
{
#endif
    void ncnn_rs_image_yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
    void ncnn_rs_image_yuv420sp2rgb_nv12(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
    void ncnn_rs_image_yuv420sp2rgb_half(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb);
    void ncnn_rs_image_resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
    void ncnn_rs_image_resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
    void ncnn_rs_image_resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
    void ncnn_rs_image_resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
    void ncnn_rs_image_resize_bilinear_c1_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
    void ncnn_rs_image_resize_bilinear_c2_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
    void ncnn_rs_image_resize_bilinear_c3_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
    void ncnn_rs_image_resize_bilinear_c4_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride);
    void ncnn_rs_image_resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h);
    void ncnn_rs_image_kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
    void ncnn_rs_image_kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
    void ncnn_rs_image_kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
    void ncnn_rs_image_kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
    void ncnn_rs_image_kanna_rotate_c1_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
    void ncnn_rs_image_kanna_rotate_c2_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
    void ncnn_rs_image_kanna_rotate_c3_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
    void ncnn_rs_image_kanna_rotate_c4_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type);
    void ncnn_rs_image_kanna_rotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type);
    void ncnn_rs_image_get_rotation_matrix(float angle, float scale, float dx, float dy, float* tm);
    void ncnn_rs_image_get_affine_transform(const float* points_from, const float* points_to, int num_point, float* tm);
    void ncnn_rs_image_invert_affine_transform(const float* tm, float* tm_inv);
    void ncnn_rs_image_warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_c1_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_c2_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_c3_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_c4_strided(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v);
    void ncnn_rs_image_draw_rectangle_c1(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_c2(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_c3(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_c4(unsigned char* pixels, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_c1_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_c2_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_c3_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_c4_strided(unsigned char* pixels, int w, int h, int stride, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_rectangle_yuv420sp(unsigned char* yuv420sp, int w, int h, int rx, int ry, int rw, int rh, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c1(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c2(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c3(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c4(unsigned char* pixels, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c1_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c2_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c3_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_c4_strided(unsigned char* pixels, int w, int h, int stride, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_circle_yuv420sp(unsigned char* yuv420sp, int w, int h, int cx, int cy, int radius, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c1(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c2(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c3(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c4(unsigned char* pixels, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c1_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c2_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c3_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_c4_strided(unsigned char* pixels, int w, int h, int stride, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_draw_line_yuv420sp(unsigned char* yuv420sp, int w, int h, int x0, int y0, int x1, int y1, unsigned int color, int thickness);
    void ncnn_rs_image_get_text_drawing_size(const char* text, int fontpixelsize, int* w, int* h);
    void ncnn_rs_image_draw_text_c1(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_c2(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_c3(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_c4(unsigned char* pixels, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_c1_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_c2_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_c3_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_c4_strided(unsigned char* pixels, int w, int h, int stride, const char* text, int x, int y, int fontpixelsize, unsigned int color);
    void ncnn_rs_image_draw_text_yuv420sp(unsigned char* yuv420sp, int w, int h, const char* text, int x, int y, int fontpixelsize, unsigned int color);
#ifdef __cplusplus
}
#endif
#endif
