#pragma once

#include <stddef.h>

#if defined(_WIN32)
#define PANOCALL_RENDER_API extern "C" __declspec(dllexport)
#else
#define PANOCALL_RENDER_API extern "C" __attribute__((visibility("default")))
#endif

// blend_mode: 0 = fast 50/50 overlap, 1 = cached feather weights.
PANOCALL_RENDER_API int panocall_renderer_create(
    int width0, int height0, int width1, int height1,
    const double* homography, int blend_mode, float feather_radius,
    void** renderer, int* output_width, int* output_height,
    char* error_message, int error_capacity);

PANOCALL_RENDER_API int panocall_renderer_render(
    void* renderer,
    const unsigned char* frame0_bgr, size_t frame0_bytes,
    const unsigned char* frame1_bgr, size_t frame1_bytes,
    unsigned char* panorama_bgr, size_t panorama_capacity,
    float* elapsed_ms, char* error_message, int error_capacity);

PANOCALL_RENDER_API int panocall_renderer_destroy(void* renderer);
