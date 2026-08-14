#include "cuda_renderer_api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <new>

#include <cuda_runtime.h>

namespace {

struct Renderer
{
    int width0 = 0, height0 = 0, width1 = 0, height1 = 0;
    int outputWidth = 0, outputHeight = 0;
    int translationX = 0, translationY = 0;
    size_t frame0Bytes = 0, frame1Bytes = 0, outputBytes = 0;
    unsigned char* frame0GPU = nullptr;
    unsigned char* frame1GPU = nullptr;
    unsigned char* outputGPU = nullptr;
    float* mapXGPU = nullptr;
    float* mapYGPU = nullptr;
    float* weight0GPU = nullptr;
    unsigned char* validityGPU = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;
};

void setError(char* output, int capacity, const char* message)
{
    if(output && capacity > 0)
    {
        std::snprintf(output, static_cast<size_t>(capacity), "%s", message ? message : "unknown error");
        output[capacity - 1] = '\0';
    }
}

bool invert3x3(const double* matrix, double* inverse)
{
    const double a = matrix[0], b = matrix[1], c = matrix[2];
    const double d = matrix[3], e = matrix[4], f = matrix[5];
    const double g = matrix[6], h = matrix[7], i = matrix[8];
    const double determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if(!std::isfinite(determinant) || std::abs(determinant) < 1e-12) return false;
    const double scale = 1.0 / determinant;
    inverse[0] =  (e * i - f * h) * scale;
    inverse[1] = -(b * i - c * h) * scale;
    inverse[2] =  (b * f - c * e) * scale;
    inverse[3] = -(d * i - f * g) * scale;
    inverse[4] =  (a * i - c * g) * scale;
    inverse[5] = -(a * f - c * d) * scale;
    inverse[6] =  (d * h - e * g) * scale;
    inverse[7] = -(a * h - b * g) * scale;
    inverse[8] =  (a * e - b * d) * scale;
    return true;
}

void release(Renderer* renderer)
{
    if(!renderer) return;
    if(renderer->started) cudaEventDestroy(renderer->started);
    if(renderer->finished) cudaEventDestroy(renderer->finished);
    if(renderer->frame0GPU) cudaFree(renderer->frame0GPU);
    if(renderer->frame1GPU) cudaFree(renderer->frame1GPU);
    if(renderer->outputGPU) cudaFree(renderer->outputGPU);
    if(renderer->mapXGPU) cudaFree(renderer->mapXGPU);
    if(renderer->mapYGPU) cudaFree(renderer->mapYGPU);
    if(renderer->weight0GPU) cudaFree(renderer->weight0GPU);
    if(renderer->validityGPU) cudaFree(renderer->validityGPU);
    if(renderer->stream) cudaStreamDestroy(renderer->stream);
    delete renderer;
}

__global__ void buildGeometryKernel(
    float* mapX, float* mapY, float* weight0, unsigned char* validity,
    int outputWidth, int outputHeight,
    int width0, int height0, int width1, int height1,
    int translationX, int translationY,
    double h00, double h01, double h02,
    double h10, double h11, double h12,
    double h20, double h21, double h22,
    int blendMode, float featherRadius)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = outputWidth * outputHeight;
    if(index >= pixels) return;
    const int x = index % outputWidth;
    const int y = index / outputWidth;
    const double denominator = h20 * x + h21 * y + h22;
    float sourceX = -1.0f, sourceY = -1.0f;
    bool valid0 = false;
    if(fabs(denominator) > 1e-12)
    {
        sourceX = static_cast<float>((h00 * x + h01 * y + h02) / denominator);
        sourceY = static_cast<float>((h10 * x + h11 * y + h12) / denominator);
        valid0 = sourceX >= 0.0f && sourceY >= 0.0f &&
                 sourceX <= width0 - 1.0f && sourceY <= height0 - 1.0f;
    }
    const int referenceX = x - translationX;
    const int referenceY = y - translationY;
    const bool valid1 = referenceX >= 0 && referenceX < width1 &&
                        referenceY >= 0 && referenceY < height1;
    mapX[index] = sourceX;
    mapY[index] = sourceY;
    validity[index] = static_cast<unsigned char>((valid0 ? 1 : 0) | (valid1 ? 2 : 0));
    float weight = valid0 ? 1.0f : 0.0f;
    if(valid0 && valid1)
    {
        if(blendMode == 1)
        {
            float distance0 = fminf(fminf(sourceX, width0 - 1.0f - sourceX),
                                    fminf(sourceY, height0 - 1.0f - sourceY));
            float distance1 = static_cast<float>(min(min(referenceX, width1 - 1 - referenceX),
                                                     min(referenceY, height1 - 1 - referenceY)));
            distance0 = fminf(fmaxf(distance0, 0.0f), featherRadius);
            distance1 = fminf(fmaxf(distance1, 0.0f), featherRadius);
            const float total = distance0 + distance1;
            weight = total > 1e-6f ? distance0 / total : 0.5f;
        }
        else
        {
            weight = 0.5f;
        }
    }
    weight0[index] = weight;
}

__device__ float sampleChannel(const unsigned char* image, int width, int height,
                               float x, float y, int channel)
{
    int x0 = min(max(static_cast<int>(floorf(x)), 0), width - 1);
    int y0 = min(max(static_cast<int>(floorf(y)), 0), height - 1);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float dx = x - x0;
    const float dy = y - y0;
    const float p00 = image[(y0 * width + x0) * 3 + channel];
    const float p01 = image[(y0 * width + x1) * 3 + channel];
    const float p10 = image[(y1 * width + x0) * 3 + channel];
    const float p11 = image[(y1 * width + x1) * 3 + channel];
    return (1.0f - dy) * ((1.0f - dx) * p00 + dx * p01) +
           dy * ((1.0f - dx) * p10 + dx * p11);
}

__global__ void warpBlendKernel(
    const unsigned char* frame0, const unsigned char* frame1, unsigned char* output,
    const float* mapX, const float* mapY, const float* weight0,
    const unsigned char* validity, int width0, int height0, int width1,
    int outputWidth, int outputHeight, int translationX, int translationY)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = outputWidth * outputHeight;
    if(index >= pixels) return;
    const unsigned char valid = validity[index];
    const int x = index % outputWidth;
    const int y = index / outputWidth;
    const int referenceX = x - translationX;
    const int referenceY = y - translationY;
    const float sourceWeight = weight0[index];
    for(int channel = 0; channel < 3; ++channel)
    {
        float source = 0.0f, reference = 0.0f;
        if(valid & 1)
            source = sampleChannel(frame0, width0, height0, mapX[index], mapY[index], channel);
        if(valid & 2)
            reference = frame1[(referenceY * width1 + referenceX) * 3 + channel];
        float value = 0.0f;
        if(valid == 3) value = sourceWeight * source + (1.0f - sourceWeight) * reference;
        else if(valid & 1) value = source;
        else if(valid & 2) value = reference;
        output[index * 3 + channel] = static_cast<unsigned char>(fminf(fmaxf(value, 0.0f), 255.0f));
    }
}

} // namespace

PANOCALL_RENDER_API int panocall_renderer_create(
    int width0, int height0, int width1, int height1,
    const double* homography, int blendMode, float featherRadius,
    void** outputRenderer, int* outputWidth, int* outputHeight,
    char* errorMessage, int errorCapacity)
{
    if(outputRenderer) *outputRenderer = nullptr;
    if(!outputRenderer || !outputWidth || !outputHeight || !homography ||
       width0 < 2 || height0 < 2 || width1 < 1 || height1 < 1 ||
       (blendMode != 0 && blendMode != 1) || featherRadius <= 0.0f)
    {
        setError(errorMessage, errorCapacity, "invalid renderer arguments");
        return 1;
    }

    double minX = 0.0, minY = 0.0, maxX = static_cast<double>(width1), maxY = static_cast<double>(height1);
    const double corners[4][2] = {{0, 0}, {static_cast<double>(width0 - 1), 0},
                                  {static_cast<double>(width0 - 1), static_cast<double>(height0 - 1)},
                                  {0, static_cast<double>(height0 - 1)}};
    for(const auto& corner : corners)
    {
        const double denominator = homography[6] * corner[0] + homography[7] * corner[1] + homography[8];
        if(std::abs(denominator) < 1e-12)
        {
            setError(errorMessage, errorCapacity, "homography maps a corner to infinity");
            return 2;
        }
        const double x = (homography[0] * corner[0] + homography[1] * corner[1] + homography[2]) / denominator;
        const double y = (homography[3] * corner[0] + homography[4] * corner[1] + homography[5]) / denominator;
        minX = std::min(minX, x); minY = std::min(minY, y);
        maxX = std::max(maxX, x); maxY = std::max(maxY, y);
    }
    const int minimumX = static_cast<int>(std::floor(minX));
    const int minimumY = static_cast<int>(std::floor(minY));
    const int maximumX = static_cast<int>(std::ceil(maxX));
    const int maximumY = static_cast<int>(std::ceil(maxY));
    const int panoramaWidth = maximumX - minimumX;
    const int panoramaHeight = maximumY - minimumY;
    if(panoramaWidth <= 0 || panoramaHeight <= 0 ||
       static_cast<long long>(panoramaWidth) * panoramaHeight > 250000000LL)
    {
        setError(errorMessage, errorCapacity, "invalid or excessive panorama dimensions");
        return 3;
    }

    double translated[9] = {
        homography[0] - minimumX * homography[6],
        homography[1] - minimumX * homography[7],
        homography[2] - minimumX * homography[8],
        homography[3] - minimumY * homography[6],
        homography[4] - minimumY * homography[7],
        homography[5] - minimumY * homography[8],
        homography[6], homography[7], homography[8]
    };
    double inverse[9];
    if(!invert3x3(translated, inverse))
    {
        setError(errorMessage, errorCapacity, "homography is singular");
        return 4;
    }

    Renderer* renderer = new(std::nothrow) Renderer();
    if(!renderer)
    {
        setError(errorMessage, errorCapacity, "renderer allocation failed");
        return 5;
    }
    renderer->width0 = width0; renderer->height0 = height0;
    renderer->width1 = width1; renderer->height1 = height1;
    renderer->outputWidth = panoramaWidth; renderer->outputHeight = panoramaHeight;
    renderer->translationX = -minimumX; renderer->translationY = -minimumY;
    renderer->frame0Bytes = static_cast<size_t>(width0) * height0 * 3;
    renderer->frame1Bytes = static_cast<size_t>(width1) * height1 * 3;
    renderer->outputBytes = static_cast<size_t>(panoramaWidth) * panoramaHeight * 3;
    const size_t pixels = static_cast<size_t>(panoramaWidth) * panoramaHeight;

    cudaError_t status = cudaStreamCreateWithFlags(&renderer->stream, cudaStreamNonBlocking);
    if(status == cudaSuccess) status = cudaEventCreate(&renderer->started);
    if(status == cudaSuccess) status = cudaEventCreate(&renderer->finished);
    if(status == cudaSuccess) status = cudaMalloc((void**)&renderer->frame0GPU, renderer->frame0Bytes);
    if(status == cudaSuccess) status = cudaMalloc((void**)&renderer->frame1GPU, renderer->frame1Bytes);
    if(status == cudaSuccess) status = cudaMalloc((void**)&renderer->outputGPU, renderer->outputBytes);
    if(status == cudaSuccess) status = cudaMalloc((void**)&renderer->mapXGPU, pixels * sizeof(float));
    if(status == cudaSuccess) status = cudaMalloc((void**)&renderer->mapYGPU, pixels * sizeof(float));
    if(status == cudaSuccess) status = cudaMalloc((void**)&renderer->weight0GPU, pixels * sizeof(float));
    if(status == cudaSuccess) status = cudaMalloc((void**)&renderer->validityGPU, pixels);
    if(status != cudaSuccess)
    {
        setError(errorMessage, errorCapacity, cudaGetErrorString(status));
        release(renderer);
        return 6;
    }

    constexpr int threads = 256;
    buildGeometryKernel<<<static_cast<int>((pixels + threads - 1) / threads), threads, 0, renderer->stream>>>(
        renderer->mapXGPU, renderer->mapYGPU, renderer->weight0GPU, renderer->validityGPU,
        panoramaWidth, panoramaHeight, width0, height0, width1, height1,
        renderer->translationX, renderer->translationY,
        inverse[0], inverse[1], inverse[2], inverse[3], inverse[4], inverse[5],
        inverse[6], inverse[7], inverse[8], blendMode, featherRadius);
    status = cudaStreamSynchronize(renderer->stream);
    if(status != cudaSuccess)
    {
        setError(errorMessage, errorCapacity, cudaGetErrorString(status));
        release(renderer);
        return 7;
    }
    *outputRenderer = renderer;
    *outputWidth = panoramaWidth;
    *outputHeight = panoramaHeight;
    setError(errorMessage, errorCapacity, "");
    return 0;
}

PANOCALL_RENDER_API int panocall_renderer_render(
    void* handle,
    const unsigned char* frame0, size_t frame0Bytes,
    const unsigned char* frame1, size_t frame1Bytes,
    unsigned char* panorama, size_t panoramaCapacity,
    float* elapsedMs, char* errorMessage, int errorCapacity)
{
    Renderer* renderer = static_cast<Renderer*>(handle);
    if(!renderer || !frame0 || !frame1 || !panorama || frame0Bytes < renderer->frame0Bytes ||
       frame1Bytes < renderer->frame1Bytes || panoramaCapacity < renderer->outputBytes)
    {
        setError(errorMessage, errorCapacity, "invalid render buffers");
        return 1;
    }
    cudaEventRecord(renderer->started, renderer->stream);
    cudaMemcpyAsync(renderer->frame0GPU, frame0, renderer->frame0Bytes,
                    cudaMemcpyHostToDevice, renderer->stream);
    cudaMemcpyAsync(renderer->frame1GPU, frame1, renderer->frame1Bytes,
                    cudaMemcpyHostToDevice, renderer->stream);
    constexpr int threads = 256;
    const size_t pixels = static_cast<size_t>(renderer->outputWidth) * renderer->outputHeight;
    warpBlendKernel<<<static_cast<int>((pixels + threads - 1) / threads), threads, 0, renderer->stream>>>(
        renderer->frame0GPU, renderer->frame1GPU, renderer->outputGPU,
        renderer->mapXGPU, renderer->mapYGPU, renderer->weight0GPU, renderer->validityGPU,
        renderer->width0, renderer->height0, renderer->width1,
        renderer->outputWidth, renderer->outputHeight,
        renderer->translationX, renderer->translationY);
    cudaMemcpyAsync(panorama, renderer->outputGPU, renderer->outputBytes,
                    cudaMemcpyDeviceToHost, renderer->stream);
    cudaEventRecord(renderer->finished, renderer->stream);
    const cudaError_t status = cudaStreamSynchronize(renderer->stream);
    if(status != cudaSuccess)
    {
        setError(errorMessage, errorCapacity, cudaGetErrorString(status));
        return 2;
    }
    if(elapsedMs) cudaEventElapsedTime(elapsedMs, renderer->started, renderer->finished);
    setError(errorMessage, errorCapacity, "");
    return 0;
}

PANOCALL_RENDER_API int panocall_renderer_destroy(void* renderer)
{
    release(static_cast<Renderer*>(renderer));
    return 0;
}
