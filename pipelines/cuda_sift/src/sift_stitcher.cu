
//==========================================================
//
// CUDA SIFT VR Stitcher
//
// Version 2A
//
// Stage 1  : Image Loading
// Stage 2  : GPU Memory Verification
// Stage 3A : Gaussian Kernel Generation
//
//==========================================================

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

#define MAX_KERNEL_SIZE 64



//----------------------------------------------------------
// Project Configuration
//----------------------------------------------------------

#define MAX_CAMERAS 6
#define ACTIVE_CAMERAS 2
#define NUM_OCTAVES 4
#define NUM_SCALES 8
#define NUM_DOG_IMAGES (NUM_SCALES - 1)
#define CONTRAST_THRESHOLD 1.0f

__constant__ float d_gaussianKernel[MAX_KERNEL_SIZE];

__constant__ int d_kernelRadius;
__constant__ float d_sigmaLevels[NUM_SCALES];

//----------------------------------------------------------
// Gaussian Parameters
//----------------------------------------------------------

//==========================================================
//
// NMS Keypoint Structure
//
//==========================================================

struct NMSKeypoint
{
    float response;

    int x;
    int y;

    int octave;
    int dog;
};

//==========================================================
//
// Oriented Keypoint
//
// Output of Orientation Assignment
//
//==========================================================

struct OrientedKeypoint
{
    int x;
    int y;

    int octave;
    int dog;

    float angle;
};

//==========================================================
//
// Descriptor
//
// Output of Descriptor Generation
//
//==========================================================

struct Descriptor
{
    //------------------------------------------------------
    // Keypoint Information
    //------------------------------------------------------

    int x;
    int y;

    int octave;
    int dog;

    float angle;

    //------------------------------------------------------
    // 128-D SIFT Descriptor
    //------------------------------------------------------

    float data[128];
};

//------------------------------------------------------
//
// Descriptor Match
//
// One accepted match after Lowe Ratio Test.
//
//------------------------------------------------------

struct DescriptorMatch
{
    //--------------------------------------------------
    // Descriptor index in Camera A
    //--------------------------------------------------

    int queryIndex;

    //--------------------------------------------------
    // Descriptor index in Camera B
    //--------------------------------------------------

    int trainIndex;

    //--------------------------------------------------
    // Euclidean Distance
    //--------------------------------------------------

    float distance;
};

//----------------------------------------------------------
// Camera Structure
//----------------------------------------------------------

struct Camera
{
    int width = 0;
    int height = 0;

    // Image stored on CPU
    vector<float> imageCPU;

    // Image stored on GPU
    float* imageGPU = nullptr;
    float* horizontalBufferGPU = nullptr;
    float* octaveBaseGPU = nullptr;
    //----------------------------------------------------------
    // Gaussian Pyramid
    //----------------------------------------------------------

    std::vector<float*> gaussianGPU;
    std::vector<float*> dogGPU;

    //----------------------------------------------------------
    //
    // Gradient Pyramid
    //
    //----------------------------------------------------------

    std::vector<float*> magnitudeGPU;

    std::vector<float*> orientationGPU;
    std::vector<unsigned char*> extremaGPU;

    std::vector<unsigned char*> contrastGPU;
    vector<unsigned char*> edgeGPU;
    vector<unsigned char*> nmsGPU;
    std::vector<NMSKeypoint> nmsKeypointsCPU;

    NMSKeypoint* nmsKeypointsGPU = nullptr;
    std::vector<OrientedKeypoint> orientedKeypointsCPU;
    // GPU Output Buffer
    OrientedKeypoint* orientedKeypointsGPU = nullptr;

    //----------------------------------------------------------
    //
    // Descriptor Buffers
    //
    //----------------------------------------------------------

    std::vector<Descriptor> descriptorsCPU;
    //--------------------------------------------------
    //
    // Descriptor Matches
    //
    //--------------------------------------------------

    DescriptorMatch* matchesGPU =
        nullptr;

    std::vector<DescriptorMatch>
        matchesCPU;

    int* matchCountGPU =
        nullptr;

    Descriptor* descriptorsGPU = nullptr;

    int* descriptorCountGPU = nullptr;

    // Number of Oriented Keypoints on GPU
    int* orientationCountGPU = nullptr;

    // Gradient Pointer Tables
    float** magnitudePointersGPU = nullptr;
    float** orientationPointersGPU = nullptr;

    // Pyramid Dimensions
    int* pyramidWidthsGPU = nullptr;
    int* pyramidHeightsGPU = nullptr;

    std::vector<int> pyramidWidths;

    std::vector<int> pyramidHeights;

    // Image copied back from GPU
    vector<float> imageFromGPU;
};



//----------------------------------------------------------
//
// Generate 1D Gaussian Kernel
//
// Radius = ceil(3*sigma)
//
// Kernel Size = 2*radius+1
//
// Kernel is normalized.
//
//----------------------------------------------------------

vector<float> generateGaussianKernel(
    float sigma,
    int& radius
)
{
    radius = (int)ceil(3.0f * sigma);

    int kernelSize =
        2 * radius + 1;

    vector<float> kernel(kernelSize);

    float sum = 0.0f;

    for(int i=-radius;i<=radius;i++)
    {
        float value =
            exp(
                -(float)(i*i) /
                (2.0f*sigma*sigma)
            );

        kernel[i+radius]=value;

        sum += value;
    }

    //------------------------------------------------------
    // Normalize
    //------------------------------------------------------

    for(int i=0;i<kernelSize;i++)
    {
        kernel[i]/=sum;
    }

    return kernel;
}

//----------------------------------------------------------
//
// Upload Gaussian Kernel to CUDA Constant Memory
//
// Purpose:
//
// Copies the Gaussian kernel and its radius
// from CPU memory to GPU constant memory.
//
//----------------------------------------------------------

bool uploadGaussianKernel(
    const std::vector<float>& kernel,
    int radius
)
{
    //------------------------------------------------------
    // Copy kernel coefficients
    //------------------------------------------------------

    cudaError_t err =
        cudaMemcpyToSymbol(
            d_gaussianKernel,
            kernel.data(),
            kernel.size() * sizeof(float)
        );

    if(err != cudaSuccess)
    {
        cout << "Failed to upload Gaussian Kernel\n";
        cout << cudaGetErrorString(err)
             << endl;

        return false;
    }

    //------------------------------------------------------
    // Copy kernel radius
    //------------------------------------------------------

    err =
        cudaMemcpyToSymbol(
            d_kernelRadius,
            &radius,
            sizeof(int)
        );

    if(err != cudaSuccess)
    {
        cout << "Failed to upload Kernel Radius\n";
        cout << cudaGetErrorString(err)
             << endl;

        return false;
    }

    //------------------------------------------------------
    // Synchronize
    //------------------------------------------------------

    err = cudaDeviceSynchronize();

    if(err != cudaSuccess)
    {
        cout << cudaGetErrorString(err)
             << endl;

        return false;
    }

    return true;
}
//----------------------------------------------------------
//
// Horizontal Gaussian Blur CUDA Kernel
//
// Purpose:
//
// Computes thread coordinates.
//
// Image processing will be added later.
//
//----------------------------------------------------------

__global__
void horizontalGaussianKernel(const float* inputImage, float* outputImage, int width, int height)
{
    extern __shared__ float tile[];
    const int radius = d_kernelRadius;
    const int tileWidth = blockDim.x + 2 * radius;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = threadIdx.y * tileWidth;
    const int clampedY = min(y, height - 1);
    for(int localX = threadIdx.x; localX < tileWidth; localX += blockDim.x)
    {
        const int globalX = min(max(blockIdx.x * blockDim.x + localX - radius, 0), width - 1);
        tile[row + localX] = inputImage[clampedY * width + globalX];
    }
    __syncthreads();
    if(x >= width || y >= height) return;
    float sum = 0.0f;
    #pragma unroll 4
    for(int offset = -radius; offset <= radius; ++offset)
        sum += tile[row + threadIdx.x + radius + offset] * d_gaussianKernel[offset + radius];
    outputImage[y * width + x] = sum;
}

__global__
void verticalGaussianKernel(const float* inputImage, float* outputImage, int width, int height)
{
    extern __shared__ float tile[];
    const int radius = d_kernelRadius;
    const int tileHeight = blockDim.y + 2 * radius;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    for(int localY = threadIdx.y; localY < tileHeight; localY += blockDim.y)
    {
        const int globalY = min(max(blockIdx.y * blockDim.y + localY - radius, 0), height - 1);
        const int clampedX = min(x, width - 1);
        tile[localY * blockDim.x + threadIdx.x] = inputImage[globalY * width + clampedX];
    }
    __syncthreads();
    if(x >= width || y >= height) return;
    float sum = 0.0f;
    #pragma unroll 4
    for(int offset = -radius; offset <= radius; ++offset)
        sum += tile[(threadIdx.y + radius + offset) * blockDim.x + threadIdx.x]
             * d_gaussianKernel[offset + radius];
    outputImage[y * width + x] = sum;
}

//----------------------------------------------------------
//
// Downsample CUDA Kernel
//
// Purpose:
//
// Downsample image by factor of 2
// using 2x2 averaging.
//
//----------------------------------------------------------

__global__
void downsampleKernel(
    const float* inputImage,
    float* outputImage,
    int inputWidth,
    int inputHeight
)
{
    //------------------------------------------------------
    // Output Pixel
    //------------------------------------------------------

    int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;

    //------------------------------------------------------
    // Output Size
    //------------------------------------------------------

    int outputWidth =
        inputWidth / 2;

    int outputHeight =
        inputHeight / 2;

    if(x >= outputWidth ||
       y >= outputHeight)
    {
        return;
    }

    //------------------------------------------------------
    // Corresponding Input Pixel
    //------------------------------------------------------

    int inputX =
        x * 2;

    int inputY =
        y * 2;

    //------------------------------------------------------
    // Read 2x2 Block
    //------------------------------------------------------

    float p00 =
        inputImage[
            inputY * inputWidth +
            inputX
        ];

    float p01 =
        inputImage[
            inputY * inputWidth +
            (inputX + 1)
        ];

    float p10 =
        inputImage[
            (inputY + 1) * inputWidth +
            inputX
        ];

    float p11 =
        inputImage[
            (inputY + 1) * inputWidth +
            (inputX + 1)
        ];

    //------------------------------------------------------
    // Average
    //------------------------------------------------------

    outputImage[
        y * outputWidth +
        x
    ] =
    (
        p00 +
        p01 +
        p10 +
        p11
    ) * 0.25f;
}


//----------------------------------------------------------
//
// Difference of Gaussian CUDA Kernel
//
// Purpose:
//
// Computes:
//
// DoG = Gaussian(i+1) - Gaussian(i)
//
// One CUDA thread computes one pixel.
//
//----------------------------------------------------------

__global__
void dogKernel(
    const float* gaussianImage1,
    const float* gaussianImage2,
    float* dogImage,
    int width,
    int height
)
{
    //------------------------------------------------------
    // Pixel Coordinates
    //------------------------------------------------------

    int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;

    //------------------------------------------------------
    // Boundary Check
    //------------------------------------------------------

    if(x >= width ||
       y >= height)
    {
        return;
    }

    //------------------------------------------------------
    // Pixel Index
    //------------------------------------------------------

    int index =
        y * width + x;

    //------------------------------------------------------
    // Difference of Gaussian
    //------------------------------------------------------

    dogImage[index] =
        gaussianImage2[index] -
        gaussianImage1[index];
}


//----------------------------------------------------------
//
// Gradient CUDA Kernel
//
// Purpose:
//
// Computes:
//
// 1. Gradient Magnitude
// 2. Gradient Orientation
//
// Using Sobel Operator.
//
//----------------------------------------------------------

__global__
void gradientKernel(
    const float* gaussianImage,
    float* magnitudeImage,
    float* orientationImage,
    int width,
    int height
)
{
    //------------------------------------------------------
    // Pixel Coordinates
    //------------------------------------------------------

    int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;

    //------------------------------------------------------
    // FIX: Grid Bounds Guard
    //
    // Grid is rounded up to multiples of 16.
    // Without this guard, threads with
    // x >= width (or y >= height) write out
    // of bounds whenever a dimension is not
    // a multiple of 16 (e.g. octave 1: 360x640),
    // corrupting the gradient pyramid.
    //------------------------------------------------------

    if(x >= width ||
       y >= height)
    {
        return;
    }

    //------------------------------------------------------
    // Ignore Boundary Pixels
    //------------------------------------------------------

    if(x == 0 ||
       y == 0 ||
       x == width - 1 ||
       y == height - 1)
    {
        int index = y * width + x;

        magnitudeImage[index] = 0.0f;
        orientationImage[index] = 0.0f;
        return;
    }

    //------------------------------------------------------
    // Sobel X
    //------------------------------------------------------

    float gx =

        -gaussianImage[(y-1)*width + (x-1)]
        +gaussianImage[(y-1)*width + (x+1)]

        -2.0f *
        gaussianImage[y*width + (x-1)]

        +2.0f *
        gaussianImage[y*width + (x+1)]

        -gaussianImage[(y+1)*width + (x-1)]
        +gaussianImage[(y+1)*width + (x+1)];

    //------------------------------------------------------
    // Sobel Y
    //------------------------------------------------------

    float gy =

        -gaussianImage[(y-1)*width + (x-1)]

        -2.0f *
        gaussianImage[(y-1)*width + x]

        -gaussianImage[(y-1)*width + (x+1)]

        +gaussianImage[(y+1)*width + (x-1)]

        +2.0f *
        gaussianImage[(y+1)*width + x]

        +gaussianImage[(y+1)*width + (x+1)];

    //------------------------------------------------------
    // Pixel Index
    //------------------------------------------------------

    int index =
        y * width + x;

    //------------------------------------------------------
    // Magnitude
    //------------------------------------------------------

    magnitudeImage[index] =
        sqrtf(
            gx * gx +
            gy * gy
        );

    //------------------------------------------------------
    // Orientation
    //------------------------------------------------------

    float angle =
        atan2f(
            gy,
            gx
        ) *
        180.0f /
        3.14159265358979323846f;

    if(angle < 0.0f)
    {
        angle += 360.0f;
    }

    orientationImage[index] =
        angle;
}

//----------------------------------------------------------
//
// CUDA Extrema Detection Kernel
//
// One thread processes one DoG pixel.
//
// Output:
//
// 0 = Not Extrema
// 1 = Extrema
//
//----------------------------------------------------------

__global__
void extremaKernel(
    const float* lowerDog,
    const float* currentDog,
    const float* upperDog,
    unsigned char* extremaMap,
    int width,
    int height
)
{
    //------------------------------------------------------
    // Pixel Coordinates
    //------------------------------------------------------

    int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;

    //------------------------------------------------------
    // FIX: Grid Bounds Guard
    //------------------------------------------------------

    if(x >= width ||
       y >= height)
    {
        return;
    }

    //------------------------------------------------------
    // Ignore Image Boundary
    //
    // FIX: cudaMalloc memory is uninitialized,
    // so border pixels must be explicitly
    // written to 0 (previously they were left
    // as garbage and could become phantom
    // keypoints).
    //------------------------------------------------------

    if(x == 0 ||
       y == 0 ||
       x == width - 1 ||
       y == height - 1)
    {
        extremaMap[y * width + x] = 0;
        return;
    }

    //------------------------------------------------------
    // Center Pixel
    //------------------------------------------------------

    int centerIndex =
        y * width + x;

    float centerValue =
        currentDog[centerIndex];

    bool isMaximum = true;
    bool isMinimum = true;

    //------------------------------------------------------
    // Check All 26 Neighbours
    //------------------------------------------------------

    for(int dy = -1;
        dy <= 1;
        dy++)
    {
        for(int dx = -1;
            dx <= 1;
            dx++)
        {
            int neighbourIndex =
                (y + dy) * width +
                (x + dx);

            //--------------------------------------------------
            // Lower Scale
            //--------------------------------------------------

            if(lowerDog[neighbourIndex] >= centerValue)
            {
                isMaximum = false;
            }

            if(lowerDog[neighbourIndex] <= centerValue)
            {
                isMinimum = false;
            }

            //--------------------------------------------------
            // Upper Scale
            //--------------------------------------------------

            if(upperDog[neighbourIndex] >= centerValue)
            {
                isMaximum = false;
            }

            if(upperDog[neighbourIndex] <= centerValue)
            {
                isMinimum = false;
            }

            //--------------------------------------------------
            // Same Scale
            //--------------------------------------------------

            if(dx == 0 &&
               dy == 0)
            {
                continue;
            }

            if(currentDog[neighbourIndex] >= centerValue)
            {
                isMaximum = false;
            }

            if(currentDog[neighbourIndex] <= centerValue)
            {
                isMinimum = false;
            }
        }
    }

    //------------------------------------------------------
    // Save Result
    //------------------------------------------------------

    extremaMap[centerIndex] =
        (isMaximum || isMinimum)
        ? 1
        : 0;
}

//----------------------------------------------------------
//
// CUDA Contrast Filter Kernel
//
// Keeps only strong extrema.
//
// Input:
//
// extremaMap
// dogImage
//
// Output:
//
// contrastMap
//
//----------------------------------------------------------

__global__
void contrastKernel(
    const float* dogImage,
    const unsigned char* extremaMap,
    unsigned char* contrastMap,
    int width,
    int height
)
{
    //------------------------------------------------------
    // Pixel Coordinates
    //------------------------------------------------------

    int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;

    //------------------------------------------------------
    // Bounds Check
    //------------------------------------------------------

    if(x >= width ||
       y >= height)
    {
        return;
    }

    //------------------------------------------------------
    // Pixel Index
    //------------------------------------------------------

    int index =
        y * width + x;

    //------------------------------------------------------
    // Not an extrema
    //------------------------------------------------------

    if(extremaMap[index] == 0)
    {
        contrastMap[index] = 0;
        return;
    }

    //------------------------------------------------------
    // Contrast Test
    //------------------------------------------------------

    //------------------------------------------------------
    // Contrast Test
    //------------------------------------------------------

    if(fabsf(dogImage[index]) >= CONTRAST_THRESHOLD)
    {
        contrastMap[index] = 1;
    }
    else
    {
        contrastMap[index] = 0;
    }
}

//=====================================================
// EDGE RESPONSE KERNEL
//=====================================================

__global__
void edgeKernel(
    const float* dogImage,
    const unsigned char* contrastMap,
    unsigned char* edgeMap,
    int width,
    int height
)
{
    int x =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    //--------------------------------------------------
    // FIX: Grid Bounds Guard
    //--------------------------------------------------

    if(x >= width || y >= height)
        return;

    //--------------------------------------------------
    // Boundary Check
    //
    // FIX: write 0 instead of leaving the
    // border bytes uninitialized.
    //--------------------------------------------------

    if(x == 0 || x == width - 1 ||
       y == 0 || y == height - 1)
    {
        edgeMap[y * width + x] = 0;
        return;
    }

    int index =
        y * width + x;

    //--------------------------------------------------
    // Must Pass Contrast Filter
    //--------------------------------------------------

    if(contrastMap[index] == 0)
    {
        edgeMap[index] = 0;
        return;
    }

    //--------------------------------------------------
    // Second Order Derivatives
    //--------------------------------------------------

    float center =
        dogImage[index];

    float Dxx =
        dogImage[y * width + (x + 1)]
        +
        dogImage[y * width + (x - 1)]
        -
        2.0f * center;

    float Dyy =
        dogImage[(y + 1) * width + x]
        +
        dogImage[(y - 1) * width + x]
        -
        2.0f * center;

    float Dxy =
        (
            dogImage[(y + 1) * width + (x + 1)]
            -
            dogImage[(y + 1) * width + (x - 1)]
            -
            dogImage[(y - 1) * width + (x + 1)]
            +
            dogImage[(y - 1) * width + (x - 1)]
        )
        * 0.25f;

    //--------------------------------------------------
    // Hessian
    //--------------------------------------------------

    float trace =
        Dxx + Dyy;

    float determinant =
        Dxx * Dyy -
        Dxy * Dxy;

    //--------------------------------------------------
    // Reject Singular Hessian
    //--------------------------------------------------

    if(determinant <= 0.0f)
    {
        edgeMap[index] = 0;
        return;
    }

    //--------------------------------------------------
    // Lowe Edge Response Test
    //--------------------------------------------------

    const float EDGE_THRESHOLD = 10.0f;

    float curvatureRatio =
        (trace * trace) /
        determinant;

    float limit =
        ((EDGE_THRESHOLD + 1.0f) *
         (EDGE_THRESHOLD + 1.0f))
        /
        EDGE_THRESHOLD;

    if(curvatureRatio < limit)
    {
        edgeMap[index] = 1;
    }
    else
    {
        edgeMap[index] = 0;
    }
}



//=====================================================
// EDGE GPU
//=====================================================

bool edgeGPU(
    const float* dogGPU,
    const unsigned char* contrastGPU,
    unsigned char* edgeGPU,
    int width,
    int height
)
{
    dim3 blockSize(16,16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    edgeKernel<<<
        gridSize,
        blockSize
    >>>(
        dogGPU,
        contrastGPU,
        edgeGPU,
        width,
        height
    );

    cudaError_t error =
        cudaGetLastError();

    if(error != cudaSuccess)
    {
        cout << "Edge Kernel Error : "
             << cudaGetErrorString(error)
             << endl;

        return false;
    }

    error =
        cudaDeviceSynchronize();

    if(error != cudaSuccess)
    {
        cout << "Edge Synchronization Error : "
             << cudaGetErrorString(error)
             << endl;

        return false;
    }

    return true;
}

//=====================================================
//
// Orientation Kernel
//
//=====================================================

__global__
void orientationKernel(

    const NMSKeypoint* nmsKeypoints,

    int numKeypoints,

    float** magnitudePyramid,

    float** orientationPyramid,

    int* pyramidWidths,

    int* pyramidHeights,

    OrientedKeypoint* orientedKeypoints,

    int* orientationCount,

    int maxOrientations
)
{
    //--------------------------------------------------
    // Thread Index
    //--------------------------------------------------

    int tid =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    //--------------------------------------------------
    // Bounds Check
    //--------------------------------------------------

    if(tid >= numKeypoints)
    {
        return;
    }

    //--------------------------------------------------
    // Read One NMS Keypoint
    //--------------------------------------------------

    NMSKeypoint kp =
        nmsKeypoints[tid];

    int x =
        kp.x;

    int y =
        kp.y;

    int octave =
        kp.octave;

    int scale =
        kp.dog;

     //--------------------------------------------------
    //
    // Read Gradient Images
    //
    //--------------------------------------------------

    int pyramidIndex =
        octave *
        NUM_SCALES +
        scale;

    float* magnitude =
        magnitudePyramid[
            pyramidIndex
        ];

    float* orientation =
        orientationPyramid[
            pyramidIndex
        ];
    //------------------------------------------------------
    //
    // Debug Gradient Pointers
    //
    //------------------------------------------------------

    if(tid == 0)
    {
        printf(
            "Octave = %d  Scale = %d  PyramidIndex = %d\n",
            octave,
            scale,
            pyramidIndex
        );

        printf(
            "Magnitude Pointer = %p\n",
            magnitude
        );

        printf(
            "Orientation Pointer = %p\n",
            orientation
        );
    }


    //--------------------------------------------------
    //
    // Image Dimensions
    //
    //--------------------------------------------------

    int width =
        pyramidWidths[
            octave
        ];

    int height =
        pyramidHeights[
            octave
        ];

    //--------------------------------------------------
    //
    // Sigma
    //
    //--------------------------------------------------

    float sigma =
        d_sigmaLevels[
            scale
        ];

    if(tid == 0)
    {
        printf(
            "Sigma = %f\n",
            sigma
        );
    }

    //--------------------------------------------------
    //
    // Orientation Radius
    //
    //--------------------------------------------------

    int radius =
        (int)roundf(
            3.0f *
            1.5f *
            sigma
        );

    if(tid == 0)
    {
        printf(
            "Radius = %d\n",
            radius
        );
    }

          //--------------------------------------------------
    //
    // Orientation Histogram
    //
    // 36 bins
    // Each bin = 10 degrees
    //
    //--------------------------------------------------

    float histogram[36];

    #pragma unroll
    for(
        int i = 0;
        i < 36;
        i++
    )
    {
        histogram[i] = 0.0f;
    }

    //--------------------------------------------------
    //
    // Scan Orientation Window
    //
    //--------------------------------------------------

    for(
        int dy = -radius;
        dy <= radius;
        dy++
    )
    {
        for(
            int dx = -radius;
            dx <= radius;
            dx++
        )
        {
            //------------------------------------------
            // Current Pixel
            //------------------------------------------

            int xx =
                x + dx;

            int yy =
                y + dy;

            //------------------------------------------
            // Boundary Check
            //------------------------------------------

            if(
                xx < 0 ||
                xx >= width ||
                yy < 0 ||
                yy >= height
            )
            {
                continue;
            }

            //--------------------------------------------------
            //
            // Gaussian Weight
            //
            //--------------------------------------------------

            float distanceSquared =
                (float)(
                    dx * dx +
                    dy * dy
                );

            float sigmaWeight =
                1.5f * sigma;

            float weight =
                expf(
                    -distanceSquared /
                    (
                        2.0f *
                        sigmaWeight *
                        sigmaWeight
                    )
                );

            //--------------------------------------------------
            //
            // Validate Weight
            //
            //--------------------------------------------------

            if(isnan(weight))
            {
                continue;
            }

            //--------------------------------------------------
            //
            // Read Magnitude
            //
            //--------------------------------------------------

            int pixelIndex =
                yy * width +
                xx;

            float magnitudeValue =
                magnitude[
                    pixelIndex
                ];

            //--------------------------------------------------
            //
            // Validate Magnitude
            //
            //--------------------------------------------------

            if(isnan(magnitudeValue))
            {
                continue;
            }

            //--------------------------------------------------
            //
            // Read Orientation
            //
            //--------------------------------------------------

            float angle =
                orientation[
                    pixelIndex
                ];

            //------------------------------------------------------
            //
            // Validate Orientation
            //
            //------------------------------------------------------

            if(isnan(angle))
            {
                continue;
            }

            if(angle < 0.0f)
            {
                angle += 360.0f;
            }

            if(angle >= 360.0f)
            {
                angle -= 360.0f;
            }

            if(tid == 0 && dx == 0 && dy == 0)
            {
                printf(
                    "dx=%d dy=%d mag=%f angle=%f weight=%f \n",
                    dx,
                    dy,
                    magnitudeValue,
                    angle,
                    weight
                );
            }

            //--------------------------------------------------
            //
            // Histogram Bin
            //
            //--------------------------------------------------

            int bin =
                (int)(
                    angle /
                    10.0f
                );

            if(tid == 0 && dx == 0 && dy == 0)
            {
                printf(
                    "Computed Bin = %d\n",
                    bin
                );
            }

            if(bin < 0)
            {
                bin = 0;
            }

            if(bin > 35)
            {
                bin = 35;
            }

            //--------------------------------------------------
            //
            // Histogram Vote
            //
            // (removed the per-pixel "Vote:" printf here:
            // it fired for every pixel of keypoint 0's
            // window — thousands of lines per run — and
            // could overflow the device printf buffer)
            //
            //--------------------------------------------------

            histogram[bin] +=
                weight *
                magnitudeValue;
        }
    }

        //--------------------------------------------------
        //
        // Find Histogram Peak
        //
        //--------------------------------------------------

        float maximumPeak =
            histogram[0];

        for(
            int i = 1;
            i < 36;
            i++
        )
        {
            if(
                histogram[i] >
                maximumPeak
            )
            {
                maximumPeak =
                    histogram[i];
            }
        }

        //--------------------------------------------------
        //
        // Secondary Peak Threshold
        //
        //--------------------------------------------------

        float peakThreshold =
            0.8f *
            maximumPeak;

          if(tid == 0)
            {
                printf(
                    "Maximum Peak = %f\n",
                    maximumPeak
                );

                printf(
                    "Peak Threshold = %f\n",
                    peakThreshold
                );

                for(int i = 0; i < 36; i++)
                {
                    printf(
                        "%d : %f\n",
                        i,
                        histogram[i]
                    );
                }
            }


            //--------------------------------------------------
            //
            // Create Orientation(s)
            //
            //--------------------------------------------------

            for(
                int bin = 0;
                bin < 36;
                bin++
            )
            {
                //----------------------------------------------
                // Keep Only Significant Peaks
                //----------------------------------------------

                if(
                    histogram[bin] <
                    peakThreshold
                )
                {
                    continue;
                }

                //----------------------------------------------
                // Convert Bin To Angle
                //----------------------------------------------

                float angle =
    (
                      bin +
                      0.5f
                  ) *
                  10.0f;

                //----------------------------------------------
                // Reserve Output Slot
                //----------------------------------------------

                int outputIndex =
                    atomicAdd(
                        orientationCount,
                        1
                    );

                //----------------------------------------------
                // FIX: Capacity Guard
                //
                // The output buffer holds 4 orientations
                // per NMS keypoint, but a flat histogram
                // (maximumPeak near 0) makes EVERY bin pass
                // the 0.8 * peak test, emitting up to 36
                // orientations from one keypoint. Without
                // this guard atomicAdd walks past the end
                // of orientedKeypoints and corrupts GPU
                // memory (garbage keypoints/descriptors).
                //----------------------------------------------

                if(outputIndex >= maxOrientations)
                {
                    continue;
                }

                //----------------------------------------------
                // Store Oriented Keypoint
                //----------------------------------------------

                orientedKeypoints[
                    outputIndex
                ].x =
                    x;

                orientedKeypoints[
                    outputIndex
                ].y =
                    y;

                orientedKeypoints[
                    outputIndex
                ].octave =
                    octave;

                orientedKeypoints[
                    outputIndex
                ].dog =
                    kp.dog;

                orientedKeypoints[
                    outputIndex
                ].angle =
                    angle;
            }
}

//==========================================================
//
// Descriptor Kernel
//
// One Thread = One Oriented Keypoint
//
//==========================================================

__global__
void descriptorKernel(

    OrientedKeypoint* orientedKeypoints,

    int totalKeypoints,

    float** magnitudePyramid,

    float** orientationPyramid,

    int* pyramidWidths,

    int* pyramidHeights,

    Descriptor* descriptors,

    int* descriptorCount
)
{
    //--------------------------------------------------
    // Thread Index
    //--------------------------------------------------

    int tid =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    if(tid >= totalKeypoints)
    {
        return;
    }

    //--------------------------------------------------
    // Read Oriented Keypoint
    //--------------------------------------------------

    OrientedKeypoint kp =
        orientedKeypoints[tid];

    int x =
        kp.x;

    int y =
        kp.y;

    int octave =
        kp.octave;

    int scale =
        kp.dog;

    float keypointAngle =
        kp.angle;

    //--------------------------------------------------
    // Pyramid Index
    //--------------------------------------------------

    int pyramidIndex =
        octave *
        NUM_SCALES +
        scale;

    //--------------------------------------------------
    // Read Gradient Images
    //--------------------------------------------------

    float* magnitude =
        magnitudePyramid[
            pyramidIndex
        ];

    float* orientation =
        orientationPyramid[
            pyramidIndex
        ];

    //--------------------------------------------------
    //
    // Read Pyramid Dimensions
    //
    //--------------------------------------------------

    int width =
        pyramidWidths[
            octave
        ];

    int height =
        pyramidHeights[
            octave
        ];

    //--------------------------------------------------
    //
    // Create Descriptor
    //
    //--------------------------------------------------

    Descriptor descriptor;

    descriptor.x =
        x;

    descriptor.y =
        y;

    descriptor.octave =
        octave;

    descriptor.dog =
        scale;

    descriptor.angle =
        keypointAngle;

    //--------------------------------------------------
    //
    // Initialize Descriptor
    //
    //--------------------------------------------------

    #pragma unroll
    for(
        int i = 0;
        i < 128;
        i++
    )
    {
        descriptor.data[i] =
            0.0f;
    }

    //--------------------------------------------------
    //
    // Descriptor Window
    //
    //--------------------------------------------------

    const int radius =
        8;

    //--------------------------------------------------
    //
    // Keypoint Angle
    //
    // CPU stores angle in degrees.
    //
    // Convert to radians only for
    // cos() and sin().
    //
    //--------------------------------------------------

    const float angleRadians =
        keypointAngle *
        (3.14159265358979323846f / 180.0f);

    const float cosTheta =
        cosf(
            angleRadians
        );

    const float sinTheta =
        sinf(
            angleRadians
        );

    //--------------------------------------------------
    //
    // Scan Descriptor Window
    //
    // CPU:
    //
    // for dy in range(-radius, radius):
    //     for dx in range(-radius, radius)
    //
    //--------------------------------------------------

    for(
        int dy = -radius;
        dy < radius;
        dy++
    )
    {
        for(
            int dx = -radius;
            dx < radius;
            dx++
        )
        {
            //------------------------------------------
            // Current Pixel
            //------------------------------------------

            int xx =
                x + dx;

            int yy =
                y + dy;

            //------------------------------------------
            // Boundary Check
            //------------------------------------------

            if(
                xx < 0 ||
                xx >= width ||
                yy < 0 ||
                yy >= height
            )
            {
                continue;
            }

            //--------------------------------------------------
            //
            // Rotate Coordinates Into Keypoint Frame
            //
            // CPU:
            //
            // x_rot = cos_t * dx + sin_t * dy
            // y_rot = -sin_t * dx + cos_t * dy
            //
            //--------------------------------------------------

            float xRot =
                cosTheta *
                static_cast<float>(dx)
                +
                sinTheta *
                static_cast<float>(dy);

            float yRot =
                -sinTheta *
                static_cast<float>(dx)
                +
                cosTheta *
                static_cast<float>(dy);

            //--------------------------------------------------
            //
            // Debug
            //
            //--------------------------------------------------

            if(
                tid == 0 &&
                dx == 0 &&
                dy == 0
            )
            {
                printf(
                    "xRot=%f  yRot=%f\n",
                    xRot,
                    yRot
                );
            }

            //--------------------------------------------------
            //
            // Convert To Descriptor Coordinates
            //
            // CPU:
            //
            // col_bin = (x_rot / 4.0) + 1.5
            // row_bin = (y_rot / 4.0) + 1.5
            //
            //--------------------------------------------------

            float colBin =
                (
                    xRot / 4.0f
                ) + 1.5f;

            float rowBin =
                (
                    yRot / 4.0f
                ) + 1.5f;

            //--------------------------------------------------
            //
            // Descriptor Cell Boundary Check
            //
            // CPU:
            //
            // if(
            //     row_bin < 0 ||
            //     row_bin >= 4 ||
            //     col_bin < 0 ||
            //     col_bin >= 4
            // )
            //     continue;
            //
            //--------------------------------------------------

            if(
                rowBin < 0.0f ||
                rowBin >= 4.0f ||
                colBin < 0.0f ||
                colBin >= 4.0f
            )
            {
                continue;
            }

            //--------------------------------------------------
            //
            // Relative Orientation
            //
            // CPU:
            //
            // angle =
            // (
            //     ori[yy,xx]
            //     -
            //     kp_angle
            // ) % 360
            //
            //--------------------------------------------------

            int pixelIndex =
                yy * width +
                xx;

            float angle =
                orientation[
                    pixelIndex
                ]
                -
                keypointAngle;

            //--------------------------------------------------
            //
            // Wrap Into [0,360)
            //
            //--------------------------------------------------

            while(angle < 0.0f)
            {
                angle += 360.0f;
            }

            while(angle >= 360.0f)
            {
                angle -= 360.0f;
            }

            //--------------------------------------------------
            //
            // Orientation Bin
            //
            // CPU:
            //
            // ori_bin = angle / 45
            //
            //--------------------------------------------------

            float orientationBin =
                angle /
                45.0f;



            //--------------------------------------------------
            //
            // Debug
            //
            //--------------------------------------------------

            if(
                tid == 0 &&
                dx == 0 &&
                dy == 0
            )
            {
                printf(
                    "Relative Angle=%f  Orientation Bin=%f\n",
                    angle,
                    orientationBin
                );
            }


            //--------------------------------------------------
            //
            // Descriptor Gaussian Weight
            //
            // CPU:
            //
            // sigma_desc = 8.0
            //
            //--------------------------------------------------

            const float sigmaDescriptor =
                8.0f;

            //--------------------------------------------------
            //
            // Squared Distance
            //
            // CPU:
            //
            // x_rot*x_rot + y_rot*y_rot
            //
            //--------------------------------------------------

            float distanceSquared =
                xRot *
                xRot
                +
                yRot *
                yRot;

            //--------------------------------------------------
            //
            // Gaussian Weight
            //
            // CPU:
            //
            // exp(
            // -(x_rot²+y_rot²)
            // /(2*sigma²)
            // )
            //
            //--------------------------------------------------

            float gaussianWeight =
                expf(
                    -distanceSquared
                    /
                    (
                        2.0f *
                        sigmaDescriptor *
                        sigmaDescriptor
                    )
                );

            //--------------------------------------------------
            //
            // Weighted Magnitude
            //
            // CPU:
            //
            // magnitude =
            // mag *
            // gaussian_weight
            //
            //--------------------------------------------------

            float weightedMagnitude =
                magnitude[
                    pixelIndex
                ]
                *
                gaussianWeight;

            //--------------------------------------------------
            //
            // Debug
            //
            //--------------------------------------------------

            if(
                tid == 0 &&
                dx == 0 &&
                dy == 0
            )
            {
                printf(
                    "Weight=%f  Weighted Magnitude=%f\n",
                    gaussianWeight,
                    weightedMagnitude
                );
            }

            //--------------------------------------------------
            //
            // Debug
            //
            //--------------------------------------------------

            if(
                tid == 0 &&
                dx == 0 &&
                dy == 0
            )
            {
                printf(
                    "rowBin=%f  colBin=%f\n",
                    rowBin,
                    colBin
                );
            }

            //--------------------------------------------------
            //
            // Row Interpolation
            //
            // CPU:
            //
            // r0 = floor(row_bin)
            // r1 = r0 + 1
            //
            // wr1 = row_bin - r0
            // wr0 = 1 - wr1
            //
            //--------------------------------------------------

            int r0 =
                (int)floorf(
                    rowBin
                );

            int r1 =
                r0 + 1;

            float wr1 =
                rowBin -
                (float)r0;

            float wr0 =
                1.0f -
                wr1;

            //--------------------------------------------------
            //
            // Column Interpolation
            //
            // CPU:
            //
            // c0 = floor(col_bin)
            // c1 = c0 + 1
            //
            // wc1 = col_bin - c0
            // wc0 = 1 - wc1
            //
            //--------------------------------------------------

            int c0 =
                (int)floorf(
                    colBin
                );

            int c1 =
                c0 + 1;

            float wc1 =
                colBin -
                (float)c0;

            float wc0 =
                1.0f -
                wc1;

            //--------------------------------------------------
            //
            // Debug
            //
            //--------------------------------------------------

            if(
                tid == 0 &&
                dx == 0 &&
                dy == 0
            )
            {
                printf(
                    "r0=%d r1=%d wr0=%f wr1=%f\n",
                    r0,
                    r1,
                    wr0,
                    wr1
                );

                printf(
                    "c0=%d c1=%d wc0=%f wc1=%f\n",
                    c0,
                    c1,
                    wc0,
                    wc1
                );
            }

            //--------------------------------------------------
            //
            // Orientation Interpolation
            //
            // CPU:
            //
            // o0 = floor(orientationBin) % 8
            // o1 = (o0 + 1) % 8
            //
            //--------------------------------------------------

            int o0 =
                (
                    int
                )floorf(
                    orientationBin
                ) % 8;

            int o1 =
                (
                    o0 + 1
                ) % 8;

            //--------------------------------------------------
            //
            // Orientation Weights
            //
            // CPU:
            //
            // wo1 =
            // orientationBin - floor(orientationBin)
            //
            // wo0 = 1 - wo1
            //
            //--------------------------------------------------

            float wo1 =
                orientationBin
                -
                floorf(
                    orientationBin
                );

            float wo0 =
                1.0f
                -
                wo1;

            //--------------------------------------------------
            //
            // Debug
            //
            //--------------------------------------------------

            if(
                tid == 0 &&
                dx == 0 &&
                dy == 0
            )
            {
                printf(
                    "o0=%d o1=%d wo0=%f wo1=%f\n",
                    o0,
                    o1,
                    wo0,
                    wo1
                );
            }

            //--------------------------------------------------
            //
            // Row Interpolation Loop
            //
            // CPU:
            //
            // for r,wr in
            // [
            //      (r0,wr0),
            //      (r1,wr1)
            // ]
            //
            //--------------------------------------------------

            for(
                int rowIteration = 0;
                rowIteration < 2;
                rowIteration++
            )
            {
                //----------------------------------------------
                // Current Row
                //----------------------------------------------

                int r =
                    (
                        rowIteration == 0
                    )
                    ?
                    r0
                    :
                    r1;

                //----------------------------------------------
                // Current Row Weight
                //----------------------------------------------

                float wr =
                    (
                        rowIteration == 0
                    )
                    ?
                    wr0
                    :
                    wr1;

                //----------------------------------------------
                // CPU:
                //
                // if not (0 <= r < 4)
                //      continue
                //
                //----------------------------------------------

                if(
                    r < 0 ||
                    r >= 4
                )
                {
                    continue;
                }

                //----------------------------------------------
                // Debug
                //----------------------------------------------

                if(
                    tid == 0 &&
                    dx == 0 &&
                    dy == 0
                )
                {
                    printf(
                        "Row = %d  Weight = %f\n",
                        r,
                        wr
                    );
                }

                //--------------------------------------------------
                //
                // Column Interpolation Loop
                //
                // CPU:
                //
                // for c,wc in
                // [
                //      (c0,wc0),
                //      (c1,wc1)
                // ]
                //
                //--------------------------------------------------

                for(
                    int columnIteration = 0;
                    columnIteration < 2;
                    columnIteration++
                )
                {
                    //----------------------------------------------
                    // Current Column
                    //----------------------------------------------

                    int c =
                        (
                            columnIteration == 0
                        )
                        ?
                        c0
                        :
                        c1;

                    //----------------------------------------------
                    // Current Column Weight
                    //----------------------------------------------

                    float wc =
                        (
                            columnIteration == 0
                        )
                        ?
                        wc0
                        :
                        wc1;

                    //----------------------------------------------
                    // CPU:
                    //
                    // if not (0 <= c < 4)
                    //      continue
                    //
                    //----------------------------------------------

                    if(
                        c < 0 ||
                        c >= 4
                    )
                    {
                        continue;
                    }

                    //----------------------------------------------
                    // Debug
                    //----------------------------------------------

                    if(
                        tid == 0 &&
                        dx == 0 &&
                        dy == 0
                    )
                    {
                        printf(
                            "Column = %d  Weight = %f\n",
                            c,
                            wc
                        );
                    }

                    //--------------------------------------------------
                    //
                    // Flatten Descriptor Index
                    //
                    // CPU:
                    //
                    // descriptor[r][c][o]
                    //
                    // CUDA:
                    //
                    // descriptor.data[index]
                    //
                    //--------------------------------------------------

                    int index0 =
                        r * 32 +
                        c * 8 +
                        o0;

                    int index1 =
                        r * 32 +
                        c * 8 +
                        o1;

                    //--------------------------------------------------
                    //
                    // Descriptor Accumulation
                    //
                    // CPU:
                    //
                    // descriptor[r,c,o0] +=
                    // magnitude *
                    // wr *
                    // wc *
                    // wo0
                    //
                    // descriptor[r,c,o1] +=
                    // magnitude *
                    // wr *
                    // wc *
                    // wo1
                    //
                    //--------------------------------------------------

                    descriptor.data[
                        index0
                    ] +=
                        weightedMagnitude *
                        wr *
                        wc *
                        wo0;

                    descriptor.data[
                        index1
                    ] +=
                        weightedMagnitude *
                        wr *
                        wc *
                        wo1;

                    //--------------------------------------------------
                    //
                    // Debug
                    //
                    //--------------------------------------------------

                    if(
                        tid == 0 &&
                        dx == 0 &&
                        dy == 0
                    )
                    {
                        printf(
                            "index0=%d value=%f\n",
                            index0,
                            descriptor.data[index0]
                        );

                        printf(
                            "index1=%d value=%f\n",
                            index1,
                            descriptor.data[index1]
                        );
                    }
                }
            }

            //------------------------------------------
            // Debug (first thread only)
            //------------------------------------------

            if(
                tid == 0 &&
                dx == 0 &&
                dy == 0
            )
            {
                printf(
                    "Descriptor Center Pixel : (%d,%d)\n",
                    xx,
                    yy
                );
            }
        }
    }


    //--------------------------------------------------
    //
    // First L2 Normalization
    //
    // CPU:
    //
    // norm = np.linalg.norm(descriptor)
    //
    //--------------------------------------------------

    float norm = 0.0f;

    #pragma unroll
    for(int i = 0; i < 128; i++)
    {
        norm +=
            descriptor.data[i] *
            descriptor.data[i];
    }

    norm = sqrtf(norm);

    if(norm > 1e-7f)
    {
        #pragma unroll
        for(int i = 0; i < 128; i++)
        {
            descriptor.data[i] /= norm;
        }
    }

    //--------------------------------------------------
    //
    // Clamp To 0.2
    //
    // CPU:
    //
    // descriptor = np.clip(descriptor,0,0.2)
    //
    //--------------------------------------------------

    #pragma unroll
    for(int i = 0; i < 128; i++)
    {
        if(descriptor.data[i] > 0.2f)
        {
            descriptor.data[i] = 0.2f;
        }
    }

    //--------------------------------------------------
    //
    // Second L2 Normalization
    //
    // CPU:
    //
    // descriptor /=
    // np.linalg.norm(descriptor)
    //
    //--------------------------------------------------

    norm = 0.0f;

    #pragma unroll
    for(int i = 0; i < 128; i++)
    {
        norm +=
            descriptor.data[i] *
            descriptor.data[i];
    }

    norm = sqrtf(norm) + 1e-7f;

    #pragma unroll
    for(int i = 0; i < 128; i++)
    {
        descriptor.data[i] /= norm;
    }

    //--------------------------------------------------
    //
    // Reserve Output Slot
    //
    // CPU:
    //
    // descriptors.append(descriptor)
    //
    //--------------------------------------------------

    int outputIndex =
        atomicAdd(
            descriptorCount,
            1
        );

    //--------------------------------------------------
    //
    // Store Descriptor
    //
    //--------------------------------------------------

    descriptors[
        outputIndex
    ] = descriptor;

    //--------------------------------------------------
    //
    // Debug
    //
    //--------------------------------------------------

    if(tid == 0)
    {
        printf(
            "Descriptor Stored At : %d\n",
            outputIndex
        );

        printf("\nFirst Descriptor Values\n");

      for(int i = 0; i < 16; i++)
      {
          printf(
              "%d : %f\n",
              i,
              descriptor.data[i]
          );
      }

      float descriptorNorm = 0.0f;

      for(int i = 0; i < 128; i++)
      {
          descriptorNorm +=
              descriptor.data[i] *
              descriptor.data[i];
      }

      printf(
          "Descriptor Norm = %f\n",
          sqrtf(descriptorNorm)
      );
    }


    //--------------------------------------------------
    // Debug
    //--------------------------------------------------

    if(tid == 0)
    {
        printf(
            "Descriptor Kernel\n"
        );

        printf(
            "x=%d y=%d octave=%d scale=%d angle=%f\n",
            x,
            y,
            octave,
            scale,
            keypointAngle
        );

        printf(
            "Magnitude=%p Orientation=%p\n",
            magnitude,
            orientation
        );
    }
}

//--------------------------------------------------
//
// Descriptor Matching Kernel
//
// Stage 1
//
// One Thread
// → One Query Descriptor
//
//--------------------------------------------------

__global__
void packDescriptorDataKernel(
    const Descriptor* descriptors, int count, float* data, float* norms)
{
    const int descriptor = blockIdx.x * blockDim.x + threadIdx.x;
    if(descriptor >= count) return;
    float norm = 0.0f;
    #pragma unroll
    for(int component = 0; component < 128; ++component)
    {
        const float value = descriptors[descriptor].data[component];
        data[descriptor * 128 + component] = value;
        norm = fmaf(value, value, norm);
    }
    norms[descriptor] = norm;
}

__global__
void descriptorMatchingKernel(
    const float* __restrict__ negativeTwoDotProducts,
    const float* __restrict__ queryNorms,
    const float* __restrict__ trainNorms,
    int queryCount,
    int trainCount,
    DescriptorMatch* matches,
    int* matchCount)
{
    const int queryIndex = blockIdx.x;
    if(queryIndex >= queryCount) return;
    float localBest = 1e30f;
    float localSecond = 1e30f;
    int localBestIndex = -1;
    const size_t row = static_cast<size_t>(queryIndex) * trainCount;
    for(int train = threadIdx.x; train < trainCount; train += blockDim.x)
    {
        const float distance = fmaxf(0.0f, queryNorms[queryIndex] + trainNorms[train]
                                           + negativeTwoDotProducts[row + train]);
        if(distance < localBest)
        {
            localSecond = localBest;
            localBest = distance;
            localBestIndex = train;
        }
        else if(distance < localSecond) localSecond = distance;
    }

    __shared__ float blockBest[256];
    __shared__ float blockSecond[256];
    __shared__ int blockBestIndex[256];
    blockBest[threadIdx.x] = localBest;
    blockSecond[threadIdx.x] = localSecond;
    blockBestIndex[threadIdx.x] = localBestIndex;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        float best = 1e30f;
        float second = 1e30f;
        int bestIndex = -1;
        for(int candidate = 0; candidate < blockDim.x; ++candidate)
        {
            const float candidateBest = blockBest[candidate];
            if(candidateBest < best)
            {
                second = fminf(best, blockSecond[candidate]);
                best = candidateBest;
                bestIndex = blockBestIndex[candidate];
            }
            else second = fminf(second, candidateBest);
        }
        constexpr float LOWE_RATIO_SQUARED = 0.75f * 0.75f;
        if(bestIndex >= 0 && best < LOWE_RATIO_SQUARED * second)
        {
            const int outputIndex = atomicAdd(matchCount, 1);
            matches[outputIndex].queryIndex = queryIndex;
            matches[outputIndex].trainIndex = bestIndex;
            matches[outputIndex].distance = sqrtf(best);
        }
    }
}

//=====================================================
// BUILD EDGE PYRAMID
//=====================================================

bool buildEdgePyramid(
    Camera cameras[]
)
{
    cout << "\n";
    cout << "========================================\n";
    cout << "Building Edge Pyramid\n";
    cout << "========================================\n";

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        int totalEdge = 0;

        cout << "\nCamera "
             << camera
             << endl;

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].width >> octave;

            int height =
                cameras[camera].height >> octave;

            cout << "\nOctave "
                 << octave
                 << endl;

            for(int dog = 1;
                dog < NUM_DOG_IMAGES - 1;
                dog++)
            {
                int index =
                    octave * NUM_DOG_IMAGES + dog;

                bool status =
                    edgeGPU(
                        cameras[camera].dogGPU[index],
                        cameras[camera].contrastGPU[index],
                        cameras[camera].edgeGPU[index],
                        width,
                        height
                    );

                if(!status)
                {
                    return false;
                }

                //--------------------------------------------------
                // Copy Edge Map
                //--------------------------------------------------

                vector<unsigned char> edgeCPU(
                    width * height
                );

                cudaMemcpy(
                    edgeCPU.data(),
                    cameras[camera].edgeGPU[index],
                    width * height *
                    sizeof(unsigned char),
                    cudaMemcpyDeviceToHost
                );

                //--------------------------------------------------
                // Count Points
                //--------------------------------------------------

                int edgeCount = 0;

                for(int i = 0;
                    i < width * height;
                    i++)
                {
                    if(edgeCPU[i])
                    {
                        edgeCount++;
                    }
                }

                totalEdge += edgeCount;

                cout << "Dog "
                     << dog
                     << " : "
                     << edgeCount
                     << endl;
            }
        }

        cout << "\n=================================\n";

        cout << "Camera "
             << camera
             << " Summary\n";

        cout << "=================================\n";

        cout << "Total Edge Points : "
             << totalEdge
             << endl;
    }

    return true;
}

//----------------------------------------------------------
//
// Allocate Gaussian Pyramid
//
// Purpose:
//
// Allocates GPU memory for
// all octaves and scales.
//
//----------------------------------------------------------

bool allocateGaussianPyramid(
    Camera& camera
)
{
    //------------------------------------------------------
    // Reserve metadata
    //------------------------------------------------------

    camera.gaussianGPU.resize(
        NUM_OCTAVES * NUM_SCALES,
        nullptr
    );

    camera.pyramidWidths.resize(
        NUM_OCTAVES
    );

    camera.pyramidHeights.resize(
        NUM_OCTAVES
    );

    //------------------------------------------------------
    // Initial size
    //------------------------------------------------------

    int currentWidth =
        camera.width;

    int currentHeight =
        camera.height;

    //------------------------------------------------------
    // Allocate every octave
    //------------------------------------------------------

    for(int octave = 0;
        octave < NUM_OCTAVES;
        octave++)
    {



        camera.pyramidWidths[octave] =
            currentWidth;

        camera.pyramidHeights[octave] =
            currentHeight;

        size_t imageBytes =
            currentWidth *
            currentHeight *
            sizeof(float);

        //--------------------------------------------------
        // Allocate every scale
        //--------------------------------------------------

        for(int scale = 0;
            scale < NUM_SCALES;
            scale++)
        {
            int index =
                octave * NUM_SCALES +
                scale;

            cudaError_t pyramidAllocStatus =
                cudaMalloc(
                    (void**)&camera.gaussianGPU[index],
                    imageBytes
                );

            if(pyramidAllocStatus != cudaSuccess)
            {
                cout
                    << "Pyramid Allocation Failed\n";

                return false;
            }
        }

        //--------------------------------------------------
        // Next Octave Size
        //--------------------------------------------------

        currentWidth /= 2;
        currentHeight /= 2;
    }

    return true;
}


//----------------------------------------------------------
//
// Allocate DoG Pyramid
//
//----------------------------------------------------------

bool allocateDoGPyramid(
    Camera cameras[]
)
{
    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cameras[camera].dogGPU.resize(
            NUM_OCTAVES *
            NUM_DOG_IMAGES
        );

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].pyramidWidths[octave];

            int height =
                cameras[camera].pyramidHeights[octave];

            size_t bytes =
                width *
                height *
                sizeof(float);

            for(int dog = 0;
                dog < NUM_DOG_IMAGES;
                dog++)
            {
                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cudaError_t dogAllocStatus =
                    cudaMalloc(
                        (void**)&
                        cameras[camera].
                        dogGPU[index],
                        bytes
                    );

                if(dogAllocStatus
                    != cudaSuccess)
                {
                    cout
                        << "DoG Allocation Failed\n";

                    return false;
                }
            }
        }
    }

    return true;
}

//----------------------------------------------------------
//
// Allocate Gradient Pyramid
//
//----------------------------------------------------------

bool allocateGradientPyramid(
    Camera cameras[]
)
{
    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        //--------------------------------------------------
        // Allocate Magnitude Pyramid
        //--------------------------------------------------

        cameras[camera].magnitudeGPU.resize(
            NUM_OCTAVES *
            NUM_SCALES
        );

        //--------------------------------------------------
        // Allocate Orientation Pyramid
        //--------------------------------------------------

        cameras[camera].orientationGPU.resize(
            NUM_OCTAVES *
            NUM_SCALES
        );

        //--------------------------------------------------
        // Allocate Every Image
        //--------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            size_t bytes =
                width *
                height *
                sizeof(float);

            for(int scale = 0;
                scale < NUM_SCALES;
                scale++)
            {
                int index =
                    octave *
                    NUM_SCALES +
                    scale;

                cudaError_t magnitudeStatus =
                    cudaMalloc(
                        (void**)&
                        cameras[camera].
                        magnitudeGPU[index],
                        bytes
                    );

                if(magnitudeStatus != cudaSuccess)
                {
                    cout
                        << "Magnitude Allocation Failed\n";

                    return false;
                }

                cudaError_t orientationStatus =
                    cudaMalloc(
                        (void**)&
                        cameras[camera].
                        orientationGPU[index],
                        bytes
                    );

                if(orientationStatus != cudaSuccess)
                {
                    cout
                        << "Orientation Allocation Failed\n";

                    return false;
                }
            }
        }
    }

    return true;
}

//----------------------------------------------------------
//
// Allocate Extrema Pyramid
//
// One byte per pixel.
//
// 0 = Not Extrema
// 1 = Extrema
//
//----------------------------------------------------------

bool allocateExtremaPyramid(
    Camera cameras[]
)
{
    //------------------------------------------------------
    // Every Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        //--------------------------------------------------
        // Total Extrema Images
        //--------------------------------------------------

        cameras[camera].extremaGPU.resize(
            NUM_OCTAVES *
            NUM_DOG_IMAGES
        );

        //--------------------------------------------------
        // Allocate Every Octave
        //--------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            size_t bytes =
                width *
                height *
                sizeof(unsigned char);

            //--------------------------------------------------
            // Allocate Every DoG Image
            //--------------------------------------------------

            for(int dog = 0;
                dog < NUM_DOG_IMAGES;
                dog++)
            {
                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cudaError_t extremaAllocStatus =
                    cudaMalloc(
                        (void**)&
                        cameras[camera].
                        extremaGPU[index],
                        bytes
                    );

                if(extremaAllocStatus
                    != cudaSuccess)
                {
                    cout
                        << "Extrema Allocation Failed\n";

                    cout
                        << cudaGetErrorString(
                            extremaAllocStatus
                        )
                        << endl;

                    return false;
                }

                //--------------------------------------------------
                // Initialize To Zero
                //--------------------------------------------------

                cudaError_t extremaClearStatus =
                    cudaMemset(
                        cameras[camera].
                        extremaGPU[index],
                        0,
                        bytes
                    );

                if(extremaClearStatus
                    != cudaSuccess)
                {
                    cout
                        << "Extrema Initialization Failed\n";

                    cout
                        << cudaGetErrorString(
                            extremaClearStatus
                        )
                        << endl;

                    return false;
                }
            }
        }
    }

    return true;
}

//----------------------------------------------------------
//
// Allocate Contrast Pyramid
//
// One byte per pixel.
//
// 0 = Rejected
// 1 = Passed Contrast Test
//
//----------------------------------------------------------

bool allocateContrastPyramid(
    Camera cameras[]
)
{
    //------------------------------------------------------
    // Every Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        //--------------------------------------------------
        // Total Contrast Images
        //--------------------------------------------------

        cameras[camera].contrastGPU.resize(
            NUM_OCTAVES *
            NUM_DOG_IMAGES
        );

        //--------------------------------------------------
        // Every Octave
        //--------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            size_t bytes =
                width *
                height *
                sizeof(unsigned char);

            //--------------------------------------------------
            // Every DoG Image
            //--------------------------------------------------

            for(int dog = 0;
                dog < NUM_DOG_IMAGES;
                dog++)
            {
                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cudaError_t allocStatus =
                    cudaMalloc(
                        (void**)&
                        cameras[camera].
                        contrastGPU[index],
                        bytes
                    );

                if(allocStatus != cudaSuccess)
                {
                    cout
                        << "Contrast Allocation Failed\n";

                    cout
                        << cudaGetErrorString(
                            allocStatus
                        )
                        << endl;

                    return false;
                }

                cudaError_t clearStatus =
                    cudaMemset(
                        cameras[camera].
                        contrastGPU[index],
                        0,
                        bytes
                    );

                if(clearStatus != cudaSuccess)
                {
                    cout
                        << "Contrast Initialization Failed\n";

                    cout
                        << cudaGetErrorString(
                            clearStatus
                        )
                        << endl;

                    return false;
                }
            }
        }
    }

    return true;
}


//==========================================================
//
// Allocate Edge Pyramid
//
//==========================================================

bool allocateEdgePyramid(
    Camera cameras[]
)
{
    cout << "\n=====================================\n";
    cout << "Allocating Edge Pyramid\n";
    cout << "=====================================\n";

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cameras[camera].edgeGPU.resize(
            NUM_OCTAVES * NUM_DOG_IMAGES,
            nullptr
        );

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].width >>
                octave;

            int height =
                cameras[camera].height >>
                octave;

            int pixels =
                width * height;

            for(int dog = 1;
                dog < NUM_DOG_IMAGES-1;
                dog++)
            {
                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cudaError_t err =
                    cudaMalloc(
                        &cameras[camera].edgeGPU[index],
                        pixels *
                        sizeof(unsigned char)
                    );

                if(err != cudaSuccess)
                {
                    cout
                        << "Edge Allocation Failed\n";

                    return false;
                }

                cudaMemset(
                    cameras[camera].edgeGPU[index],
                    0,
                    pixels *
                    sizeof(unsigned char)
                );
            }
        }
    }

    cout << "Edge Pyramid Allocated.\n";

    return true;
}

//==========================================================
//
// Allocate NMS Pyramid
//
//==========================================================

bool allocateNMSPyramid(
    Camera cameras[]
)
{
    cout << "\n=====================================\n";
    cout << "Allocating NMS Pyramid\n";
    cout << "=====================================\n";

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cameras[camera].nmsGPU.resize(
            NUM_OCTAVES * NUM_DOG_IMAGES,
            nullptr
        );

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].pyramidWidths[octave];

            int height =
                cameras[camera].pyramidHeights[octave];

            int pixels =
                width * height;

            for(int dog = 1;
                dog < NUM_DOG_IMAGES - 1;
                dog++)
            {
                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cudaError_t err =
                    cudaMalloc(
                        (void**)&
                        cameras[camera].
                        nmsGPU[index],
                        pixels *
                        sizeof(unsigned char)
                    );

                if(err != cudaSuccess)
                {
                    cout
                        << "NMS Allocation Failed\n";

                    return false;
                }

                cudaMemset(
                    cameras[camera].
                    nmsGPU[index],
                    0,
                    pixels *
                    sizeof(unsigned char)
                );
            }
        }
    }

    cout
        << "NMS Pyramid Allocated.\n";

    return true;
}



//==========================================================
//
// Extract Edge Keypoints
//
// Purpose:
//
// Converts the binary edge maps into a list of keypoints
// together with their DoG response.
//
//==========================================================

vector<NMSKeypoint> extractEdgeKeypoints(
    Camera cameras[],
    int cameraID
)
{
    vector<NMSKeypoint> keypoints;

    //------------------------------------------------------
    // Every Octave
    //------------------------------------------------------

    for(int octave = 0;
        octave < NUM_OCTAVES;
        octave++)
    {
        int width =
            cameras[cameraID].
            pyramidWidths[octave];

        int height =
            cameras[cameraID].
            pyramidHeights[octave];

        int pixels =
            width * height;

        //--------------------------------------------------
        // Download Edge Map
        //--------------------------------------------------

        vector<unsigned char> edgeCPU(
            pixels
        );

        //--------------------------------------------------
        // Download DoG Image
        //--------------------------------------------------

        vector<float> dogCPU(
            pixels
        );

        //--------------------------------------------------
        // Every Middle DoG
        //--------------------------------------------------

        for(int dog = 1;
            dog < NUM_DOG_IMAGES - 1;
            dog++)
        {
            int index =
                octave *
                NUM_DOG_IMAGES +
                dog;

            cudaMemcpy(
                edgeCPU.data(),
                cameras[cameraID].
                edgeGPU[index],
                pixels *
                sizeof(unsigned char),
                cudaMemcpyDeviceToHost
            );

            cudaMemcpy(
                dogCPU.data(),
                cameras[cameraID].
                dogGPU[index],
                pixels *
                sizeof(float),
                cudaMemcpyDeviceToHost
            );

            //--------------------------------------------------
            // Extract Keypoints
            //--------------------------------------------------

            for(int y = 0;
                y < height;
                y++)
            {
                for(int x = 0;
                    x < width;
                    x++)
                {
                    int pixel =
                        y * width + x;

                    if(edgeCPU[pixel] == 0)
                        continue;

                    NMSKeypoint kp;

                    kp.response =
                        fabs(
                            dogCPU[pixel]
                        );

                    kp.x = x;
                    kp.y = y;

                    kp.octave = octave;
                    kp.dog = dog;

                    keypoints.push_back(
                        kp
                    );
                }
            }
        }
    }

    cout << "\nExtracted "
         << keypoints.size()
         << " Edge Keypoints\n";

    return keypoints;
}

//==========================================================
//
// CPU Non Maximum Suppression
//
//==========================================================

vector<NMSKeypoint> performNMS(
    vector<NMSKeypoint>& keypoints,
    int radius = 10
)
{
    //------------------------------------------------------
    // Sort by response (largest first)
    //------------------------------------------------------

    sort(
        keypoints.begin(),
        keypoints.end(),
        [](const NMSKeypoint& a,
           const NMSKeypoint& b)
        {
            return a.response > b.response;
        }
    );

    //------------------------------------------------------
    // Selected Keypoints
    //------------------------------------------------------

    vector<NMSKeypoint> selected;

    int radiusSquared =
        radius * radius;

    //------------------------------------------------------
    // Greedy Selection
    //------------------------------------------------------

    for(const auto& kp : keypoints)
    {
        bool keep = true;

        for(const auto& chosen : selected)
        {
            //--------------------------------------------------
            // Different octave
            //--------------------------------------------------

            if(kp.octave != chosen.octave)
                continue;

            //--------------------------------------------------
            // Same DoG level
            //--------------------------------------------------



            //--------------------------------------------------
            // Squared Distance
            //--------------------------------------------------

            int dx =
                kp.x - chosen.x;

            int dy =
                kp.y - chosen.y;

            if(dx * dx + dy * dy < radiusSquared)
            {
                keep = false;
                break;
            }
        }

        if(keep)
        {
            selected.push_back(kp);
        }
    }

    cout
        << "\nAfter NMS : "
        << selected.size()
        << endl;

    return selected;
}


//==========================================================
//
// Build NMS Pyramid
//
//==========================================================

bool buildNMSPyramid(
    Camera cameras[]
)
{
    cout << "\n=========================================\n";
    cout << "Building NMS Pyramid\n";
    cout << "=========================================\n";

    //------------------------------------------------------
    // Every Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        //--------------------------------------------------
        // Extract Edge Keypoints
        //--------------------------------------------------

        vector<NMSKeypoint> edgeKeypoints =
            extractEdgeKeypoints(
                cameras,
                camera
            );

        //--------------------------------------------------
        // Perform NMS
        //--------------------------------------------------

        vector<NMSKeypoint> nmsKeypoints =
            performNMS(
                edgeKeypoints,
                10
            );

        cout
            << "Final NMS Keypoints : "
            << nmsKeypoints.size()
            << endl;


        //------------------------------------------------------
        //
        // Store Final NMS Keypoints
        //
        //------------------------------------------------------

        cameras[camera].
        nmsKeypointsCPU =
            nmsKeypoints;

        cout
            << "Final NMS Keypoints : "
            << nmsKeypoints.size()
            << endl;

        //--------------------------------------------------
        // Clear Every NMS Map
        //--------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            int pixels =
                width * height;

            for(int dog = 1;
                dog < NUM_DOG_IMAGES-1;
                dog++)
            {
                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cudaMemset(
                    cameras[camera].
                    nmsGPU[index],
                    0,
                    pixels *
                    sizeof(unsigned char)
                );
            }
        }

        //--------------------------------------------------
        // Write Selected Keypoints
        //--------------------------------------------------

        //------------------------------------------------------
        // Build NMS Binary Maps
        //------------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            int pixels =
                width * height;

            for(int dog = 1;
                dog < NUM_DOG_IMAGES - 1;
                dog++)
            {
                //--------------------------------------------------
                // Create one map for this DoG image
                //--------------------------------------------------

                vector<unsigned char> map(
                    pixels,
                    0
                );

                //--------------------------------------------------
                // Mark all surviving keypoints
                //--------------------------------------------------

                for(const auto& kp : nmsKeypoints)
                {
                    if(kp.octave != octave)
                        continue;

                    if(kp.dog != dog)
                        continue;

                    map[
                        kp.y * width +
                        kp.x
                    ] = 1;
                }

                //--------------------------------------------------
                // Upload once
                //--------------------------------------------------

                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cudaMemcpy(
                    cameras[camera].
                    nmsGPU[index],
                    map.data(),
                    pixels *
                    sizeof(unsigned char),
                    cudaMemcpyHostToDevice
                );
            }
        }
    }

    return true;
}



//=====================================================
//
// Allocate Orientation Buffers
//
//=====================================================

bool allocateOrientationBuffers(
    Camera& camera
)
{
    //--------------------------------------------------
    // Maximum Possible Orientations
    //
    // A keypoint can generate multiple orientations.
    // Reserve space for up to 4 orientations per keypoint.
    //--------------------------------------------------

    int maxOrientations =
        static_cast<int>(
            camera.nmsKeypointsCPU.size()
        ) * 4;

    if(maxOrientations == 0)
    {
        return true;
    }

    //--------------------------------------------------
    // Allocate Oriented Keypoints
    //--------------------------------------------------

    cudaError_t status =
        cudaMalloc(
            (void**)&
            camera.orientedKeypointsGPU,

            maxOrientations *
            sizeof(OrientedKeypoint)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Oriented Keypoints Buffer\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Allocate Orientation Counter
    //--------------------------------------------------

    status =
        cudaMalloc(
            (void**)&
            camera.orientationCountGPU,

            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Orientation Counter\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Reset Counter
    //--------------------------------------------------

    status =
        cudaMemset(
            camera.orientationCountGPU,
            0,
            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Reset Orientation Counter\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    return true;
}

//==========================================================
//
// Allocate Descriptor Buffers
//
//==========================================================

bool allocateDescriptorBuffers(
    Camera& camera
)
{
    //--------------------------------------------------
    // One Descriptor Per Oriented Keypoint
    //--------------------------------------------------

    int descriptorCount =
        static_cast<int>(
            camera.orientedKeypointsCPU.size()
        );

    if(descriptorCount == 0)
    {
        return true;
    }

    //--------------------------------------------------
    // Allocate Descriptor Buffer
    //--------------------------------------------------

    cudaError_t status =
        cudaMalloc(
            (void**)&
            camera.descriptorsGPU,

            descriptorCount *
            sizeof(Descriptor)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Descriptor Buffer\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Allocate Descriptor Counter
    //--------------------------------------------------

    status =
        cudaMalloc(
            (void**)&
            camera.descriptorCountGPU,

            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Descriptor Counter\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Reset Counter
    //--------------------------------------------------

    status =
        cudaMemset(
            camera.descriptorCountGPU,
            0,
            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Reset Descriptor Counter\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    return true;
}

//--------------------------------------------------
//
// Allocate Descriptor Matching Buffers
//
//--------------------------------------------------

bool allocateMatchBuffers(
    Camera& camera
)
{
    //--------------------------------------------------
    // Maximum Possible Matches
    //
    // Worst case:
    // Every descriptor finds one valid match.
    //--------------------------------------------------

    int maxMatches =
        static_cast<int>(
            camera.descriptorsCPU.size()
        );

    if(maxMatches == 0)
    {
        return true;
    }

    //--------------------------------------------------
    // Allocate Match Buffer
    //--------------------------------------------------

    cudaError_t status =
        cudaMalloc(
            (void**)&
            camera.matchesGPU,

            maxMatches *
            sizeof(
                DescriptorMatch
            )
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Match Buffer\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Allocate Match Counter
    //--------------------------------------------------

    status =
        cudaMalloc(
            (void**)&
            camera.matchCountGPU,

            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Match Counter\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Reset Counter
    //--------------------------------------------------

    status =
        cudaMemset(
            camera.matchCountGPU,

            0,

            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Reset Match Counter\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    return true;
}

//--------------------------------------------------
//
// Download Descriptor Matches
//
//--------------------------------------------------

bool downloadMatchResults(
    Camera& camera
)
{
    //--------------------------------------------------
    // Read Match Count
    //--------------------------------------------------

    int totalMatches = 0;

    cudaError_t status =
        cudaMemcpy(

            &totalMatches,

            camera.matchCountGPU,

            sizeof(int),

            cudaMemcpyDeviceToHost
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Download Match Count\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Resize CPU Vector
    //--------------------------------------------------

    cout
        << "Downloaded Match Count : "
        << totalMatches
        << endl;

    camera.matchesCPU.resize(
        totalMatches
    );

    //--------------------------------------------------
    // No Matches
    //--------------------------------------------------

    if(totalMatches == 0)
    {
        return true;
    }

    //--------------------------------------------------
    // Download Match Buffer
    //--------------------------------------------------

    status =
        cudaMemcpy(

            camera.matchesCPU.data(),

            camera.matchesGPU,

            totalMatches *
            sizeof(
                DescriptorMatch
            ),

            cudaMemcpyDeviceToHost
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Download Matches\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Debug
    //--------------------------------------------------

    cout
        << "Matches : "
        << camera.matchesCPU.size()
        << endl;

    //--------------------------------------------------
    //
    // Debug First Few Matches
    //
    //--------------------------------------------------

    int matchesToPrint =
        std::min(
            totalMatches,
            5
        );

    for(
        int i = 0;
        i < matchesToPrint;
        i++
    )
    {
        cout
            << "\nMatch "
            << i
            << endl;

        cout
            << "Query Index : "
            << camera.matchesCPU[i].queryIndex
            << endl;

        cout
            << "Train Index : "
            << camera.matchesCPU[i].trainIndex
            << endl;

        cout
            << "Distance    : "
            << camera.matchesCPU[i].distance
            << endl;
    }

        //--------------------------------------------------
        //
        // Match Statistics
        //
        //--------------------------------------------------

        float minDistance = 1e30f;
        float maxDistance = 0.0f;
        float sumDistance = 0.0f;

        for(const DescriptorMatch& match : camera.matchesCPU)
        {
            minDistance =
                std::min(
                    minDistance,
                    match.distance
                );

            maxDistance =
                std::max(
                    maxDistance,
                    match.distance
                );

            sumDistance +=
                match.distance;
        }

        float averageDistance =
            sumDistance /
            camera.matchesCPU.size();

        cout << "\n";
        cout << "==============================\n";
        cout << "Match Statistics\n";
        cout << "==============================\n";

        cout
            << "Minimum Distance : "
            << minDistance
            << endl;

        cout
            << "Maximum Distance : "
            << maxDistance
            << endl;

        cout
            << "Average Distance : "
            << averageDistance
            << endl;

        //--------------------------------------------------
        //
        // Match Validation
        //
        //--------------------------------------------------

        for(
            int i = 0;
            i < totalMatches;
            i++
        )
        {
            const DescriptorMatch& currentMatch =
                camera.matchesCPU[i];

            if(
                currentMatch.queryIndex < 0 ||
                currentMatch.queryIndex >=
                static_cast<int>(
                    camera.descriptorsCPU.size()
                )
            )
            {
                cout
                    << "Invalid Query Index : "
                    << currentMatch.queryIndex
                    << endl;
            }
            //--------------------------------------------------
            //
            // Validate Train Index
            //
            //--------------------------------------------------

            if(
                currentMatch.trainIndex < 0
            )
            {
                cout
                  << "Invalid Train Index : "
                  << currentMatch.trainIndex
                  << endl;
            }
        }


    return true;
}

bool exportDescriptorsForVisualization(
    const Camera& camera,
    const string& fileName
)
{
    //--------------------------------------------------
    // Open Output File
    //--------------------------------------------------

    ofstream outputFile(
        fileName
    );

    if(
        !outputFile.is_open()
    )
    {
        cout
            << "Failed To Open "
            << fileName
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Write Descriptor Count
    //--------------------------------------------------

    outputFile
        << camera.descriptorsCPU.size()
        << "\n";

    //--------------------------------------------------
    // Write Descriptor Information
    //--------------------------------------------------

    for(
        const Descriptor& descriptor :
        camera.descriptorsCPU
    )
    {
        outputFile

            << descriptor.x << " "

            << descriptor.y << " "

            << descriptor.octave << " "

            << descriptor.dog << " "

            << descriptor.angle

            << "\n";
    }

    //--------------------------------------------------
    // Close File
    //--------------------------------------------------

    outputFile.close();

    cout
        << "Exported "
        << camera.descriptorsCPU.size()
        << " descriptors to "
        << fileName
        << endl;

    return true;
}

bool exportMatchesForVisualization(
    const Camera& camera,
    const Camera* trainCamera,
    const string& fileName
)
{
    //--------------------------------------------------
    // Open Output File
    //--------------------------------------------------

    ofstream outputFile(
        fileName
    );

    if(
        !outputFile.is_open()
    )
    {
        cout
            << "Failed To Open "
            << fileName
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Write Match Count
    //--------------------------------------------------

    outputFile
        << camera.matchesCPU.size()
        << "\n";

    //--------------------------------------------------
    // Write Matches
    //--------------------------------------------------

    //--------------------------------------------------
    // FIX: also export coordinates.
    //
    // Descriptor output order is nondeterministic
    // (atomicAdd slot reservation), so indices are
    // only meaningful against descriptor files from
    // the SAME run. Exporting coordinates makes the
    // match file self-describing and safe to
    // visualize regardless of which run produced
    // the descriptor files.
    //
    // Format per line:
    // queryIndex trainIndex distance qx qy qOct tx ty tOct
    //--------------------------------------------------

    for(
        const DescriptorMatch& match :
        camera.matchesCPU
    )
    {
        const Descriptor& q =
            camera.descriptorsCPU[match.queryIndex];

        const Descriptor& t =
            trainCamera->descriptorsCPU[match.trainIndex];

        outputFile

            << match.queryIndex
            << " "

            << match.trainIndex
            << " "

            << match.distance
            << " "

            << q.x << " " << q.y << " " << q.octave
            << " "

            << t.x << " " << t.y << " " << t.octave

            << "\n";
    }

    //--------------------------------------------------
    // Close File
    //--------------------------------------------------

    outputFile.close();

    cout
        << "Exported "
        << camera.matchesCPU.size()
        << " matches to "
        << fileName
        << endl;

    return true;
}

//==========================================================
//
// Upload Oriented Keypoints
//
//==========================================================

bool uploadOrientedKeypoints(
    Camera& camera
)
{
    //--------------------------------------------------
    // Nothing To Upload
    //--------------------------------------------------

    if(
        camera.orientedKeypointsCPU.empty()
    )
    {
        return true;
    }

    //--------------------------------------------------
    // Copy CPU → GPU
    //--------------------------------------------------

    cudaError_t status =
        cudaMemcpy(
            camera.orientedKeypointsGPU,

            camera.orientedKeypointsCPU.data(),

            camera.orientedKeypointsCPU.size() *
            sizeof(OrientedKeypoint),

            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Upload Oriented Keypoints\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    return true;
}

//=====================================================
//
// Upload NMS Keypoints
//
//=====================================================

bool uploadNMSKeypoints(
    Camera& camera
)
{
    //--------------------------------------------------
    // Number of Keypoints
    //--------------------------------------------------

    int totalKeypoints =
        static_cast<int>(
            camera.nmsKeypointsCPU.size()
        );

    if(totalKeypoints == 0)
    {
        return true;
    }

    //--------------------------------------------------
    // Allocate GPU Buffer
    //--------------------------------------------------

    cudaError_t status =
        cudaMalloc(
            (void**)&
            camera.nmsKeypointsGPU,

            totalKeypoints *
            sizeof(NMSKeypoint)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate NMS GPU Buffer\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Upload Keypoints
    //--------------------------------------------------

    status =
        cudaMemcpy(

            camera.nmsKeypointsGPU,

            camera.nmsKeypointsCPU.data(),

            totalKeypoints *
            sizeof(NMSKeypoint),

            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Upload NMS Keypoints\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    return true;
}

//=====================================================
//
// Upload Gradient Pointer Tables
//
//=====================================================

bool uploadGradientPointers(
    Camera& camera
)
{
    //--------------------------------------------------
    // Total Pyramid Images
    //--------------------------------------------------

    int totalLevels =
        NUM_OCTAVES *
        NUM_SCALES;

    //--------------------------------------------------
    // Upload Magnitude Pointer Table
    //--------------------------------------------------

    cudaError_t status =
        cudaMalloc(
            (void**)&
            camera.magnitudePointersGPU,

            totalLevels *
            sizeof(float*)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Magnitude Pointer Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    status =
        cudaMemcpy(

            camera.magnitudePointersGPU,

            camera.magnitudeGPU.data(),

            totalLevels *
            sizeof(float*),

            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Upload Magnitude Pointer Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Upload Orientation Pointer Table
    //--------------------------------------------------

    status =
        cudaMalloc(
            (void**)&
            camera.orientationPointersGPU,

            totalLevels *
            sizeof(float*)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Orientation Pointer Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    status =
        cudaMemcpy(

            camera.orientationPointersGPU,

            camera.orientationGPU.data(),

            totalLevels *
            sizeof(float*),

            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Upload Orientation Pointer Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    return true;
}

//=====================================================
//
// Upload Pyramid Dimensions
//
//=====================================================

bool uploadPyramidDimensions(
    Camera& camera
)
{
    //--------------------------------------------------
    // Upload Width Table
    //--------------------------------------------------

    cudaError_t status =
        cudaMalloc(
            (void**)&
            camera.pyramidWidthsGPU,

            NUM_OCTAVES *
            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Width Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    status =
        cudaMemcpy(

            camera.pyramidWidthsGPU,

            camera.pyramidWidths.data(),

            NUM_OCTAVES *
            sizeof(int),

            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Upload Width Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Upload Height Table
    //--------------------------------------------------

    status =
        cudaMalloc(
            (void**)&
            camera.pyramidHeightsGPU,

            NUM_OCTAVES *
            sizeof(int)
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Allocate Height Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    status =
        cudaMemcpy(

            camera.pyramidHeightsGPU,

            camera.pyramidHeights.data(),

            NUM_OCTAVES *
            sizeof(int),

            cudaMemcpyHostToDevice
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Upload Height Table\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    return true;
}

//=====================================================
//
// Launch Orientation Kernel
//
//=====================================================

bool orientationGPU(

    NMSKeypoint* nmsKeypoints,

    int numKeypoints,

    float** magnitudePyramid,

    float** orientationPyramid,

    int* pyramidWidths,

    int* pyramidHeights,

    OrientedKeypoint* orientedKeypoints,

    int* orientationCount,

    int maxOrientations
)
{
    //--------------------------------------------------
    // Threads
    //--------------------------------------------------

    const int threads = 256;

    //--------------------------------------------------
    // Blocks
    //--------------------------------------------------

    int blocks =
        (numKeypoints +
         threads - 1) /
        threads;

    //--------------------------------------------------
    // Launch Kernel
    //--------------------------------------------------

    orientationKernel<<<
        blocks,
        threads
    >>>(

        nmsKeypoints,

        numKeypoints,

        magnitudePyramid,

        orientationPyramid,

        pyramidWidths,

        pyramidHeights,

        orientedKeypoints,

        orientationCount,

        maxOrientations
    );

    //--------------------------------------------------
    // Check Launch Error
    //--------------------------------------------------

    cudaError_t status =
        cudaGetLastError();

    if(status != cudaSuccess)
    {
        cout
            << "Orientation Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Synchronize
    //--------------------------------------------------

    status =
        cudaDeviceSynchronize();

    if(status != cudaSuccess)
    {
        cout
            << "Orientation Kernel Execution Failed\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    return true;
}

//=====================================================
//
// Descriptor GPU Launcher
//
//=====================================================

bool descriptorGPU(

    OrientedKeypoint* orientedKeypoints,

    int numKeypoints,

    float** magnitudePyramid,

    float** orientationPyramid,

    int* pyramidWidths,

    int* pyramidHeights,

    Descriptor* descriptors,

    int* descriptorCount
)
{
    //--------------------------------------------------
    // Threads
    //--------------------------------------------------

    const int threads = 256;

    //--------------------------------------------------
    // Blocks
    //--------------------------------------------------

    int blocks =
        (
            numKeypoints +
            threads - 1
        ) / threads;

    //--------------------------------------------------
    // Launch Kernel
    //--------------------------------------------------

    descriptorKernel<<<
        blocks,
        threads
    >>>(
        orientedKeypoints,

        numKeypoints,

        magnitudePyramid,

        orientationPyramid,

        pyramidWidths,

        pyramidHeights,

        descriptors,

        descriptorCount
    );

    //--------------------------------------------------
    // Check Launch
    //--------------------------------------------------

    cudaError_t status =
        cudaGetLastError();

    if(status != cudaSuccess)
    {
        cout
            << "Descriptor Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Synchronize
    //--------------------------------------------------

    status =
        cudaDeviceSynchronize();

    if(status != cudaSuccess)
    {
        cout
            << "Descriptor Kernel Execution Failed\n";

        cout
            << cudaGetErrorString(status)
            << endl;

        return false;
    }

    return true;
}

//=====================================================
//
// Download Orientation Results
//
//=====================================================

bool downloadOrientationResults(
    Camera& camera
)
{
    //--------------------------------------------------
    // Read Orientation Count
    //--------------------------------------------------

    int totalOrientations = 0;

    cudaError_t status =
        cudaMemcpy(

            &totalOrientations,

            camera.orientationCountGPU,

            sizeof(int),

            cudaMemcpyDeviceToHost
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Download Orientation Count\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Resize CPU Vector
    //--------------------------------------------------

    cout
        << "Downloaded Orientation Count : "
        << totalOrientations
        << endl;

    //--------------------------------------------------
    // FIX: the counter can exceed capacity when the
    // kernel hit the guard (flat-histogram keypoints);
    // clamp before copying so we never read past the
    // end of the GPU buffer.
    //--------------------------------------------------

    int capacity =
        static_cast<int>(
            camera.nmsKeypointsCPU.size()
        ) * 4;

    if(totalOrientations > capacity)
    {
        cout
            << "Orientation Count Clamped To Capacity : "
            << capacity
            << endl;

        totalOrientations = capacity;
    }

    camera.orientedKeypointsCPU.resize(
        totalOrientations
    );


    //--------------------------------------------------
    // No Keypoints
    //--------------------------------------------------

    if(totalOrientations == 0)
    {
        return true;
    }

    //--------------------------------------------------
    // Download Keypoints
    //--------------------------------------------------

    status =
        cudaMemcpy(

            camera.orientedKeypointsCPU.data(),

            camera.orientedKeypointsGPU,

            totalOrientations *
            sizeof(OrientedKeypoint),

            cudaMemcpyDeviceToHost
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Download Oriented Keypoints\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    return true;
}
//=====================================================
//
// Download Descriptor Results
//
//=====================================================

bool downloadDescriptorResults(
    Camera& camera
)
{
    //--------------------------------------------------
    // Read Descriptor Count
    //--------------------------------------------------

    int totalDescriptors = 0;

    cudaError_t status =
        cudaMemcpy(

            &totalDescriptors,

            camera.descriptorCountGPU,

            sizeof(int),

            cudaMemcpyDeviceToHost
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Download Descriptor Count\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    //--------------------------------------------------
    // Resize CPU Vector
    //--------------------------------------------------

    cout
        << "Downloaded Descriptor Count : "
        << totalDescriptors
        << endl;

    camera.descriptorsCPU.resize(
        totalDescriptors
    );

    //--------------------------------------------------
    // No Descriptors
    //--------------------------------------------------

    if(totalDescriptors == 0)
    {
        return true;
    }

    //--------------------------------------------------
    // Download Descriptor Buffer
    //--------------------------------------------------

    status =
        cudaMemcpy(

            camera.descriptorsCPU.data(),

            camera.descriptorsGPU,

            totalDescriptors *
            sizeof(Descriptor),

            cudaMemcpyDeviceToHost
        );

    if(status != cudaSuccess)
    {
        cout
            << "Failed To Download Descriptors\n";

        cout
            << cudaGetErrorString(
                status
            )
            << endl;

        return false;
    }

    cout
        << "Descriptors : "
        << camera.descriptorsCPU.size()
        << endl;

    return true;
}


//=====================================================
//
// Build Orientation Assignment
//
//=====================================================

bool buildOrientationAssignment(
    Camera cameras[]
)
{
    cout << "\n";
    cout << "========================================\n";
    cout << "Building Orientation Assignment\n";
    cout << "========================================\n";

    //--------------------------------------------------
    // Every Camera
    //--------------------------------------------------

    for(
        int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++
    )
    {
        cout
            << "\nCamera "
            << camera
            << endl;

        //--------------------------------------------------
        // Allocate Buffers
        //--------------------------------------------------

        if(
            !allocateOrientationBuffers(
                cameras[camera]
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        // Upload NMS Keypoints
        //--------------------------------------------------

        if(
            !uploadNMSKeypoints(
                cameras[camera]
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        // Upload Gradient Pointer Tables
        //--------------------------------------------------

        if(
            !uploadGradientPointers(
                cameras[camera]
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        // Upload Pyramid Dimensions
        //--------------------------------------------------

        if(
            !uploadPyramidDimensions(
                cameras[camera]
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        // Number of Keypoints
        //--------------------------------------------------

        int totalKeypoints =
            (int)
            cameras[camera]
            .nmsKeypointsCPU.size();

        if(totalKeypoints == 0)
        {
            continue;
        }

        //--------------------------------------------------
        // Launch Orientation Kernel
        //--------------------------------------------------

        if(
            !orientationGPU(

                cameras[camera]
                .nmsKeypointsGPU,

                totalKeypoints,

                cameras[camera]
                .magnitudePointersGPU,

                cameras[camera]
                .orientationPointersGPU,

                cameras[camera]
                .pyramidWidthsGPU,

                cameras[camera]
                .pyramidHeightsGPU,

                cameras[camera]
                .orientedKeypointsGPU,

                cameras[camera]
                .orientationCountGPU,

                totalKeypoints * 4
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        // Download Results
        //--------------------------------------------------

        if(
            !downloadOrientationResults(
                cameras[camera]
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        // Print Count
        //--------------------------------------------------

        cout
            << "Oriented Keypoints : "
            << cameras[camera]
               .orientedKeypointsCPU.size()
            << endl;
    }

    return true;
}


//==========================================================
//
// Build Descriptor Generation
//
//==========================================================

bool buildDescriptorGeneration(
    Camera cameras[]
)
{
    cout
        << "\n========================================\n";

    cout
        << "Building Descriptor Generation\n";

    cout
        << "========================================\n";

    //--------------------------------------------------
    //
    // Every Camera
    //
    //--------------------------------------------------

    for(
        int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++
    )
    {
        cout
            << "\nCamera "
            << camera
            << endl;

        //--------------------------------------------------
        //
        // Number Of Oriented Keypoints
        //
        //--------------------------------------------------

        int totalKeypoints =
            static_cast<int>(
                cameras[camera]
                .orientedKeypointsCPU.size()
            );

        cout
            << "Oriented Keypoints : "
            << totalKeypoints
            << endl;

        if(totalKeypoints == 0)
        {
            continue;
        }

        //--------------------------------------------------
        //
        // Allocate Descriptor Buffers
        //
        //--------------------------------------------------

        bool allocationStatus =
            allocateDescriptorBuffers(
                cameras[camera]
            );

        if(!allocationStatus)
        {
            return false;
        }

        //--------------------------------------------------
        //
        // Upload Oriented Keypoints
        //
        //--------------------------------------------------

        if(
            !uploadOrientedKeypoints(
                cameras[camera]
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        //
        // Reset Descriptor Counter
        //
        //--------------------------------------------------

        cout
            << "Descriptor Buffer      : "
            << cameras[camera].descriptorsGPU
            << endl;

        cout
            << "Descriptor Counter     : "
            << cameras[camera].descriptorCountGPU
            << endl;

        cudaError_t status =
            cudaMemset(

                cameras[camera]
                .descriptorCountGPU,

                0,

                sizeof(int)
            );

        if(status != cudaSuccess)
        {
            cout
                << "Failed To Reset Descriptor Counter\n";

            cout
                << cudaGetErrorString(
                    status
                )
                << endl;

            return false;
        }

        //--------------------------------------------------
        //
        // Launch Configuration
        //
        //--------------------------------------------------

        const int threads = 256;

        int blocks =
            (
                totalKeypoints +
                threads - 1
            )
            /
            threads;

        //--------------------------------------------------
        //
        // Launch Descriptor Kernel
        //
        //--------------------------------------------------

        descriptorKernel<<<
            blocks,
            threads
        >>>(

            cameras[camera]
            .orientedKeypointsGPU,

            totalKeypoints,

            cameras[camera]
            .magnitudePointersGPU,

            cameras[camera]
            .orientationPointersGPU,

            cameras[camera]
            .pyramidWidthsGPU,

            cameras[camera]
            .pyramidHeightsGPU,

            cameras[camera]
            .descriptorsGPU,

            cameras[camera]
            .descriptorCountGPU
        );

        //--------------------------------------------------
        //
        // Check Launch Error
        //
        //--------------------------------------------------

        status =
            cudaGetLastError();

        if(status != cudaSuccess)
        {
            cout
                << "Descriptor Kernel Launch Failed\n";

            cout
                << cudaGetErrorString(
                    status
                )
                << endl;

            return false;
        }

        //--------------------------------------------------
        //
        // Synchronize
        //
        //--------------------------------------------------

        status =
            cudaDeviceSynchronize();

        if(status != cudaSuccess)
        {
            cout
                << "Descriptor Kernel Execution Failed\n";

            cout
                << cudaGetErrorString(
                    status
                )
                << endl;

            return false;
        }

        //--------------------------------------------------
        //
        // Download Descriptor Results
        //
        //--------------------------------------------------

        if(
            !downloadDescriptorResults(
                cameras[camera]
            )
        )
        {
            return false;
        }

        //--------------------------------------------------
        //
        // Print Descriptor Count
        //
        //--------------------------------------------------

        cout
            << "Descriptors : "
            << cameras[camera]
               .descriptorsCPU.size()
            << endl;
    }

    return true;
}

//--------------------------------------------------
//
// Descriptor Matching
//
//--------------------------------------------------

bool buildDescriptorMatching(Camera cameras[])
{
    Camera& queryCamera = cameras[0];
    Camera& trainCamera = cameras[1];
    const int queryCount = static_cast<int>(queryCamera.descriptorsCPU.size());
    const int trainCount = static_cast<int>(trainCamera.descriptorsCPU.size());
    cout << "Query Descriptors : " << queryCount << endl;
    cout << "Train Descriptors : " << trainCount << endl;
    if(queryCount == 0 || trainCount == 0) return true;
    if(!allocateMatchBuffers(queryCamera)) return false;

    float *queryData = nullptr, *trainData = nullptr;
    float *queryNorms = nullptr, *trainNorms = nullptr, *distances = nullptr;
    cublasHandle_t handle = nullptr;
    auto cleanup = [&]() {
        if(handle) cublasDestroy(handle);
        if(queryData) cudaFree(queryData);
        if(trainData) cudaFree(trainData);
        if(queryNorms) cudaFree(queryNorms);
        if(trainNorms) cudaFree(trainNorms);
        if(distances) cudaFree(distances);
    };
    const size_t queryDataBytes = static_cast<size_t>(queryCount) * 128 * sizeof(float);
    const size_t trainDataBytes = static_cast<size_t>(trainCount) * 128 * sizeof(float);
    const size_t distanceBytes = static_cast<size_t>(queryCount) * trainCount * sizeof(float);
    if(cudaMalloc((void**)&queryData, queryDataBytes) != cudaSuccess ||
       cudaMalloc((void**)&trainData, trainDataBytes) != cudaSuccess ||
       cudaMalloc((void**)&queryNorms, queryCount * sizeof(float)) != cudaSuccess ||
       cudaMalloc((void**)&trainNorms, trainCount * sizeof(float)) != cudaSuccess ||
       cudaMalloc((void**)&distances, distanceBytes) != cudaSuccess)
    {
        cout << "Descriptor matrix allocation failed" << endl;
        cleanup(); return false;
    }
    constexpr int threads = 256;
    packDescriptorDataKernel<<<(queryCount + threads - 1) / threads, threads>>>(
        queryCamera.descriptorsGPU, queryCount, queryData, queryNorms);
    packDescriptorDataKernel<<<(trainCount + threads - 1) / threads, threads>>>(
        trainCamera.descriptorsGPU, trainCount, trainData, trainNorms);
    if(cudaGetLastError() != cudaSuccess || cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        cout << "Descriptor packing or cuBLAS initialization failed" << endl;
        cleanup(); return false;
    }
    const float alpha = -2.0f, beta = 0.0f;
    const cublasStatus_t gemmStatus = cublasSgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        trainCount, queryCount, 128,
        &alpha, trainData, 128, queryData, 128,
        &beta, distances, trainCount);
    if(gemmStatus != CUBLAS_STATUS_SUCCESS)
    {
        cout << "cuBLAS descriptor distance calculation failed" << endl;
        cleanup(); return false;
    }
    descriptorMatchingKernel<<<queryCount, threads>>>(
        distances, queryNorms, trainNorms, queryCount, trainCount,
        queryCamera.matchesGPU, queryCamera.matchCountGPU);
    const cudaError_t matchingStatus = cudaDeviceSynchronize();
    cleanup();
    if(matchingStatus != cudaSuccess)
    {
        cout << "Descriptor matching kernel failed: " << cudaGetErrorString(matchingStatus) << endl;
        return false;
    }
    if(!downloadMatchResults(queryCamera)) return false;
    if(!exportDescriptorsForVisualization(queryCamera, "camera0_descriptors.txt")) return false;
    if(!exportDescriptorsForVisualization(trainCamera, "camera1_descriptors.txt")) return false;
    if(!exportMatchesForVisualization(queryCamera, &trainCamera, "matches.txt")) return false;
    cout << "Total Accepted Matches : " << queryCamera.matchesCPU.size() << endl;
    return true;
}

//----------------------------------------------------------
//
// Generate Sigma Levels
//
// Purpose:
//
// Returns the fixed sigma values used
// by the Gaussian Pyramid.
//
//----------------------------------------------------------

std::vector<float> generateSigmaLevels()
{
    return
    {
       1.60f,
        2.00f,
        2.52f,
        3.17f,
        4.00f,
        5.04f,
        6.35f,
        8.00f
    };
}





//----------------------------------------------------------
//
// Gaussian Blur GPU
//
// Purpose:
//
// Performs complete Gaussian Blur on GPU.
//
// Input:
//
// inputImage
// outputImage
// width
// height
// sigma
//
//----------------------------------------------------------

bool gaussianBlurGPU(
    float* inputImage,
    float* horizontalBuffer,
    float* outputImage,
    int width,
    int height,
    float sigma
)
{


    //------------------------------------------------------
    // Remaining implementation
    //------------------------------------------------------

    //------------------------------------------------------
    //
    // Generate Gaussian Kernel
    //
    //------------------------------------------------------

    int kernelRadius = 0;

    std::vector<float> gaussianKernel =
        generateGaussianKernel(
            sigma,
            kernelRadius
        );



    //------------------------------------------------------
    //
    // Upload Kernel To Constant Memory
    //
    //------------------------------------------------------

    bool kernelUploadStatus =
        uploadGaussianKernel(
            gaussianKernel,
            kernelRadius
        );

    if(!kernelUploadStatus)
    {

        return false;
    }

    cout
        << "Sigma : "
        << sigma
        << " uploaded successfully."
        << endl;

    //------------------------------------------------------
    //
    // Configure CUDA Launch
    //
    //------------------------------------------------------

    dim3 blockSize(32,8);

    dim3 gridSize(

        (width + blockSize.x - 1)
        / blockSize.x,

        (height + blockSize.y - 1)
        / blockSize.y
    );


    //------------------------------------------------------
    //
    // Horizontal Gaussian Blur
    //
    //------------------------------------------------------

    const size_t horizontalSharedBytes = blockSize.y * (blockSize.x + 2 * kernelRadius) * sizeof(float);

    horizontalGaussianKernel<<<
        gridSize,
        blockSize,
        horizontalSharedBytes
    >>>(
        inputImage,
        horizontalBuffer,
        width,
        height
    );

    //------------------------------------------------------
    //
    // Verify Horizontal Kernel
    //
    //------------------------------------------------------

    cudaError_t horizontalLaunchStatus =
        cudaGetLastError();

    if(horizontalLaunchStatus != cudaSuccess)
    {
        cout
            << "Horizontal Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                horizontalLaunchStatus
            )
            << endl;



        return false;
    }


    cudaError_t horizontalSyncStatus =
    cudaDeviceSynchronize();

    if(horizontalSyncStatus != cudaSuccess)
    {
        cout
            << "Horizontal Kernel Failed\n";

        cout
            << cudaGetErrorString(
                horizontalSyncStatus
            )
            << endl;



        return false;
    }
    cout
    << "Horizontal Blur Completed.\n";


    //------------------------------------------------------
    //
    // Vertical Gaussian Blur
    //
    //------------------------------------------------------

    const size_t verticalSharedBytes = blockSize.x * (blockSize.y + 2 * kernelRadius) * sizeof(float);

    verticalGaussianKernel<<<
        gridSize,
        blockSize,
        verticalSharedBytes
    >>>(
        horizontalBuffer,
        outputImage,
        width,
        height
    );

    //------------------------------------------------------
    //
    // Verify Vertical Kernel Launch
    //
    //------------------------------------------------------

    cudaError_t verticalLaunchStatus =
        cudaGetLastError();

    if(verticalLaunchStatus != cudaSuccess)
    {
        cout
            << "Vertical Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                verticalLaunchStatus
            )
            << endl;



        return false;
    }

    //------------------------------------------------------
    //
    // Synchronize Vertical Kernel
    //
    //------------------------------------------------------

    cudaError_t verticalSyncStatus =
        cudaDeviceSynchronize();

    if(verticalSyncStatus != cudaSuccess)
    {
        cout
            << "Vertical Kernel Execution Failed\n";

        cout
            << cudaGetErrorString(
                verticalSyncStatus
            )
            << endl;



        return false;
    }

    cout
        << "Vertical Blur Completed.\n";
        return true;
}

//----------------------------------------------------------
//
// Downsample GPU
//
// Purpose:
//
// Downsamples an image by a factor of 2.
//
//----------------------------------------------------------

bool downsampleGPU(
    float* inputImage,
    float* outputImage,
    int inputWidth,
    int inputHeight
)
{
    //------------------------------------------------------
    // Output Size
    //------------------------------------------------------

    int outputWidth  = inputWidth  / 2;
    int outputHeight = inputHeight / 2;

    //------------------------------------------------------
    // CUDA Launch Configuration
    //------------------------------------------------------

    dim3 blockSize(16,16);

    dim3 gridSize(
        (outputWidth  + blockSize.x - 1) / blockSize.x,
        (outputHeight + blockSize.y - 1) / blockSize.y
    );

    //------------------------------------------------------
    // Launch Kernel
    //------------------------------------------------------

    downsampleKernel<<<
        gridSize,
        blockSize
    >>>(
        inputImage,
        outputImage,
        inputWidth,
        inputHeight
    );

    //------------------------------------------------------
    // Verify Launch
    //------------------------------------------------------

    cudaError_t downsampleLaunchStatus =
        cudaGetLastError();

    if(downsampleLaunchStatus != cudaSuccess)
    {
        cout
            << "Downsample Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                downsampleLaunchStatus
            )
            << endl;

        return false;
    }

    //------------------------------------------------------
    // Synchronize
    //------------------------------------------------------

    cudaError_t downsampleSyncStatus =
        cudaDeviceSynchronize();

    if(downsampleSyncStatus != cudaSuccess)
    {
        cout
            << "Downsample Kernel Failed\n";

        cout
            << cudaGetErrorString(
                downsampleSyncStatus
            )
            << endl;

        return false;
    }

    return true;
}



//----------------------------------------------------------
//
// Difference of Gaussian GPU
//
// Purpose:
//
// Launches the CUDA DoG kernel.
//
//----------------------------------------------------------

bool dogGPU(
    float* gaussianImage1,
    float* gaussianImage2,
    float* dogImage,
    int width,
    int height
)
{
    //------------------------------------------------------
    // CUDA Configuration
    //------------------------------------------------------

    dim3 blockSize(16,16);

    dim3 gridSize(
        (width  + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    //------------------------------------------------------
    // Launch CUDA Kernel
    //------------------------------------------------------

    dogKernel<<<
        gridSize,
        blockSize
    >>>(
        gaussianImage1,
        gaussianImage2,
        dogImage,
        width,
        height
    );

    //------------------------------------------------------
    // Check Launch
    //------------------------------------------------------

    cudaError_t dogLaunchStatus =
        cudaGetLastError();

    if(dogLaunchStatus != cudaSuccess)
    {
        cout
            << "DoG Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                dogLaunchStatus
            )
            << endl;

        return false;
    }

    //------------------------------------------------------
    // Synchronize
    //------------------------------------------------------

    cudaError_t dogSyncStatus =
        cudaDeviceSynchronize();

    if(dogSyncStatus != cudaSuccess)
    {
        cout
            << "DoG Kernel Execution Failed\n";

        cout
            << cudaGetErrorString(
                dogSyncStatus
            )
            << endl;

        return false;
    }

    return true;
}


//----------------------------------------------------------
//
// Gradient GPU
//
// Purpose:
//
// Launches Gradient CUDA Kernel
// to generate:
//
// 1. Magnitude Image
// 2. Orientation Image
//
//----------------------------------------------------------

bool gradientGPU(
    float* gaussianImage,
    float* magnitudeImage,
    float* orientationImage,
    int width,
    int height
)
{
    //------------------------------------------------------
    // CUDA Launch Configuration
    //------------------------------------------------------

    dim3 blockSize(16,16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    //------------------------------------------------------
    // Launch Gradient Kernel
    //------------------------------------------------------

    gradientKernel<<<
        gridSize,
        blockSize
    >>>(
        gaussianImage,
        magnitudeImage,
        orientationImage,
        width,
        height
    );

    //------------------------------------------------------
    // Check Launch Status
    //------------------------------------------------------

    cudaError_t gradientLaunchStatus =
        cudaGetLastError();

    if(gradientLaunchStatus != cudaSuccess)
    {
        cout
            << "Gradient Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                gradientLaunchStatus
            )
            << endl;

        return false;
    }

    //------------------------------------------------------
    // Synchronize Device
    //------------------------------------------------------

    cudaError_t gradientSyncStatus =
        cudaDeviceSynchronize();

    if(gradientSyncStatus != cudaSuccess)
    {
        cout
            << "Gradient Kernel Execution Failed\n";

        cout
            << cudaGetErrorString(
                gradientSyncStatus
            )
            << endl;

        return false;
    }

    return true;
}

//----------------------------------------------------------
//
// Extrema GPU
//
// Launches CUDA Extrema Kernel.
//
//----------------------------------------------------------

bool extremaGPU(
    float* lowerDog,
    float* currentDog,
    float* upperDog,
    unsigned char* extremaMap,
    int width,
    int height
)
{
    //------------------------------------------------------
    // CUDA Launch Configuration
    //------------------------------------------------------

    dim3 blockSize(16,16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    //------------------------------------------------------
    // Launch CUDA Kernel
    //------------------------------------------------------

    extremaKernel<<<
        gridSize,
        blockSize
    >>>(
        lowerDog,
        currentDog,
        upperDog,
        extremaMap,
        width,
        height
    );

    //------------------------------------------------------
    // Check Launch Status
    //------------------------------------------------------

    cudaError_t extremaLaunchStatus =
        cudaGetLastError();

    if(extremaLaunchStatus != cudaSuccess)
    {
        cout
            << "Extrema Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                extremaLaunchStatus
            )
            << endl;

        return false;
    }

    //------------------------------------------------------
    // Synchronize
    //------------------------------------------------------

    cudaError_t extremaSyncStatus =
        cudaDeviceSynchronize();

    if(extremaSyncStatus != cudaSuccess)
    {
        cout
            << "Extrema Kernel Execution Failed\n";

        cout
            << cudaGetErrorString(
                extremaSyncStatus
            )
            << endl;

        return false;
    }

    return true;
}

//----------------------------------------------------------
//
// Contrast GPU
//
// Launches CUDA Contrast Kernel.
//
//----------------------------------------------------------

bool contrastGPU(
    float* dogImage,
    unsigned char* extremaMap,
    unsigned char* contrastMap,
    int width,
    int height
)
{
    //------------------------------------------------------
    // CUDA Launch Configuration
    //------------------------------------------------------

    dim3 blockSize(16,16);

    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    //------------------------------------------------------
    // Launch CUDA Kernel
    //------------------------------------------------------

    contrastKernel<<<
        gridSize,
        blockSize
    >>>(
        dogImage,
        extremaMap,
        contrastMap,
        width,
        height
    );

    //------------------------------------------------------
    // Check Launch Status
    //------------------------------------------------------

    cudaError_t launchStatus =
        cudaGetLastError();

    if(launchStatus != cudaSuccess)
    {
        cout
            << "Contrast Kernel Launch Failed\n";

        cout
            << cudaGetErrorString(
                launchStatus
            )
            << endl;

        return false;
    }

    //------------------------------------------------------
    // Synchronize
    //------------------------------------------------------

    cudaError_t syncStatus =
        cudaDeviceSynchronize();

    if(syncStatus != cudaSuccess)
    {
        cout
            << "Contrast Kernel Execution Failed\n";

        cout
            << cudaGetErrorString(
                syncStatus
            )
            << endl;

        return false;
    }

    return true;
}

//----------------------------------------------------------
//
// Build Gaussian Pyramid
//
// Purpose:
//
// Builds the complete Gaussian Pyramid
// for every active camera.
//
//----------------------------------------------------------

bool buildGaussianPyramid(
    Camera cameras[],
    const std::vector<float>& sigmaLevels
)
{
    cout << "\n=================================\n";
    cout << "Building Gaussian Pyramid\n";
    cout << "=================================\n";

    //------------------------------------------------------
    // Process Every Active Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cout << "\nCamera "
             << camera
             << endl;

        //--------------------------------------------------
        // Current Octave Size
        //--------------------------------------------------

        int currentWidth =
            cameras[camera].width;

        int currentHeight =
            cameras[camera].height;

        //--------------------------------------------------
        // Process Every Octave
        //--------------------------------------------------

        float* currentImage =
          cameras[camera].octaveBaseGPU;

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            cout
                << "\n  Octave "
                << octave
                << endl;

            cout
                << "  Size : "
                << currentWidth
                << " x "
                << currentHeight
                << endl;

            //--------------------------------------------------
            // Process Every Scale
            //--------------------------------------------------

            for(int scale = 0;
                scale < NUM_SCALES;
                scale++)
            {
                //--------------------------------------------------
                //
                // Build Gaussian Scale
                //
                //--------------------------------------------------

                int pyramidIndex =
                    octave * NUM_SCALES +
                    scale;

                bool gaussianStatus =
                   gaussianBlurGPU(
                        scale == 0 ? currentImage : cameras[camera].gaussianGPU[pyramidIndex - 1],
                        cameras[camera].horizontalBufferGPU,
                        cameras[camera].gaussianGPU[pyramidIndex],
                        currentWidth,
                        currentHeight,
                        scale == 0 ? sigmaLevels[0] : sqrtf(
                            sigmaLevels[scale] * sigmaLevels[scale] -
                            sigmaLevels[scale - 1] * sigmaLevels[scale - 1])
                    );

                if(!gaussianStatus)
                {
                    cout
                        << "Gaussian Blur Failed\n";

                    return false;
                }

                cout
                    << "Camera "
                    << camera
                    << "  Octave "
                    << octave
                    << "  Scale "
                    << scale
                    << " Completed\n";
            }

            //--------------------------------------------------
            // Next Octave Size
            //--------------------------------------------------

            //--------------------------------------------------
            //
            // Build Next Octave
            //
            //--------------------------------------------------



            // Build Next Octave
            //
            //--------------------------------------------------

            if(octave < NUM_OCTAVES - 1)
            {
                //--------------------------------------------------
                // FIX: downsample the RAW octave base,
                // not gaussianGPU[octave][NUM_SCALES-1].
                //
                // The CPU reference halves the UNBLURRED
                // image between octaves:
                //
                //     current = cv2.resize(current, (w//2, h//2))
                //
                // The old code downsampled the sigma = 8.0
                // blurred scale instead, so every octave >= 1
                // started from a massively over-blurred base
                // (effective sigma ~4 in octave pixels BEFORE
                // the per-scale blur was applied again).
                // Result: DoG responses collapse, octave 1-3
                // keypoints disappear or move, and the GPU
                // output no longer matches the CPU pipeline.
                //
                // Downsample cannot run in place (parallel
                // threads read rows 2y/2y+1 while others
                // write row y), so route through the spare
                // horizontal buffer, then copy back.
                //--------------------------------------------------

                bool downsampleStatus =
                    downsampleGPU(
                        cameras[camera].octaveBaseGPU,
                        cameras[camera].horizontalBufferGPU,
                        currentWidth,
                        currentHeight
                    );

                if(!downsampleStatus)
                {
                    cout
                        << "Downsample Failed\n";

                    return false;
                }

                int nextPixels =
                    (currentWidth  / 2) *
                    (currentHeight / 2);

                cudaError_t copyStatus =
                    cudaMemcpy(
                        cameras[camera].octaveBaseGPU,
                        cameras[camera].horizontalBufferGPU,
                        nextPixels * sizeof(float),
                        cudaMemcpyDeviceToDevice
                    );

                if(copyStatus != cudaSuccess)
                {
                    cout
                        << "Octave Base Copy Failed : "
                        << cudaGetErrorString(copyStatus)
                        << endl;

                    return false;
                }

                currentImage =
                    cameras[camera].octaveBaseGPU;
            }

            //--------------------------------------------------
            // Update Size
            //--------------------------------------------------

            currentWidth /= 2;
            currentHeight /= 2;
                    }
                }

    return true;
}


//----------------------------------------------------------
//
// Build Difference of Gaussian Pyramid
//
//----------------------------------------------------------

bool buildDoGPyramid(
    Camera cameras[]
)
{
    cout
        << "\n=================================\n";

    cout
        << "Building DoG Pyramid\n";

    cout
        << "=================================\n";

    //------------------------------------------------------
    // Process Every Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cout
            << "\nCamera "
            << camera
            << endl;

        //--------------------------------------------------
        // Process Every Octave
        //--------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            cout
                << "\n  Octave "
                << octave
                << endl;

            //--------------------------------------------------
            // Build DoG Images
            //--------------------------------------------------

            for(int dog = 0;
                dog < NUM_DOG_IMAGES;
                dog++)
            {
                int gaussianIndex1 =
                    octave *
                    NUM_SCALES +
                    dog;

                int gaussianIndex2 =
                    gaussianIndex1 + 1;

                int dogIndex =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                bool dogStatus =
                    dogGPU(
                        cameras[camera].
                        gaussianGPU[
                            gaussianIndex1
                        ],

                        cameras[camera].
                        gaussianGPU[
                            gaussianIndex2
                        ],

                        cameras[camera].
                        dogGPU[
                            dogIndex
                        ],

                        width,
                        height
                    );

                if(!dogStatus)
                {
                    cout
                        << "DoG Generation Failed\n";

                    return false;
                }

                cout
                    << "Camera "
                    << camera
                    << "  Octave "
                    << octave
                    << "  DoG "
                    << dog
                    << " Completed\n";
            }
        }
    }

    return true;
}


//----------------------------------------------------------
//
// Build Gradient Pyramid
//
//----------------------------------------------------------

bool buildGradientPyramid(
    Camera cameras[]
)
{
    cout
        << "\n=================================\n";

    cout
        << "Building Gradient Pyramid\n";

    cout
        << "=================================\n";

    //------------------------------------------------------
    // Every Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cout
            << "\nCamera "
            << camera
            << endl;

        //--------------------------------------------------
        // Every Octave
        //--------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            cout
                << "\n  Octave "
                << octave
                << endl;

            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            //--------------------------------------------------
            // Every Scale
            //--------------------------------------------------

            for(int scale = 0;
                scale < NUM_SCALES;
                scale++)
            {
                int index =
                    octave *
                    NUM_SCALES +
                    scale;

                bool gradientStatus =
                    gradientGPU(
                        cameras[camera].
                        gaussianGPU[index],

                        cameras[camera].
                        magnitudeGPU[index],

                        cameras[camera].
                        orientationGPU[index],

                        width,
                        height
                    );

                if(!gradientStatus)
                {
                    cout
                        << "Gradient Pyramid Failed\n";

                    return false;
                }

                cout
                    << "Camera "
                    << camera
                    << "  Octave "
                    << octave
                    << "  Scale "
                    << scale
                    << " Completed\n";
            }
        }
    }

    return true;
}

//----------------------------------------------------------
//
// Build Extrema Pyramid
//
//----------------------------------------------------------

bool buildExtremaPyramid(
    Camera cameras[]
)
{
    cout
        << "\n=================================\n";

    cout
        << "Building Extrema Pyramid\n";

    cout
        << "=================================\n";

    //------------------------------------------------------
    // Every Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cout
            << "\nCamera "
            << camera
            << endl;

        //--------------------------------------------------
        // Every Octave
        //--------------------------------------------------

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            cout
                << "\n  Octave "
                << octave
                << endl;

            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            //--------------------------------------------------
            // Skip First and Last DoG
            //--------------------------------------------------

            for(int dog = 1;
                dog < NUM_DOG_IMAGES - 1;
                dog++)
            {


                int lowerIndex =
                    octave *
                    NUM_DOG_IMAGES +
                    (dog - 1);

                int currentIndex =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                int upperIndex =
                    octave *
                    NUM_DOG_IMAGES +
                    (dog + 1);

                bool extremaStatus =
                    extremaGPU(

                        cameras[camera].
                        dogGPU[
                            lowerIndex
                        ],

                        cameras[camera].
                        dogGPU[
                            currentIndex
                        ],

                        cameras[camera].
                        dogGPU[
                            upperIndex
                        ],

                        cameras[camera].
                        extremaGPU[
                            currentIndex
                        ],

                        width,
                        height
                    );

                if(!extremaStatus)
                {
                    cout
                        << "Extrema Detection Failed\n";

                    return false;
                }

                std::vector<unsigned char> extremaCPU(width * height);

                cudaMemcpy(
                    extremaCPU.data(),
                    cameras[camera].extremaGPU[currentIndex],
                    width * height * sizeof(unsigned char),
                    cudaMemcpyDeviceToHost
                );

                int extremaCount = 0;

                for(int i = 0; i < width * height; i++)
                {
                    if(extremaCPU[i])
                        extremaCount++;
                }

                cout << "Dog "
                    << dog
                    << " : "
                    << extremaCount
                    << endl;

                cout
                    << "Camera "
                    << camera
                    << "  Octave "
                    << octave
                    << "  DoG "
                    << dog
                    << " Completed\n";
            }
        }
    }

    return true;
}


//----------------------------------------------------------
//
// Build Contrast Pyramid
//
//----------------------------------------------------------

bool buildContrastPyramid(
    Camera cameras[]
)
{
    cout
        << "\n=================================\n";

    cout
        << "Building Contrast Pyramid\n";

    cout
        << "=================================\n";

    //------------------------------------------------------
    // Every Camera
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        cout
            << "\nCamera "
            << camera
            << endl;

        //--------------------------------------------------
        // Every Octave
        //--------------------------------------------------


        int totalExtrema = 0;
        int totalContrast = 0;

        for(int octave = 0;
            octave < NUM_OCTAVES;
            octave++)
        {
            cout
                << "\n  Octave "
                << octave
                << endl;

            int width =
                cameras[camera].
                pyramidWidths[octave];

            int height =
                cameras[camera].
                pyramidHeights[octave];

            //--------------------------------------------------
            // Only Middle DoG Images
            //--------------------------------------------------

            for(int dog = 1;
                dog < NUM_DOG_IMAGES - 1;
                dog++)
            {
                int index =
                    octave *
                    NUM_DOG_IMAGES +
                    dog;

                cout << "Index = "
                << index
                << "  DoG = "
                << dog
                << endl;

                bool contrastStatus =
                    contrastGPU(

                        cameras[camera].
                        dogGPU[index],

                        cameras[camera].
                        extremaGPU[index],

                        cameras[camera].
                        contrastGPU[index],

                        width,
                        height
                    );

                    std::vector<unsigned char> extremaCPU(width * height);
                    std::vector<unsigned char> contrastCPU(width * height);

                    cudaMemcpy(
                        extremaCPU.data(),
                        cameras[camera].extremaGPU[index],
                        width * height,
                        cudaMemcpyDeviceToHost
                    );

                    cudaMemcpy(
                        contrastCPU.data(),
                        cameras[camera].contrastGPU[index],
                        width * height,
                        cudaMemcpyDeviceToHost
                    );

                    int extremaCount = 0;
                    int contrastCount = 0;

                    for(int i = 0; i < width * height; i++)
                    {
                        if(extremaCPU[i]) extremaCount++;
                        if(contrastCPU[i]) contrastCount++;
                    }
                    totalExtrema += extremaCount;
                    totalContrast += contrastCount;

                    cout << "Extrema = " << extremaCount
                        << "   Contrast = " << contrastCount
                        << endl;



                if(!contrastStatus)
                {
                    cout
                        << "Contrast Filtering Failed\n";

                    return false;
                }

                cout
                    << "Camera "
                    << camera
                    << "  Octave "
                    << octave
                    << "  DoG "
                    << dog
                    << " Completed\n";
                    cout << "\n=================================\n";

                    cout << "Camera "
                        << camera
                        << " Summary\n";

                    cout << "=================================\n";

                    cout << "Total Extrema  : "
                        << totalExtrema
                        << endl;

                    cout << "Total Contrast : "
                        << totalContrast
                        << endl;
            }

        }
    }

    return true;
}

//----------------------------------------------------------
// Read Metadata
//----------------------------------------------------------

bool readMetadata(Camera cameras[])
{
    ifstream file("temp/image_info.txt");

    if(!file.is_open())
    {
        cout << "Cannot Open Metadata File" << endl;
        return false;
    }

    int active = 0;

    file >> active;

    for(int i = 0; i < active; i++)
    {
        int id;
        int width;
        int height;

        file >> id;
        file >> width;
        file >> height;

        cameras[id].width  = width;
        cameras[id].height = height;
    }

    file.close();

    return true;
}

//----------------------------------------------------------
// Load Binary Image
//----------------------------------------------------------

bool loadBinaryImage(
    int cameraID,
    Camera cameras[]
)
{
    string filename =
        "temp/image_" +
        to_string(cameraID) +
        ".bin";

    ifstream file(
        filename,
        ios::binary
    );

    if(!file.is_open())
    {
        cout << "Cannot Open "
             << filename
             << endl;

        return false;
    }

    int pixels =
        cameras[cameraID].width *
        cameras[cameraID].height;

    cameras[cameraID].imageCPU.resize(pixels);

    file.read(
        reinterpret_cast<char*>(
            cameras[cameraID].imageCPU.data()
        ),
        pixels * sizeof(float)
    );

    file.close();

    return true;
}

//----------------------------------------------------------
// Main
//----------------------------------------------------------

int main()
{
    cout << "=================================\n";
    cout << "CUDA SIFT VR Stitcher\n";
    cout << "=================================\n\n";

    Camera cameras[MAX_CAMERAS];

        //------------------------------------------------------
    //
    // Generate Sigma Levels
    //
    //------------------------------------------------------





    //------------------------------------------------------
    // Read Metadata
    //------------------------------------------------------

    if(!readMetadata(cameras))
    {
        return -1;
    }

    //------------------------------------------------------
    // Load Images
    //------------------------------------------------------

    for(int i = 0; i < ACTIVE_CAMERAS; i++)
    {
        if(!loadBinaryImage(i, cameras))
        {
            return -1;
        }

        cout << "\nImage Range Camera " << i << endl;

        float minimum = cameras[i].imageCPU[0];
        float maximum = cameras[i].imageCPU[0];

        for(float value : cameras[i].imageCPU)
        {
            minimum = min(minimum, value);
            maximum = max(maximum, value);
        }

        cout << "Min : " << minimum << endl;
        cout << "Max : " << maximum << endl;

        cout << "Camera " << i << endl;
        cout << "Width  : "
             << cameras[i].width
             << endl;

        cout << "Height : "
             << cameras[i].height
             << endl;

        cout << "Pixels : "
             << cameras[i].imageCPU.size()
             << endl << endl;
    }

    cout << "Images Loaded Successfully.\n";
        //------------------------------------------------------
    // Upload Images to GPU
    //------------------------------------------------------

    for(int i = 0; i < ACTIVE_CAMERAS; i++)
    {
        int pixels =
            cameras[i].width *
            cameras[i].height;

        cudaError_t err =
            cudaMalloc(
                (void**)&cameras[i].imageGPU,
                pixels * sizeof(float)
            );
        cudaError_t horizontalBufferStatus =
            cudaMalloc(
                (void**)&cameras[i].horizontalBufferGPU,
                pixels * sizeof(float)
            );

        if(horizontalBufferStatus != cudaSuccess)
        {
            cout
                << "Horizontal Buffer Allocation Failed\n";

            return -1;
        }


        cudaError_t octaveBaseStatus =
            cudaMalloc(
                (void**)&cameras[i].octaveBaseGPU,
                pixels * sizeof(float)
            );

        if(octaveBaseStatus != cudaSuccess)
        {
            cout
                << "Octave Base Allocation Failed\n";

            return -1;
        }

        if(err != cudaSuccess)
        {
            cout << "GPU Allocation Failed : "
                 << cudaGetErrorString(err)
                 << endl;

            return -1;
        }

        //------------------------------------------------------
        // Upload CPU Image To GPU
        //------------------------------------------------------

        err =
            cudaMemcpy(
                cameras[i].imageGPU,
                cameras[i].imageCPU.data(),
                pixels * sizeof(float),
                cudaMemcpyHostToDevice
            );

        if(err != cudaSuccess)
        {
            cout
                << "Image Upload Failed : "
                << cudaGetErrorString(err)
                << endl;

            return -1;
        }

        //------------------------------------------------------
        // Initialize Octave Base
        //------------------------------------------------------

        err =
            cudaMemcpy(
                cameras[i].octaveBaseGPU,
                cameras[i].imageGPU,
                pixels * sizeof(float),
                cudaMemcpyDeviceToDevice
            );

        if(err != cudaSuccess)
        {
            cout
                << "Octave Base Copy Failed : "
                << cudaGetErrorString(err)
                << endl;

            return -1;
        }



        if(err != cudaSuccess)
        {
            cout << "Upload Failed : "
                 << cudaGetErrorString(err)
                 << endl;

            return -1;
        }
    }

    cudaError_t err =
        cudaDeviceSynchronize();

    if(err != cudaSuccess)
    {
        cout << cudaGetErrorString(err)
             << endl;

        return -1;
    }

    cout << "\nImages Uploaded Successfully.\n";


    //------------------------------------------------------
    //
    // Allocate Gaussian Pyramid
    //
    //------------------------------------------------------

    for(int camera = 0;
        camera < ACTIVE_CAMERAS;
        camera++)
    {
        bool allocationSuccess =
            allocateGaussianPyramid(
                cameras[camera]
            );

        if(!allocationSuccess)
        {
            return -1;
        }
    }

    cout << "\nGaussian Pyramid Allocated Successfully.\n";

    //------------------------------------------------------
    //
    // Allocate DoG Pyramid
    //
    //------------------------------------------------------

    bool dogAllocationStatus =
        allocateDoGPyramid(
            cameras
        );

    if(!dogAllocationStatus)
    {
        return -1;
    }

    cout
        << "DoG Pyramid Allocated Successfully.\n";

    //------------------------------------------------------
    //
    // Allocate Gradient Pyramid
    //
    //------------------------------------------------------

    bool gradientAllocationStatus =
        allocateGradientPyramid(
            cameras
        );

    if(!gradientAllocationStatus)
    {
        return -1;
    }

    cout
        << "Gradient Pyramid Allocated Successfully.\n";

    //------------------------------------------------------
    //
    // Allocate Extrema Pyramid
    //
    //------------------------------------------------------

    bool extremaAllocationStatus =
        allocateExtremaPyramid(
            cameras
        );

    if(!extremaAllocationStatus)
    {
        return -1;
    }

    cout
        << "Extrema Pyramid Allocated Successfully.\n";

    //------------------------------------------------------
    //
    // Allocate Contrast Pyramid
    //
    //------------------------------------------------------

    bool contrastAllocationStatus =
        allocateContrastPyramid(
            cameras
        );

    if(!contrastAllocationStatus)
    {
        return -1;
    }

    cout
        << "Contrast Pyramid Allocated Successfully.\n";

    //------------------------------------------------------
    //
    // Allocate Edge Pyramid
    //
    //------------------------------------------------------

    bool edgeAllocationStatus =
        allocateEdgePyramid(
            cameras
        );

    if(!edgeAllocationStatus)
    {
        return -1;
    }

    cout
        << "Edge Pyramid Allocated Successfully.\n";



    //------------------------------------------------------
    //
    // Allocate NMS Pyramid
    //
    //------------------------------------------------------

    bool nmsAllocationStatus =
        allocateNMSPyramid(
            cameras
        );

    if(!nmsAllocationStatus)
    {
        return -1;
    }

    cout
        << "NMS Pyramid Allocated Successfully.\n";


    cudaError_t sticky = cudaGetLastError();

    cout << "Sticky Error : "
        << cudaGetErrorString(sticky)
        << endl;

    cout << "Checkpoint 1" << endl;

    std::vector<float> sigmaLevels =
        generateSigmaLevels();


    cout << "Checkpoint 2" << endl;

    cudaError_t sigerr =
        cudaMemcpyToSymbol(
            d_sigmaLevels,
            sigmaLevels.data(),
            sizeof(float) * NUM_SCALES
        );

    cout << "Checkpoint 3" << endl;
    cout << cudaGetErrorString(sigerr) << endl;

    cout << "Checkpoint 4" << endl;







    bool pyramidBuildStatus =
        buildGaussianPyramid(
            cameras,
            sigmaLevels
        );

    if(!pyramidBuildStatus)
    {
        return -1;
    }

    cout << "Checkpoint 5" << endl;


    //------------------------------------------------------
    //
    // Build DoG Pyramid
    //
    //------------------------------------------------------

    bool dogBuildStatus =
        buildDoGPyramid(
            cameras
        );

    if(!dogBuildStatus)
    {
        return -1;
    }

    cout
        << "\nDoG Pyramid Built Successfully.\n";


    //------------------------------------------------------
    //
    // Build Gradient Pyramid
    //
    //------------------------------------------------------

    bool gradientBuildStatus =
        buildGradientPyramid(
            cameras
        );

    if(!gradientBuildStatus)
    {
        return -1;
    }

    cout
        << "\nGradient Pyramid Built Successfully.\n";

    //------------------------------------------------------
    //
    // Build Extrema Pyramid
    //
    //------------------------------------------------------

    bool extremaBuildStatus =
        buildExtremaPyramid(
            cameras
        );

    if(!extremaBuildStatus)
    {
        return -1;
    }

    cout
        << "\nExtrema Pyramid Built Successfully.\n";

    //------------------------------------------------------
    //
    // Build Contrast Pyramid
    //
    //------------------------------------------------------

    bool contrastBuildStatus =
        buildContrastPyramid(
            cameras
        );

    if(!contrastBuildStatus)
    {
        return -1;
    }

    cout
        << "\nContrast Pyramid Built Successfully.\n";

    //------------------------------------------------------
    //
    // Build Edge Pyramid
    //
    //------------------------------------------------------

    bool edgeBuildStatus =
        buildEdgePyramid(
            cameras
        );

    if(!edgeBuildStatus)
    {
        return -1;
    }

    cout
        << "\nEdge Pyramid Built Successfully.\n";

    //------------------------------------------------------
    //
    // Build NMS Pyramid
    //
    //------------------------------------------------------

    bool nmsBuildStatus =
        buildNMSPyramid(
            cameras
        );

    if(!nmsBuildStatus)
    {
        return -1;
    }

    cout
        << "\nNMS Pyramid Built Successfully.\n";

    cout << "\n=================================\n";
    cout << "Keypoint Loss Diagnosis\n";
    cout << "=================================\n";

    for(int camera = 0; camera < ACTIVE_CAMERAS; camera++)
    {
        cout << "\nCamera " << camera << "\n";
        int totalExtrema = 0, totalContrast = 0, totalEdge = 0;

        for(int octave = 0; octave < NUM_OCTAVES; octave++)
        {
            int width = cameras[camera].pyramidWidths[octave];
            int height = cameras[camera].pyramidHeights[octave];
            int pixels = width * height;

            for(int dog = 1; dog < NUM_DOG_IMAGES - 1; dog++)
            {
                int index = octave * NUM_DOG_IMAGES + dog;

                vector<unsigned char> extremaCPU(pixels), contrastCPU(pixels), edgeCPU(pixels);

                cudaMemcpy(extremaCPU.data(), cameras[camera].extremaGPU[index],
                          pixels, cudaMemcpyDeviceToHost);
                cudaMemcpy(contrastCPU.data(), cameras[camera].contrastGPU[index],
                          pixels, cudaMemcpyDeviceToHost);
                cudaMemcpy(edgeCPU.data(), cameras[camera].edgeGPU[index],
                          pixels, cudaMemcpyDeviceToHost);

                int extremaCount = 0, contrastCount = 0, edgeCount = 0;
                for(int i = 0; i < pixels; i++) {
                    if(extremaCPU[i]) extremaCount++;
                    if(contrastCPU[i]) contrastCount++;
                    if(edgeCPU[i]) edgeCount++;
                }

                totalExtrema += extremaCount;
                totalContrast += contrastCount;
                totalEdge += edgeCount;

                cout << "  Octave " << octave << " DoG " << dog << ": "
                    << "Extrema=" << extremaCount << " Contrast=" << contrastCount
                    << " Edge=" << edgeCount << "\n";
            }
        }

        cout << "Totals: Extrema=" << totalExtrema << " Contrast=" << totalContrast
            << " Edge=" << totalEdge << " NMS=" << cameras[camera].nmsKeypointsCPU.size()
            << " Descriptors=" << cameras[camera].descriptorsCPU.size() << "\n";
    }

    //------------------------------------------------------
    //
    // Build Orientation Assignment
    //
    //------------------------------------------------------

    bool orientationBuildStatus =
        buildOrientationAssignment(
            cameras
        );

    if(!orientationBuildStatus)
    {
        return -1;
    }

    cout
        << "\nOrientation Assignment Completed Successfully.\n";


    //------------------------------------------------------
    //
    // Descriptor Generation
    //
    //------------------------------------------------------

    bool descriptorBuildStatus =
        buildDescriptorGeneration(
            cameras
        );

    if(!descriptorBuildStatus)
    {
        return -1;
    }

    cout
        << "\nDescriptor Generation Completed Successfully.\n";

    bool matchingBuildStatus =
        buildDescriptorMatching(
            cameras
        );

    if(!matchingBuildStatus)
    {
        return -1;
    }

    cout
        << "\nDescriptor Matching Completed Successfully.\n";

    return 0; // Production path: skip diagnostic downloads and test reruns.
}
