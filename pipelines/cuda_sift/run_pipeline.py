"""One-command CUDA-SIFT to CPU-panorama orchestration.

Example
-------
python run_pipeline.py

Input images must be named input/camera0.<extension> and input/camera1.<extension>.
The current CUDA notebook source and verified CPU stitcher are two-camera code;
the command therefore rejects other active-camera counts instead of silently
running an invalid multi-camera configuration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np

from cpu_pipeline import PipelineDataError, stitch_two_cameras


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_SOURCE = PROJECT_ROOT / "Sift_till_descriptor_matching_FIXED.ipynb"
ACTIVE_CAMERAS = 2
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class PipelineError(RuntimeError):
    pass


def find_camera_images(input_dir: Path, active_cameras: int) -> list[Path]:
    if not input_dir.is_dir():
        raise PipelineError(f"Input directory does not exist: {input_dir}")
    images = []
    for camera in range(active_cameras):
        candidates = [path for path in input_dir.iterdir()
                      if path.is_file() and path.stem.lower() == f"camera{camera}" and path.suffix.lower() in IMAGE_EXTENSIONS]
        if len(candidates) != 1:
            expected = f"camera{camera}" + "{.jpg,.jpeg,.png,.bmp,.tif,.tiff}"
            detail = "none found" if not candidates else "multiple found: " + ", ".join(path.name for path in candidates)
            raise PipelineError(f"Expected exactly one input image named {expected} in {input_dir}; {detail}.")
        image = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
        if image is None:
            raise PipelineError(f"Cannot load camera {camera} image: {candidates[0]}")
        images.append(candidates[0])
    return images


def prepare_cuda_inputs(image_paths: list[Path], cuda_work_dir: Path,
                        feature_max_dimension: int) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    temp = cuda_work_dir / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    images = []
    coordinate_scales = []
    metadata = []
    for camera, path in enumerate(image_paths):
        colour = cv2.imread(str(path), cv2.IMREAD_COLOR)
        grayscale = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if colour is None or grayscale is None:
            raise PipelineError(f"Cannot load camera {camera} image: {path}")
        original_height, original_width = grayscale.shape
        longest_side = max(original_width, original_height)
        if feature_max_dimension and longest_side > feature_max_dimension:
            resize_scale = feature_max_dimension / longest_side
            feature_width = max(1, round(original_width * resize_scale))
            feature_height = max(1, round(original_height * resize_scale))
            grayscale = cv2.resize(grayscale, (feature_width, feature_height), interpolation=cv2.INTER_AREA)
        coordinate_scales.append((original_width / grayscale.shape[1],
                                  original_height / grayscale.shape[0]))
        grayscale.astype(np.float32).tofile(temp / f"image_{camera}.bin")
        metadata.append((camera, grayscale.shape[1], grayscale.shape[0]))
        images.append(colour)
    with (temp / "image_info.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(images)}\n")
        for camera, width, height in metadata:
            handle.write(f"{camera} {width} {height}\n")
    return images, coordinate_scales


def optimize_cuda_source(source: str) -> str:
    """Turn the notebook's teaching/debug CUDA cell into its production form."""
    if "#include <string>" not in source:
        source = source.replace("#include <iostream>\n", "#include <iostream>\n#include <string>\n", 1)
    if "#include <cublas_v2.h>" not in source:
        source = source.replace("#include <cuda_runtime.h>\n",
                                "#include <cuda_runtime.h>\n#include <cublas_v2.h>\n", 1)
    source = source.replace("const float CONTRAST_THRESHOLD = 1.0f;",
                            "#define CONTRAST_THRESHOLD 1.0f", 1)
    # Convert the O(query * train * 128) descriptor loop into a cuBLAS matrix
    # multiplication.  A small reduction kernel then applies Lowe's ratio test.
    kernel_start = source.index("__global__\nvoid descriptorMatchingKernel")
    kernel_end = source.index("//=====================================================\n// BUILD EDGE PYRAMID", kernel_start)
    parallel_matcher = r'''__global__
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

'''
    source = source[:kernel_start] + parallel_matcher + source[kernel_end:]

    matching_start = source.index("bool buildDescriptorMatching(")
    matching_end = source.index("// Generate Sigma Levels", matching_start)
    matching_builder = r'''bool buildDescriptorMatching(Camera cameras[])
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
'''
    source = source[:matching_start] + matching_builder + source[matching_end:]

    # Replace the naive global-memory convolution with cooperatively loaded
    # shared-memory tiles.  This removes dozens of redundant global reads per
    # pixel at the larger SIFT scales.
    blur_start = source.index("__global__\nvoid horizontalGaussianKernel")
    blur_end = source.index("//----------------------------------------------------------\n//\n// Downsample CUDA Kernel", blur_start)
    tiled_blur = r'''__global__
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

'''
    source = source[:blur_start] + tiled_blur + source[blur_end:]

    gaussian_start = source.index("bool gaussianBlurGPU(")
    gaussian_end = source.index("bool downsampleGPU(", gaussian_start)
    gaussian = source[gaussian_start:gaussian_end]
    gaussian = gaussian.replace("dim3 blockSize(16,16);", "dim3 blockSize(32,8);", 1)
    gaussian = gaussian.replace(
        "horizontalGaussianKernel<<<\n        gridSize,\n        blockSize\n    >>>(",
        "const size_t horizontalSharedBytes = blockSize.y * (blockSize.x + 2 * kernelRadius) * sizeof(float);\n\n    horizontalGaussianKernel<<<\n        gridSize,\n        blockSize,\n        horizontalSharedBytes\n    >>>(", 1)
    gaussian = gaussian.replace(
        "verticalGaussianKernel<<<\n        gridSize,\n        blockSize\n    >>>(",
        "const size_t verticalSharedBytes = blockSize.x * (blockSize.y + 2 * kernelRadius) * sizeof(float);\n\n    verticalGaussianKernel<<<\n        gridSize,\n        blockSize,\n        verticalSharedBytes\n    >>>(", 1)
    source = source[:gaussian_start] + gaussian + source[gaussian_end:]

    # Build adjacent Gaussian levels incrementally.  Convolution variances add,
    # so sqrt(sigma_n^2 - sigma_(n-1)^2) produces the requested absolute scale
    # with much smaller kernels than repeatedly blurring the raw octave image.
    pyramid_start = source.index("bool buildGaussianPyramid(")
    pyramid_end = source.index("bool buildDoGPyramid(", pyramid_start)
    pyramid = source[pyramid_start:pyramid_end]
    old_call = """gaussianBlurGPU(
                        currentImage,
                        cameras[camera].horizontalBufferGPU,
                        cameras[camera].gaussianGPU[pyramidIndex],
                        currentWidth,
                        currentHeight,
                        sigmaLevels[scale]
                    )"""
    new_call = """gaussianBlurGPU(
                        scale == 0 ? currentImage : cameras[camera].gaussianGPU[pyramidIndex - 1],
                        cameras[camera].horizontalBufferGPU,
                        cameras[camera].gaussianGPU[pyramidIndex],
                        currentWidth,
                        currentHeight,
                        scale == 0 ? sigmaLevels[0] : sqrtf(
                            sigmaLevels[scale] * sigmaLevels[scale] -
                            sigmaLevels[scale - 1] * sigmaLevels[scale - 1])
                    )"""
    if old_call not in pyramid:
        raise PipelineError("CUDA Gaussian pyramid layout changed; incremental blur could not be applied.")
    pyramid = pyramid.replace(old_call, new_call, 1)
    source = source[:pyramid_start] + pyramid + source[pyramid_end:]

    # NMS was accidentally executed twice with identical inputs.
    duplicate_nms = """         nmsKeypoints =
            performNMS(
                edgeKeypoints,
                10
            );
"""
    source = source.replace(duplicate_nms, "", 1)

    # Required descriptor/match files have already been written here.  The
    # remaining ~1,200 lines download and save diagnostic pyramids, then rerun
    # blur tests; they are inappropriate in the production executable.
    main_start = source.index("int main()")
    completion = source.index('<< "\\nDescriptor Matching Completed Successfully.\\n";', main_start)
    statement_end = source.index(";", completion) + 1
    source = source[:statement_end] + "\n\n    return 0; // Production path: skip diagnostic downloads and test reruns." + source[statement_end:]
    return source


def extract_cuda_source(destination: Path) -> None:
    """Materialize the unchanged CUDA cell from the supplied notebook when needed."""
    if not NOTEBOOK_SOURCE.is_file():
        raise PipelineError(f"CUDA source notebook is missing: {NOTEBOOK_SOURCE}")
    try:
        notebook = json.loads(NOTEBOOK_SOURCE.read_text(encoding="utf-8"))
        source = next("".join(cell.get("source", [])) for cell in notebook["cells"]
                      if "%%writefile sift_stitcher.cu" in "".join(cell.get("source", [])))
    except (KeyError, StopIteration, json.JSONDecodeError) as error:
        raise PipelineError("Could not find the CUDA source cell in the SIFT notebook.") from error
    source = source.replace("%%writefile sift_stitcher.cu\n", "", 1)
    source = optimize_cuda_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.read_text(encoding="utf-8") != source:
        destination.write_text(source, encoding="utf-8", newline="\n")


def cuda_build_environment():
    """Load MSVC's build environment when nvcc cannot already find cl.exe."""
    if sys.platform != "win32" or shutil.which("cl"):
        return None
    visual_studio = Path("C:/Program Files/Microsoft Visual Studio")
    developer_scripts = sorted(visual_studio.glob("*/**/Common7/Tools/VsDevCmd.bat"),
                               key=lambda path: str(path), reverse=True)
    if not developer_scripts:
        raise PipelineError("Visual Studio C++ Build Tools or VsDevCmd.bat were not found.")
    command = f'call "{developer_scripts[0]}" -arch=x64 >nul && set'
    completed = subprocess.run(command, shell=True,
                               executable=os.environ.get("COMSPEC", "C:/Windows/System32/cmd.exe"),
                               text=True, capture_output=True)
    if completed.returncode != 0:
        raise PipelineError("Could not initialize the Visual Studio C++ build environment:\n"
                            + completed.stdout + completed.stderr)
    environment = os.environ.copy()
    path_values = []
    for line in completed.stdout.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            if name.lower() == "path":
                path_values.append(value)
                continue
            for existing_name in list(environment):
                if existing_name != name and existing_name.lower() == name.lower():
                    del environment[existing_name]
            environment[name] = value
    if path_values:
        for existing_name in [name for name in environment if name.lower() == "path"]:
            del environment[existing_name]
        environment["PATH"] = max(path_values,
                                  key=lambda value: ("\\VC\\Tools\\MSVC\\" in value, len(value)))
    return environment


def resolve_cuda_executable(args, build_dir: Path) -> Path:
    build_dir = build_dir.resolve()
    if args.cuda_executable:
        executable = Path(args.cuda_executable).expanduser().resolve()
        if not executable.is_file():
            raise PipelineError(f"CUDA executable does not exist: {executable}")
        return executable
    executable = build_dir / ("sift_stitcher.exe" if sys.platform == "win32" else "sift_stitcher")
    source = build_dir / "sift_stitcher.cu"
    extract_cuda_source(source)
    if (executable.is_file() and not args.rebuild_cuda
            and executable.stat().st_mtime >= source.stat().st_mtime):
        return executable
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise PipelineError("CUDA executable is missing and nvcc is not available. Install CUDA or pass --cuda-executable PATH.")
    command = [nvcc, str(source), "-O3", "--use_fast_math", "-lcublas", "-o", str(executable)]
    print("Building CUDA SIFT executable:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True,
                               env=cuda_build_environment())
    if completed.returncode != 0:
        raise PipelineError("CUDA compilation failed:\n" + completed.stdout + completed.stderr)
    if not executable.is_file():
        raise PipelineError(f"CUDA compilation reported success but did not create {executable}")
    return executable


def run_cuda(executable: Path, cuda_work_dir: Path) -> None:
    expected = ("camera0_descriptors.txt", "camera1_descriptors.txt", "matches.txt")
    for name in expected:
        candidate = cuda_work_dir / name
        if candidate.exists():
            candidate.unlink()
    completed = subprocess.run([str(executable)], cwd=cuda_work_dir, text=True, capture_output=True)
    (cuda_work_dir / "cuda_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cuda_work_dir / "cuda_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise PipelineError(f"CUDA SIFT exited with code {completed.returncode}. See {cuda_work_dir / 'cuda_stdout.log'} and cuda_stderr.log.")
    missing = [name for name in expected if not (cuda_work_dir / name).is_file() or (cuda_work_dir / name).stat().st_size == 0]
    if missing:
        raise PipelineError("CUDA SIFT completed but required output is missing or empty: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CUDA SIFT + CPU panorama pipeline.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--cuda-executable", help="Use an existing CUDA SIFT executable instead of building one.")
    parser.add_argument("--rebuild-cuda", action="store_true", help="Re-extract and recompile the CUDA notebook source.")
    parser.add_argument("--active-cameras", type=int, default=ACTIVE_CAMERAS)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0, help="Fixed default makes RANSAC reproducible.")
    parser.add_argument("--feature-max-dimension", type=int, default=1280,
                        help="Downscale CUDA feature extraction to this longest side; 0 keeps full resolution.")
    parser.add_argument("--blend-mode", choices=("fast", "feather"), default="fast",
                        help="Fast is intended for video; feather gives a smoother one-shot panorama.")
    args = parser.parse_args()
    if args.active_cameras != ACTIVE_CAMERAS:
        raise PipelineError("Only --active-cameras 2 is enabled: the supplied CUDA matcher and verified CPU accumulator are two-camera implementations.")
    if args.ransac_iterations <= 0 or args.ransac_threshold <= 0:
        raise PipelineError("RANSAC iterations and threshold must be positive.")
    if args.feature_max_dimension < 0:
        raise PipelineError("Feature maximum dimension cannot be negative.")

    pipeline_started = perf_counter()
    output_dir = args.output_dir.resolve(); cuda_work_dir = output_dir / "cuda"; build_dir = output_dir / "build"
    print("=" * 30 + " PIPELINE START " + "=" * 30)
    image_paths = find_camera_images(args.input_dir.resolve(), args.active_cameras)
    for camera, path in enumerate(image_paths): print(f"Camera {camera} -> {path.name}")
    prepare_started = perf_counter()
    images, coordinate_scales = prepare_cuda_inputs(image_paths, cuda_work_dir, args.feature_max_dimension)
    prepare_seconds = perf_counter() - prepare_started
    executable = resolve_cuda_executable(args, build_dir)
    print("Running CUDA SIFT:", executable)
    cuda_started = perf_counter()
    run_cuda(executable, cuda_work_dir)
    cuda_seconds = perf_counter() - cuda_started
    cpu_started = perf_counter()
    metrics = stitch_two_cameras(images, cuda_work_dir, output_dir / "panorama.png",
                                 ransac_iterations=args.ransac_iterations,
                                 ransac_threshold=args.ransac_threshold, seed=args.seed,
                                 coordinate_scales=coordinate_scales, blend_mode=args.blend_mode)
    homography_path = output_dir / "homography.npy"
    np.save(homography_path, metrics.homography)
    cpu_seconds = perf_counter() - cpu_started
    print("\n" + "=" * 30 + " PIPELINE COMPLETE " + "=" * 30)
    print(f"Cameras        : {args.active_cameras}")
    print(f"Descriptors    : {metrics.descriptors_camera0}, {metrics.descriptors_camera1}")
    print(f"Matches        : {metrics.matches}")
    print(f"RANSAC inliers : {metrics.ransac_inliers}")
    print(f"Reprojection  : {metrics.mean_reprojection_error:.6f}")
    print(f"Symmetric err : {metrics.mean_symmetric_error:.6f}")
    print(f"Panorama      : {output_dir / 'panorama.png'}")
    print(f"Homography    : {homography_path}")
    print("\nStage timing")
    print(f"Prepare input : {prepare_seconds * 1000:.2f} ms")
    print(f"CUDA SIFT     : {cuda_seconds * 1000:.2f} ms")
    for stage, seconds in metrics.timings.items():
        print(f"{stage:14}: {seconds * 1000:.2f} ms")
    print(f"CPU total     : {cpu_seconds * 1000:.2f} ms")
    print(f"End to end    : {(perf_counter() - pipeline_started) * 1000:.2f} ms")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, PipelineDataError, OSError, cv2.error) as error:
        print(f"PIPELINE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
