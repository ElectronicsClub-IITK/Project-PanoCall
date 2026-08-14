# PanoCall implementation and run guide

This document explains how to run the current project after the repository restructuring. The primary panorama implementation is a two-camera CUDA SIFT calibration pipeline. It extracts and matches features on the GPU, estimates a homography, then builds a full-resolution panorama on the CPU.

## Repository structure

```text
PanoCall/
|-- apps/
|   `-- flutter/                         Flutter companion application
|-- pipelines/
|   |-- cuda_sift/                       Primary runnable panorama pipeline
|   |   |-- src/sift_stitcher.cu         CUDA SIFT feature extraction and matching
|   |   |-- run_pipeline.py              Command-line entry point and CUDA builder
|   |   |-- cpu_pipeline.py              Homography, blending, and real-time renderer
|   |   |-- examples/camera0.jpeg        Example first-camera image
|   |   `-- examples/camera1.jpeg        Example second-camera image
|   `-- cpu_reference/                   Educational from-scratch Python reference
|       |-- src/sift_from_scratch.py
|       `-- examples/
`-- docs/                                Theory and shorter pipeline notes
```

There are no notebook (`.ipynb`) dependencies. The CUDA code is stored permanently in `pipelines/cuda_sift/src/sift_stitcher.cu` and is compiled directly by the runner.

## What you need to run the CUDA pipeline

Required project files:

| File or folder | Purpose |
|---|---|
| `pipelines/cuda_sift/run_pipeline.py` | Starts the complete pipeline; prepares images, builds CUDA code, and saves results. |
| `pipelines/cuda_sift/src/sift_stitcher.cu` | CUDA SIFT and descriptor matcher. Do not move it without updating the runner. |
| `pipelines/cuda_sift/cpu_pipeline.py` | Reads CUDA matches, estimates the homography, and blends the panorama. |
| `pipelines/cuda_sift/input/camera0.<extension>` | First image from the fixed camera pair. You create this file. |
| `pipelines/cuda_sift/input/camera1.<extension>` | Second image from the fixed camera pair. You create this file. |

System requirements:

- Windows 10/11 with an NVIDIA CUDA-capable GPU and a current NVIDIA driver.
- NVIDIA CUDA Toolkit, with `nvcc` available on `PATH`.
- Visual Studio C++ Build Tools (or Visual Studio with Desktop development with C++) on Windows. `nvcc` uses MSVC to link the executable.
- Python 3.10 or later recommended.
- Python packages: `numpy` and `opencv-python`.

Check the two compiler requirements in PowerShell:

```powershell
nvcc --version
cl
```

`cl` may not be available in an ordinary PowerShell window. That is fine if Visual Studio Build Tools is installed: `run_pipeline.py` attempts to load Visual Studio's build environment automatically. If compilation still fails, open **x64 Native Tools Command Prompt for Visual Studio** and run the pipeline there.

## Run the CUDA panorama pipeline

### 1. Open the pipeline directory

From the repository root:

```powershell
cd pipelines\cuda_sift
```

### 2. Install Python dependencies

```powershell
python -m pip install numpy opencv-python
```

For an isolated environment, create and activate a virtual environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy opencv-python
```

### 3. Add the two camera images

Create an `input` directory in `pipelines/cuda_sift` and place exactly one image for each camera in it:

```text
pipelines/cuda_sift/
`-- input/
    |-- camera0.jpeg
    `-- camera1.jpeg
```

Allowed extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, and `.tiff`. The extension can differ between images, but the file stems must be exactly `camera0` and `camera1`.

For a quick smoke test, copy the included sample pair:

```powershell
New-Item -ItemType Directory -Force input
Copy-Item examples\camera0.jpeg input\camera0.jpeg
Copy-Item examples\camera1.jpeg input\camera1.jpeg
```

The images should be taken from fixed, overlapping cameras viewing the same scene. Strong overlap and stable exposure give the best homography.

### 4. Run calibration and create a panorama

For the normal-quality default:

```powershell
python run_pipeline.py
```

For a quicker calibration pass, suited to interactive testing:

```powershell
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

On the first run, the command compiles `src/sift_stitcher.cu` into `output/build/sift_stitcher.exe`. Later runs reuse the executable as long as the `.cu` source has not changed. To force a rebuild after changing CUDA code:

```powershell
python run_pipeline.py --rebuild-cuda
```

### 5. Read the output

On success, the terminal ends with `PIPELINE COMPLETE`. The generated files are:

| Output | Description |
|---|---|
| `output/panorama.png` | Stitched, full-resolution panorama. |
| `output/homography.npy` | 3x3 camera-to-camera transform. Reuse this for a fixed camera setup. |
| `output/cuda/matches.bin` | Compact binary geometry for accepted CUDA matches, consumed by RANSAC. Full descriptors stay on the GPU and are not exported. |
| `output/cuda/cuda_stdout.log` and `cuda_stderr.log` | CUDA program diagnostics. |
| `output/build/sift_stitcher.exe` | Cached compiled CUDA executable on Windows. |

## Command-line options

```powershell
python run_pipeline.py --help
```

Most useful options:

| Option | Default | Use |
|---|---:|---|
| `--feature-max-dimension 640` | `640` | Downscales only feature extraction; lowers CUDA calibration time while final panorama uses original image resolution. Use `0` to avoid downscaling. |
| `--blend-mode fast` | `fast` | Best option for low-latency output. |
| `--blend-mode feather` | `fast` | Smoother seam for a single panorama, with more CPU work. |
| `--ransac-iterations 2000` | `2000` | Number of robust homography trials. Increase for difficult matches. |
| `--ransac-threshold 5.0` | `5.0` | Inlier threshold in pixels. |
| `--input-dir PATH` | `input` | Use images from another folder. |
| `--output-dir PATH` | `output` | Save generated files elsewhere. |
| `--cuda-executable PATH` | build automatically | Run a prebuilt CUDA executable instead of compiling. |
| `--rebuild-cuda` | off | Force recompilation of `src/sift_stitcher.cu`. |

Example using external input and output folders:

```powershell
python run_pipeline.py --input-dir C:\captures\pair --output-dir C:\captures\result --feature-max-dimension 640 --blend-mode fast
```

## Real-time use with a fixed camera pair

The repository includes `pipelines/cuda_sift/run_realtime.py`. It performs CUDA SIFT once at startup, renders every frame with the cached homography, and starts a background recalibration every five seconds by default:

```powershell
cd pipelines\cuda_sift
python run_realtime.py --camera0 0 --camera1 1 --target-fps 24 --recalibrate-seconds 5
```

Camera arguments accept either numeric device indices or video paths. Press `q` or Escape, press Ctrl+C in the terminal, or close the preview window to stop. Add `--duration-seconds 15` to stop and finalize a recording automatically. Add `--no-display` for headless processing or `--output-video output\realtime.mp4` to record; omitting video output gives lower latency. Recalibration failure or rejection does not interrupt rendering: the last valid homography remains active.

Live warp and blending run through `src/cuda_panorama_renderer.cu`. The renderer retains frame buffers, inverse homography maps, validity masks, and blend weights in GPU memory. Its per-frame path uploads two BGR frames, executes one fused bilinear-warp/blend kernel, and downloads the finished panorama. Available blend modes are:

- `--blend-mode fast`: constant 50/50 weighting in the overlap.
- `--blend-mode feather --feather-radius 48`: a narrow, geometry-cached seam transition. Outside that band the renderer selects one camera instead of mixing the entire overlap, which reduces parallax ghosting. Weight construction runs only when a new homography is accepted.

Every periodic homography must pass minimum inlier count/ratio, reprojection and symmetric-error limits, convex projected-area and camera-overlap checks, a bounded canvas check, and a projected-corner drift check against the initial calibration. The corresponding command-line controls are `--min-inliers`, `--min-inlier-ratio`, `--max-reprojection-error`, `--max-symmetric-error`, and `--max-homography-drift`. For fixed webcams, keep the default 48-pixel drift limit unless the mounts can move.

On Windows, the convenience script checks Python dependencies and launches both webcams:

```powershell
.\run_live_webcams.ps1 -Camera0 0 -Camera1 1 -Width 1280 -Height 720 -TargetFps 24 -RecalibrateSeconds 5 -BlendMode feather -Rebuild
```

The first run builds both the SIFT executable and `panocall_renderer.dll`. Later runs reuse them unless their CUDA sources change.

Do the SIFT/homography calibration once using representative overlapping frames. Then keep the resulting `output/homography.npy` and reuse it while processing video frames. Do not rerun SIFT for every frame: feature extraction and matching are calibration work, not per-frame work for fixed cameras.

`cpu_pipeline.py` provides `RealtimePanoramaRenderer`. It caches the output geometry, transformed masks, and blend weights for a fixed frame size. A minimal integration looks like this:

```python
from pathlib import Path
import cv2
import numpy as np
from cpu_pipeline import RealtimePanoramaRenderer

homography = np.load("output/homography.npy")
camera0_frame = cv2.imread("input/camera0.jpeg")
camera1_frame = cv2.imread("input/camera1.jpeg")

renderer = RealtimePanoramaRenderer(
    camera0_frame.shape,
    camera1_frame.shape,
    homography,
    blend_mode="fast",
)

panorama = renderer.render(camera0_frame, camera1_frame)
cv2.imwrite("output/live_frame.png", panorama)
```

Run that code from `pipelines/cuda_sift`, or adjust the import path if it is placed elsewhere. All later frames passed to `render` must have the same resolution as the frames used to construct the renderer. Recalibrate if camera positions, zoom, focus geometry, or frame resolution changes.

## CPU reference implementation

`pipelines/cpu_reference/src/sift_from_scratch.py` is an educational, from-scratch Python/NumPy SIFT implementation. It is not the production real-time path. It uses the reference images in `pipelines/cpu_reference/examples/` and displays intermediate Matplotlib figures.

Run it as follows:

```powershell
cd ..\cpu_reference
python -m pip install -r requirements.txt
python src\sift_from_scratch.py
```

The script reads `examples/image_1.jpeg` and `examples/image_2.jpeg` relative to the `cpu_reference` directory. Keep that working directory when running it, or edit the two input paths near the top of the script.

## Flutter companion app

The Flutter app is independent of the panorama runner and lives under `apps/flutter`:

```powershell
cd apps\flutter
flutter pub get
flutter run
```

You need a Flutter SDK, plus a selected device or emulator. Check setup with:

```powershell
flutter doctor
```

## Troubleshooting

| Message or symptom | What to check |
|---|---|
| `nvcc is not available` | Install NVIDIA CUDA Toolkit and reopen the terminal. Confirm with `nvcc --version`. |
| CUDA compile/link error mentioning `cl.exe` | Install Visual Studio C++ Build Tools with the C++ workload, then retry from an x64 Native Tools Command Prompt. |
| `Expected exactly one input image` | Ensure `input` contains one supported `camera0.*` and one supported `camera1.*`; remove duplicates with the same stem. |
| No usable matches or RANSAC failure | Use images with more overlap, sharper detail, and less motion/exposure difference. Try the provided samples to verify installation. |
| Panorama is misaligned | Recalibrate using simultaneous frames from fixed cameras. Avoid moving cameras after calibration. |
| Later runs still rebuild CUDA | Check that `output/build/sift_stitcher.exe` can be written and that `src/sift_stitcher.cu` has not been modified. |
| `flutter` is not recognized | Install the Flutter SDK, add its `bin` directory to `PATH`, then run `flutter doctor`. |

## Recommended first run

From the repository root, the shortest verified path is:

```powershell
cd pipelines\cuda_sift
python -m pip install numpy opencv-python
New-Item -ItemType Directory -Force input
Copy-Item examples\camera0.jpeg input\camera0.jpeg
Copy-Item examples\camera1.jpeg input\camera1.jpeg
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

Then open `pipelines/cuda_sift/output/panorama.png`.
