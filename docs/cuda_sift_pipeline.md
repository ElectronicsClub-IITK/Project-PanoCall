# Run the Panorama Pipeline

This guide runs the complete pipeline in one command: CUDA SIFT feature matching, homography, blending, and panorama creation.

## 1. Open the pipeline directory

From the repository root, open `pipelines/cuda_sift`. It contains the complete runnable CUDA pipeline:

- `run_pipeline.py`
- `cpu_pipeline.py`
- `src/sift_stitcher.cu`

## 2. Install the one-time requirements

Open PowerShell inside `pipelines/cuda_sift` and run:

```powershell
python -m pip install numpy opencv-python
nvcc --version
```

The second command must show a CUDA version. If it says that `nvcc` is not found, CUDA Toolkit is not installed or not added to PATH.

On Windows, CUDA compilation also needs the Visual Studio C++ Build Tools installed. The runner initializes the MSVC build environment automatically when it compiles the CUDA executable.

## 3. Add the two images

Create an `input` folder inside the project folder. Put exactly two overlapping images in it and name them exactly:

```text
input/
  camera0.jpg
  camera1.jpg
```

`jpg`, `jpeg`, `png`, `bmp`, `tif`, and `tiff` are supported. The two images must show overlapping parts of the same scene.

## 4. Run everything

In PowerShell, from the project folder, run this single command:

```powershell
python run_pipeline.py
```

That is all. The script automatically prepares CUDA inputs, compiles `src/sift_stitcher.cu` when needed, runs descriptor matching, estimates the homography, blends the images, and saves the panorama.

The production build uses shared-memory Gaussian kernels, three octaves, six scales, concurrent two-camera Gaussian pyramids, and FP16 cuBLAS descriptor matching with FP32 accumulation. Feature extraction is capped at a 640-pixel longest side by default while the panorama remains full resolution. Run explicitly with:

```powershell
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

For fixed-camera video, do not run SIFT on every frame. The command saves `output/homography.npy`; create one `RealtimePanoramaRenderer` from `pipelines/cuda_sift/cpu_pipeline.py` with that homography and reuse its `render(camera0_frame, camera1_frame)` method for the stream. Warp masks and blend weights are cached by the renderer. Use `--blend-mode feather` when seam quality matters more than one-shot latency.

The provided persistent runner automates this design and recalibrates asynchronously every five seconds:

```powershell
python run_realtime.py --camera0 0 --camera1 1 --target-fps 24 --recalibrate-seconds 5
```

Live frames use the persistent CUDA renderer in `src/cuda_panorama_renderer.cu`; warp and blending are fused. Cached feather blending can be enabled with `--blend-mode feather --feather-radius 96`. Windows users can launch webcams with `run_live_webcams.ps1`.

## 5. Get the result

When the terminal says `PIPELINE COMPLETE`, open:

```text
output/panorama.png
output/homography.npy
```

The first run can take longer because CUDA code is compiled. Later runs reuse the compiled program automatically; replace only the two images in `input` and run the same command again.

## If an error appears

- `nvcc is not available`: install CUDA Toolkit and reopen PowerShell.
- `Expected exactly one input image`: check that the files are called `camera0` and `camera1`, with one supported image extension each.
- CUDA build error on Windows: install Visual Studio C++ Build Tools, then rerun the command.
