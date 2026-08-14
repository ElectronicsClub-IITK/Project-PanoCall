# CUDA SIFT panorama pipeline

This directory contains the runnable CUDA SIFT calibration pipeline. Its CUDA implementation is a normal source file at `src/sift_stitcher.cu`; no notebook is required.

```powershell
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

Inputs belong in `input/camera0.*` and `input/camera1.*`. Generated files go to `output/`, including `panorama.png`, `homography.npy`, and the compact CUDA match payload `cuda/matches.bin`.

For a quick test, copy `examples/camera0.jpeg` and `examples/camera1.jpeg` into `input/`.

Real-time two-camera mode runs SIFT at startup and asynchronously every five seconds. Frames between calibrations reuse the most recent homography:

```powershell
python run_realtime.py --camera0 0 --camera1 1 --target-fps 24 --recalibrate-seconds 5
```

The CUDA build uses three octaves, six scales, concurrent two-camera Gaussian pyramids, and FP16 descriptor matrices with FP32 cuBLAS accumulation.

`run_realtime.py` also builds and loads `src/cuda_panorama_renderer.cu`. It caches inverse-warp geometry and feather weights on the GPU, then fuses bilinear warp and blending into one CUDA kernel per frame. Use `--blend-mode fast` for a fixed 50/50 overlap or `--blend-mode feather --feather-radius 48` for a narrow cached seam that avoids double images across the rest of the overlap.

Periodic homographies are checked for inlier quality, reprojection error, overlap, projected area, canvas size, and drift from the initial fixed-camera calibration. Unsafe updates are rejected while the last stable renderer continues. Use `--duration-seconds 15` to make a recording stop and finalize automatically; `q`, Escape, Ctrl+C, and closing the preview window also stop cleanly.

Windows webcam shortcut:

```powershell
.\run_live_webcams.ps1 -Camera0 0 -Camera1 1 -Width 1280 -Height 720 -BlendMode feather -Rebuild
```

For installation, tuning, and fixed-camera real-time rendering, see the [pipeline guide](../../docs/cuda_sift_pipeline.md).
