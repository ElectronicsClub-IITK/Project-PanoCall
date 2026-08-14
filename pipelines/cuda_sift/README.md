# CUDA SIFT panorama pipeline

This directory contains the runnable CUDA SIFT calibration pipeline. Its CUDA implementation is a normal source file at `src/sift_stitcher.cu`; no notebook is required.

```powershell
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

Inputs belong in `input/camera0.*` and `input/camera1.*`. Generated files go to `output/`, including `panorama.png`, `homography.npy`, and the compact CUDA match payload `cuda/matches.bin`.

For a quick test, copy `examples/camera0.jpeg` and `examples/camera1.jpeg` into `input/`.

For installation, tuning, and fixed-camera real-time rendering, see the [pipeline guide](../../docs/cuda_sift_pipeline.md).
