# CUDA SIFT panorama pipeline

This directory contains the runnable CUDA SIFT calibration pipeline.

```powershell
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

Inputs belong in `input/camera0.*` and `input/camera1.*`. Generated files go to `output/`, including `panorama.png` and `homography.npy`.

For installation, tuning, and fixed-camera real-time rendering, see the [pipeline guide](../../docs/cuda_sift_pipeline.md).
