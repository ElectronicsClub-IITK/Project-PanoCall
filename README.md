# PanoCall

PanoCall combines a Flutter IMU/VR companion app with reference and CUDA-accelerated panorama-stitching pipelines.

## Repository layout

- `apps/flutter/` — Flutter application, packages, and platform runners.
- `pipelines/cuda_sift/` — production CUDA SIFT calibration and panorama pipeline.
- `pipelines/cpu_reference/` — CPU SIFT reference implementation and sample assets.
- `docs/` — setup guide and algorithm notes.

## Quick start

Run the Flutter app:

```powershell
cd apps/flutter
flutter pub get
flutter run
```

Run the CUDA panorama calibration pipeline:

```powershell
cd pipelines/cuda_sift
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

See the detailed [implementation guide](implementation.md) for prerequisites, required files, inputs, outputs, and real-time usage. The shorter [CUDA pipeline guide](docs/cuda_sift_pipeline.md) and [documentation index](docs/README.md) are also available.
