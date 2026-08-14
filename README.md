# PanoCall

PanoCall combines a Flutter IMU/VR companion app with reference and CUDA-accelerated panorama-stitching pipelines.

## Repository layout

- `lib/`, `android/`, `ios/`, `linux/`, `macos/`, `web/`, `windows/` — Flutter application and platform runners.
- `pipelines/cuda_sift/` — production CUDA SIFT calibration and panorama pipeline.
- `pipelines/cpu_reference/` — CPU SIFT reference implementation and sample assets.
- `docs/` — setup guide and algorithm notes.

## Quick start

Run the Flutter app:

```powershell
flutter pub get
flutter run
```

Run the CUDA panorama calibration pipeline:

```powershell
cd pipelines/cuda_sift
python run_pipeline.py --feature-max-dimension 640 --blend-mode fast
```

See [the CUDA pipeline guide](docs/cuda_sift_pipeline.md) for requirements and real-time rendering guidance, and [the documentation index](docs/README.md) for theory notes.
