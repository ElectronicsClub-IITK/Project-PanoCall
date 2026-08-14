"""Run CUDA SIFT calibration and create a two-camera panorama.

The CUDA implementation lives in ``src/sift_stitcher.cu``.  This script
prepares input images, builds that source when necessary, and runs the CPU
geometry/rendering stage.
"""
from __future__ import annotations

import argparse
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
CUDA_SOURCE = PROJECT_ROOT / "src" / "sift_stitcher.cu"
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
                      if path.is_file() and path.stem.lower() == f"camera{camera}"
                      and path.suffix.lower() in IMAGE_EXTENSIONS]
        if len(candidates) != 1:
            expected = f"camera{camera}" + "{.jpg,.jpeg,.png,.bmp,.tif,.tiff}"
            detail = "none found" if not candidates else "multiple found: " + ", ".join(path.name for path in candidates)
            raise PipelineError(f"Expected exactly one input image named {expected} in {input_dir}; {detail}.")
        images.append(candidates[0])
    return images


def prepare_cuda_inputs(image_paths: list[Path], cuda_work_dir: Path,
                        feature_max_dimension: int) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    temp_dir = cuda_work_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    colour_images = []
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
            scale = feature_max_dimension / longest_side
            feature_width = max(1, round(original_width * scale))
            feature_height = max(1, round(original_height * scale))
            grayscale = cv2.resize(grayscale, (feature_width, feature_height), interpolation=cv2.INTER_AREA)
        coordinate_scales.append((original_width / grayscale.shape[1],
                                  original_height / grayscale.shape[0]))
        grayscale.astype(np.float32).tofile(temp_dir / f"image_{camera}.bin")
        metadata.append((camera, grayscale.shape[1], grayscale.shape[0]))
        colour_images.append(colour)
    with (temp_dir / "image_info.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(colour_images)}\n")
        for camera, width, height in metadata:
            handle.write(f"{camera} {width} {height}\n")
    return colour_images, coordinate_scales


def cuda_build_environment():
    """Load MSVC's build environment when nvcc cannot already find cl.exe."""
    if sys.platform != "win32" or shutil.which("cl"):
        return None
    visual_studio = Path("C:/Program Files/Microsoft Visual Studio")
    scripts = sorted(visual_studio.glob("*/**/Common7/Tools/VsDevCmd.bat"),
                     key=lambda path: str(path), reverse=True)
    if not scripts:
        raise PipelineError("Visual Studio C++ Build Tools or VsDevCmd.bat were not found.")
    command = f'call "{scripts[0]}" -arch=x64 >nul && set'
    completed = subprocess.run(command, shell=True,
                               executable=os.environ.get("COMSPEC", "C:/Windows/System32/cmd.exe"),
                               text=True, capture_output=True)
    if completed.returncode != 0:
        raise PipelineError("Could not initialize the Visual Studio C++ build environment:\n"
                            + completed.stdout + completed.stderr)
    environment = os.environ.copy()
    path_values = []
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
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
    if args.cuda_executable:
        executable = Path(args.cuda_executable).expanduser().resolve()
        if not executable.is_file():
            raise PipelineError(f"CUDA executable does not exist: {executable}")
        return executable
    if not CUDA_SOURCE.is_file():
        raise PipelineError(f"CUDA source is missing: {CUDA_SOURCE}")
    build_dir = build_dir.resolve()
    executable = build_dir / ("sift_stitcher.exe" if sys.platform == "win32" else "sift_stitcher")
    if (executable.is_file() and not args.rebuild_cuda
            and executable.stat().st_mtime >= CUDA_SOURCE.stat().st_mtime):
        return executable
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise PipelineError("CUDA executable is missing and nvcc is not available. Install CUDA or pass --cuda-executable PATH.")
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [nvcc, str(CUDA_SOURCE), "-O3", "--use_fast_math", "-lcublas", "-o", str(executable)]
    print("Building CUDA SIFT executable:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True,
                               env=cuda_build_environment())
    if completed.returncode != 0:
        raise PipelineError("CUDA compilation failed:\n" + completed.stdout + completed.stderr)
    if not executable.is_file():
        raise PipelineError(f"CUDA compilation reported success but did not create {executable}")
    return executable


def run_cuda(executable: Path, cuda_work_dir: Path) -> None:
    expected = ("matches.bin",)
    for name in expected:
        candidate = cuda_work_dir / name
        if candidate.exists():
            candidate.unlink()
    completed = subprocess.run([str(executable)], cwd=cuda_work_dir, text=True, capture_output=True)
    (cuda_work_dir / "cuda_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (cuda_work_dir / "cuda_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise PipelineError(f"CUDA SIFT exited with code {completed.returncode}. See {cuda_work_dir / 'cuda_stdout.log'} and cuda_stderr.log.")
    missing = [name for name in expected
               if not (cuda_work_dir / name).is_file() or (cuda_work_dir / name).stat().st_size == 0]
    if missing:
        raise PipelineError("CUDA SIFT completed but required output is missing or empty: " + ", ".join(missing))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CUDA SIFT calibration and create a panorama.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--cuda-executable", help="Use an existing CUDA executable instead of building one.")
    parser.add_argument("--rebuild-cuda", action="store_true", help="Recompile src/sift_stitcher.cu.")
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
        raise PipelineError("Only --active-cameras 2 is enabled by the matcher and renderer.")
    if args.ransac_iterations <= 0 or args.ransac_threshold <= 0:
        raise PipelineError("RANSAC iterations and threshold must be positive.")
    if args.feature_max_dimension < 0:
        raise PipelineError("Feature maximum dimension cannot be negative.")

    pipeline_started = perf_counter()
    output_dir = args.output_dir.resolve()
    cuda_work_dir = output_dir / "cuda"
    build_dir = output_dir / "build"
    print("=" * 30 + " PIPELINE START " + "=" * 30)
    image_paths = find_camera_images(args.input_dir.resolve(), args.active_cameras)
    for camera, path in enumerate(image_paths):
        print(f"Camera {camera} -> {path.name}")
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
