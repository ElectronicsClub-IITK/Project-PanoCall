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
from pathlib import Path
import shutil
import subprocess
import sys

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


def prepare_cuda_inputs(image_paths: list[Path], cuda_work_dir: Path) -> list[np.ndarray]:
    temp = cuda_work_dir / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    images = []
    metadata = []
    for camera, path in enumerate(image_paths):
        colour = cv2.imread(str(path), cv2.IMREAD_COLOR)
        grayscale = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if colour is None or grayscale is None:
            raise PipelineError(f"Cannot load camera {camera} image: {path}")
        grayscale.astype(np.float32).tofile(temp / f"image_{camera}.bin")
        metadata.append((camera, grayscale.shape[1], grayscale.shape[0]))
        images.append(colour)
    with (temp / "image_info.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(images)}\n")
        for camera, width, height in metadata:
            handle.write(f"{camera} {width} {height}\n")
    return images


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8", newline="\n")


def resolve_cuda_executable(args, build_dir: Path) -> Path:
    if args.cuda_executable:
        executable = Path(args.cuda_executable).expanduser().resolve()
        if not executable.is_file():
            raise PipelineError(f"CUDA executable does not exist: {executable}")
        return executable
    executable = build_dir / ("sift_stitcher.exe" if sys.platform == "win32" else "sift_stitcher")
    if executable.is_file() and not args.rebuild_cuda:
        return executable
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise PipelineError("CUDA executable is missing and nvcc is not available. Install CUDA or pass --cuda-executable PATH.")
    source = build_dir / "sift_stitcher.cu"
    extract_cuda_source(source)
    command = [nvcc, str(source), "-o", str(executable)]
    print("Building CUDA SIFT executable:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
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
    args = parser.parse_args()
    if args.active_cameras != ACTIVE_CAMERAS:
        raise PipelineError("Only --active-cameras 2 is enabled: the supplied CUDA matcher and verified CPU accumulator are two-camera implementations.")
    if args.ransac_iterations <= 0 or args.ransac_threshold <= 0:
        raise PipelineError("RANSAC iterations and threshold must be positive.")

    output_dir = args.output_dir.resolve(); cuda_work_dir = output_dir / "cuda"; build_dir = output_dir / "build"
    print("=" * 30 + " PIPELINE START " + "=" * 30)
    image_paths = find_camera_images(args.input_dir.resolve(), args.active_cameras)
    for camera, path in enumerate(image_paths): print(f"Camera {camera} -> {path.name}")
    images = prepare_cuda_inputs(image_paths, cuda_work_dir)
    executable = resolve_cuda_executable(args, build_dir)
    print("Running CUDA SIFT:", executable)
    run_cuda(executable, cuda_work_dir)
    metrics = stitch_two_cameras(images, cuda_work_dir, output_dir / "panorama.png",
                                 ransac_iterations=args.ransac_iterations,
                                 ransac_threshold=args.ransac_threshold, seed=args.seed)
    print("\n" + "=" * 30 + " PIPELINE COMPLETE " + "=" * 30)
    print(f"Cameras        : {args.active_cameras}")
    print(f"Descriptors    : {metrics.descriptors_camera0}, {metrics.descriptors_camera1}")
    print(f"Matches        : {metrics.matches}")
    print(f"RANSAC inliers : {metrics.ransac_inliers}")
    print(f"Reprojection  : {metrics.mean_reprojection_error:.6f}")
    print(f"Symmetric err : {metrics.mean_symmetric_error:.6f}")
    print(f"Panorama      : {output_dir / 'panorama.png'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, PipelineDataError, OSError, cv2.error) as error:
        print(f"PIPELINE FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
