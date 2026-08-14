"""Render two video streams while recalibrating CUDA SIFT every five seconds."""
from __future__ import annotations

import argparse
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from cpu_pipeline import PipelineDataError, estimate_homography_from_cuda
from gpu_renderer import CudaPanoramaRenderer, CudaRendererError
from run_pipeline import (PROJECT_ROOT, PipelineError, prepare_cuda_frames,
                          cuda_build_environment, resolve_cuda_executable, run_cuda)


RENDERER_SOURCE = PROJECT_ROOT / "src" / "cuda_panorama_renderer.cu"


def resolve_renderer_library(args, build_dir: Path) -> Path:
    if args.renderer_dll:
        library = Path(args.renderer_dll).expanduser().resolve()
        if not library.is_file():
            raise PipelineError(f"CUDA renderer library does not exist: {library}")
        return library
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / ("panocall_renderer.dll" if sys.platform == "win32" else "libpanocall_renderer.so")
    if (library.is_file() and not args.rebuild_renderer
            and library.stat().st_mtime >= RENDERER_SOURCE.stat().st_mtime):
        return library
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise PipelineError("nvcc is required to build the CUDA warp/blend renderer.")
    command = [nvcc, str(RENDERER_SOURCE), "-O3", "--use_fast_math", "-shared"]
    if sys.platform == "win32":
        command.extend(["-Xcompiler", "/MD"])
    command.extend(["-o", str(library)])
    print("Building CUDA panorama renderer:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True,
                               env=cuda_build_environment())
    if completed.returncode != 0:
        raise PipelineError("CUDA renderer compilation failed:\n" + completed.stdout + completed.stderr)
    return library


def capture_source(value: str):
    return int(value) if value.isdecimal() else value


def read_pair(camera0, camera1):
    ok0, frame0 = camera0.read()
    ok1, frame1 = camera1.read()
    if not ok0 or not ok1 or frame0 is None or frame1 is None:
        return None
    return frame0, frame1


def calibrate(frames, executable: Path, cuda_work_dir: Path, feature_max_dimension: int,
              ransac_iterations: int, ransac_threshold: float, seed: int):
    _, coordinate_scales = prepare_cuda_frames(
        list(frames), cuda_work_dir, feature_max_dimension)
    run_cuda(executable, cuda_work_dir)
    return estimate_homography_from_cuda(
        cuda_work_dir, ransac_iterations=ransac_iterations,
        ransac_threshold=ransac_threshold, seed=seed,
        coordinate_scales=coordinate_scales)


class PeriodicCalibrator:
    """Single background calibration worker; newer frames are never queued up."""

    def __init__(self, executable, cuda_work_dir, feature_max_dimension,
                 ransac_iterations, ransac_threshold, seed):
        self.executable = executable
        self.cuda_work_dir = cuda_work_dir
        self.feature_max_dimension = feature_max_dimension
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
        self.seed = seed
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.busy = False
        self.thread = threading.Thread(target=self._run, name="cuda-sift-calibrator", daemon=True)
        self.thread.start()

    def submit(self, frame0, frame1) -> bool:
        if self.busy:
            return False
        try:
            self.requests.put_nowait((frame0.copy(), frame1.copy()))
        except queue.Full:
            return False
        self.busy = True
        return True

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def _run(self):
        while not self.stopping.is_set():
            try:
                frames = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                metrics = calibrate(
                    frames, self.executable, self.cuda_work_dir,
                    self.feature_max_dimension, self.ransac_iterations,
                    self.ransac_threshold, self.seed)
                result = (metrics, None)
            except Exception as error:  # Preserve the last good homography.
                result = (None, error)
            try:
                self.results.put_nowait(result)
            except queue.Full:
                pass

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=10.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-camera panorama stream with asynchronous CUDA SIFT recalibration.")
    parser.add_argument("--camera0", default="0", help="Camera index or video path.")
    parser.add_argument("--camera1", default="1", help="Camera index or video path.")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam capture width.")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam capture height.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "realtime")
    parser.add_argument("--output-video", type=Path, help="Optional MP4 output; omitted for lowest latency.")
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--recalibrate-seconds", type=float, default=5.0)
    parser.add_argument("--feature-max-dimension", type=int, default=640)
    parser.add_argument("--blend-mode", choices=("fast", "feather"), default="fast")
    parser.add_argument("--feather-radius", type=float, default=96.0,
                        help="Cached CUDA feather transition radius in pixels.")
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-executable")
    parser.add_argument("--rebuild-cuda", action="store_true")
    parser.add_argument("--renderer-dll", help="Use an existing CUDA renderer DLL/shared library.")
    parser.add_argument("--rebuild-renderer", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if (args.target_fps <= 0 or args.recalibrate_seconds <= 0 or args.width <= 0
            or args.height <= 0 or args.feather_radius <= 0):
        raise PipelineError("FPS, recalibration interval, dimensions, and feather radius must be positive.")

    output_dir = args.output_dir.resolve()
    cuda_work_dir = output_dir / "cuda"
    executable = resolve_cuda_executable(
        SimpleNamespace(cuda_executable=args.cuda_executable, rebuild_cuda=args.rebuild_cuda),
        output_dir / "build")
    renderer_library = resolve_renderer_library(args, output_dir / "build")

    camera0 = cv2.VideoCapture(capture_source(args.camera0))
    camera1 = cv2.VideoCapture(capture_source(args.camera1))
    if not camera0.isOpened() or not camera1.isOpened():
        raise PipelineError("Could not open both camera/video sources.")
    for capture in (camera0, camera1):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.target_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    pair = read_pair(camera0, camera1)
    if pair is None:
        raise PipelineError("Could not read the initial frame pair.")

    print("Running initial CUDA SIFT calibration...")
    metrics = calibrate(pair, executable, cuda_work_dir, args.feature_max_dimension,
                        args.ransac_iterations, args.ransac_threshold, args.seed)
    np.save(output_dir / "homography.npy", metrics.homography)
    renderer = CudaPanoramaRenderer(
        renderer_library, pair[0].shape, pair[1].shape, metrics.homography,
        blend_mode=args.blend_mode, feather_radius=args.feather_radius)
    print(f"Initial calibration: {metrics.matches} matches, "
          f"{metrics.ransac_inliers} inliers. Streaming at {args.target_fps:g} FPS target.")

    calibrator = PeriodicCalibrator(
        executable, cuda_work_dir, args.feature_max_dimension,
        args.ransac_iterations, args.ransac_threshold, args.seed)
    writer = None
    output_size = (renderer.width, renderer.height)
    if args.output_video:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.target_fps, output_size)
        if not writer.isOpened():
            raise PipelineError(f"Could not open output video: {args.output_video}")

    frame_period = 1.0 / args.target_fps
    next_deadline = time.monotonic()
    next_calibration = next_deadline + args.recalibrate_seconds
    frames_rendered = 0
    started = next_deadline
    try:
        while True:
            pair = read_pair(camera0, camera1)
            if pair is None:
                break
            now = time.monotonic()
            if now >= next_calibration and calibrator.submit(*pair):
                next_calibration = now + args.recalibrate_seconds

            update = calibrator.poll()
            if update is not None:
                updated_metrics, error = update
                if error is not None:
                    print(f"Recalibration failed; keeping previous homography: {error}")
                else:
                    replacement = CudaPanoramaRenderer(
                        renderer_library, pair[0].shape, pair[1].shape,
                        updated_metrics.homography, blend_mode=args.blend_mode,
                        feather_radius=args.feather_radius)
                    renderer.close()
                    renderer = replacement
                    np.save(output_dir / "homography.npy", updated_metrics.homography)
                    print(f"Homography updated: {updated_metrics.matches} matches, "
                          f"{updated_metrics.ransac_inliers} inliers")

            panorama = renderer.render(*pair)
            frames_rendered += 1
            elapsed = max(time.monotonic() - started, 1e-6)
            cv2.putText(panorama,
                        f"FPS {frames_rendered / elapsed:.1f} GPU {renderer.last_elapsed_ms:.1f} ms",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            if writer is not None:
                output_frame = panorama if panorama.shape[1::-1] == output_size else cv2.resize(panorama, output_size)
                writer.write(output_frame)
            if not args.no_display:
                cv2.imshow("PanoCall realtime panorama", panorama)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            next_deadline += frame_period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -frame_period:
                next_deadline = time.monotonic()
    finally:
        calibrator.close()
        renderer.close()
        camera0.release()
        camera1.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, PipelineDataError, CudaRendererError, OSError, cv2.error) as error:
        print(f"REALTIME PIPELINE FAILED: {error}")
        raise SystemExit(1)
