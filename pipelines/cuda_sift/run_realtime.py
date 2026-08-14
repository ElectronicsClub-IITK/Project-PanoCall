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
from gpu_renderer import CudaPanoramaRenderer, CudaRendererError, panorama_geometry
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
    # Grab both devices before decoding either frame.  This keeps the two
    # capture instants closer together than two sequential read() calls.
    if not camera0.grab() or not camera1.grab():
        return None
    ok0, frame0 = camera0.retrieve()
    ok1, frame1 = camera1.retrieve()
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


def _project_camera0_corners(homography, camera0_shape):
    height, width = tuple(camera0_shape[:2])
    corners = np.array(((0.0, 0.0), (width - 1.0, 0.0),
                        (width - 1.0, height - 1.0), (0.0, height - 1.0)),
                       dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)


def validate_calibration(metrics, camera0_shape, camera1_shape, *, anchor_homography=None,
                         min_inliers=20, min_inlier_ratio=0.35,
                         max_reprojection_error=6.0, max_symmetric_error=12.0,
                         max_corner_drift=48.0):
    """Reject numerically valid homographies that are unsafe for live output."""
    matrix = np.asarray(metrics.homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise PipelineDataError("homography is not a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        raise PipelineDataError("homography is singular or nearly singular")
    if metrics.ransac_inliers < min_inliers:
        raise PipelineDataError(
            f"only {metrics.ransac_inliers} RANSAC inliers (minimum {min_inliers})")
    inlier_ratio = metrics.ransac_inliers / max(metrics.matches, 1)
    if inlier_ratio < min_inlier_ratio:
        raise PipelineDataError(
            f"inlier ratio {inlier_ratio:.2f} is below {min_inlier_ratio:.2f}")
    if metrics.mean_reprojection_error > max_reprojection_error:
        raise PipelineDataError(
            f"mean reprojection error {metrics.mean_reprojection_error:.2f}px exceeds "
            f"{max_reprojection_error:.2f}px")
    if metrics.mean_symmetric_error > max_symmetric_error:
        raise PipelineDataError(
            f"mean symmetric error {metrics.mean_symmetric_error:.2f}px exceeds "
            f"{max_symmetric_error:.2f}px")

    h0, w0 = tuple(camera0_shape[:2])
    h1, w1 = tuple(camera1_shape[:2])
    source_corners = np.array(((0.0, 0.0, 1.0), (w0 - 1.0, 0.0, 1.0),
                               (w0 - 1.0, h0 - 1.0, 1.0), (0.0, h0 - 1.0, 1.0)),
                              dtype=np.float64)
    denominators = (matrix @ source_corners.T).T[:, 2]
    if np.any(np.abs(denominators) < 1e-9) or np.any(np.sign(denominators) != np.sign(denominators[0])):
        raise PipelineDataError("homography crosses the projective horizon inside camera 0")
    denominator_ratio = float(np.abs(denominators).max() / np.abs(denominators).min())
    if denominator_ratio > 4.0:
        raise PipelineDataError(
            f"perspective denominator varies by {denominator_ratio:.1f}x; transform is unstable")

    projected = _project_camera0_corners(matrix, camera0_shape)
    if not np.isfinite(projected).all():
        raise PipelineDataError("homography projects a camera corner to infinity")
    contour = projected.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise PipelineDataError("homography folds or reverses the camera image")
    signed_area = float(cv2.contourArea(contour, oriented=True))
    if signed_area <= 0:
        raise PipelineDataError("homography mirrors the camera image")
    area_ratio = signed_area / float(w0 * h0)
    if not 0.35 <= area_ratio <= 2.5:
        raise PipelineDataError(f"projected image area ratio is implausible ({area_ratio:.2f})")
    bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(contour)
    del bounding_x, bounding_y
    rectangular_fill = signed_area / max(1.0, float(bounding_width * bounding_height))
    if rectangular_fill < 0.35:
        raise PipelineDataError(
            f"projected quadrilateral is excessively skewed ({rectangular_fill:.0%} rectangular fill)")

    reference = np.array(((0.0, 0.0), (w1 - 1.0, 0.0),
                          (w1 - 1.0, h1 - 1.0), (0.0, h1 - 1.0)), dtype=np.float32)
    overlap_area, _ = cv2.intersectConvexConvex(projected.astype(np.float32), reference)
    overlap_ratio = float(overlap_area) / max(1.0, min(area_ratio * w0 * h0, w1 * h1))
    if overlap_ratio < 0.08:
        raise PipelineDataError(f"camera overlap is too small ({overlap_ratio:.2%})")

    _, _, canvas_width, canvas_height = panorama_geometry(camera0_shape, camera1_shape, matrix)
    if canvas_width > 3 * max(w0, w1) or canvas_height > 3 * max(h0, h1):
        raise PipelineDataError(f"panorama canvas is implausible ({canvas_width}x{canvas_height})")

    corner_drift = 0.0
    if anchor_homography is not None:
        anchor_corners = _project_camera0_corners(anchor_homography, camera0_shape)
        corner_drift = float(np.linalg.norm(projected - anchor_corners, axis=1).max())
        if corner_drift > max_corner_drift:
            raise PipelineDataError(
                f"corner drift {corner_drift:.1f}px exceeds {max_corner_drift:.1f}px; "
                "likely parallax or moving-object matches")
    return inlier_ratio, overlap_ratio, corner_drift


class FixedPanoramaCanvas:
    """Keep display and recording dimensions fixed across recalibrations."""

    def __init__(self, renderer):
        self.minimum_x = renderer.minimum_x
        self.minimum_y = renderer.minimum_y
        self.width = renderer.width
        self.height = renderer.height
        self._output = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def place(self, panorama, renderer):
        if (renderer.minimum_x == self.minimum_x and renderer.minimum_y == self.minimum_y
                and renderer.width == self.width and renderer.height == self.height):
            return panorama
        self._output.fill(0)
        destination_x = renderer.minimum_x - self.minimum_x
        destination_y = renderer.minimum_y - self.minimum_y
        source_x = max(0, -destination_x)
        source_y = max(0, -destination_y)
        destination_x = max(0, destination_x)
        destination_y = max(0, destination_y)
        copy_width = min(renderer.width - source_x, self.width - destination_x)
        copy_height = min(renderer.height - source_y, self.height - destination_y)
        if copy_width > 0 and copy_height > 0:
            self._output[destination_y:destination_y + copy_height,
                         destination_x:destination_x + copy_width] = panorama[
                             source_y:source_y + copy_height, source_x:source_x + copy_width]
        return self._output


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
    parser.add_argument("--duration-seconds", type=float, default=0.0,
                        help="Stop cleanly after this many seconds; 0 records until Q/Ctrl+C.")
    parser.add_argument("--feature-max-dimension", type=int, default=640)
    parser.add_argument("--blend-mode", choices=("fast", "feather"), default="fast")
    parser.add_argument("--feather-radius", type=float, default=48.0,
                        help="Width of the narrow cached CUDA seam transition in pixels.")
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--initial-calibration-attempts", type=int, default=3)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--max-reprojection-error", type=float, default=6.0)
    parser.add_argument("--max-symmetric-error", type=float, default=12.0)
    parser.add_argument("--max-homography-drift", type=float, default=48.0,
                        help="Maximum projected-corner movement from initial fixed-camera calibration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda-executable")
    parser.add_argument("--rebuild-cuda", action="store_true")
    parser.add_argument("--renderer-dll", help="Use an existing CUDA renderer DLL/shared library.")
    parser.add_argument("--rebuild-renderer", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if (args.target_fps <= 0 or args.recalibrate_seconds <= 0 or args.width <= 0
            or args.height <= 0 or args.feather_radius <= 0 or args.duration_seconds < 0
            or args.initial_calibration_attempts < 1 or args.min_inliers < 4
            or not 0 < args.min_inlier_ratio <= 1
            or args.max_reprojection_error <= 0 or args.max_symmetric_error <= 0
            or args.max_homography_drift <= 0):
        raise PipelineError("Invalid capture, blending, duration, or homography quality setting.")

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

    for attempt in range(1, args.initial_calibration_attempts + 1):
        print(f"Running initial CUDA SIFT calibration ({attempt}/{args.initial_calibration_attempts})...")
        try:
            metrics = calibrate(pair, executable, cuda_work_dir, args.feature_max_dimension,
                                args.ransac_iterations, args.ransac_threshold, args.seed)
            inlier_ratio, overlap_ratio, _ = validate_calibration(
                metrics, pair[0].shape, pair[1].shape,
                min_inliers=args.min_inliers, min_inlier_ratio=args.min_inlier_ratio,
                max_reprojection_error=args.max_reprojection_error,
                max_symmetric_error=args.max_symmetric_error,
                max_corner_drift=args.max_homography_drift)
            break
        except PipelineDataError as calibration_error:
            if attempt == args.initial_calibration_attempts:
                raise PipelineError(
                    "No safe initial homography was found. Keep both webcams fixed, point them at "
                    "the same textured scene with at least 25% overlap, and retry. Last rejection: "
                    f"{calibration_error}") from calibration_error
            print(f"Initial calibration rejected: {calibration_error}; retrying with fresh frames.")
            pair = read_pair(camera0, camera1)
            if pair is None:
                raise PipelineError("Could not read frames for another initial calibration attempt.")
    anchor_homography = metrics.homography.copy()
    np.save(output_dir / "homography.npy", metrics.homography)
    renderer = CudaPanoramaRenderer(
        renderer_library, pair[0].shape, pair[1].shape, metrics.homography,
        blend_mode=args.blend_mode, feather_radius=args.feather_radius)
    print(f"Initial calibration: {metrics.matches} matches, "
          f"{metrics.ransac_inliers} inliers ({inlier_ratio:.0%}), "
          f"{overlap_ratio:.0%} overlap. Streaming at {args.target_fps:g} FPS target.")

    calibrator = PeriodicCalibrator(
        executable, cuda_work_dir, args.feature_max_dimension,
        args.ransac_iterations, args.ransac_threshold, args.seed)
    writer = None
    fixed_canvas = FixedPanoramaCanvas(renderer)
    output_size = (fixed_canvas.width, fixed_canvas.height)
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
    stopped_by_interrupt = False
    try:
        while True:
            if args.duration_seconds and time.monotonic() - started >= args.duration_seconds:
                print(f"Recording duration {args.duration_seconds:g}s reached; stopping cleanly.")
                break
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
                    try:
                        update_inlier_ratio, update_overlap_ratio, corner_drift = validate_calibration(
                            updated_metrics, pair[0].shape, pair[1].shape,
                            anchor_homography=anchor_homography,
                            min_inliers=args.min_inliers,
                            min_inlier_ratio=args.min_inlier_ratio,
                            max_reprojection_error=args.max_reprojection_error,
                            max_symmetric_error=args.max_symmetric_error,
                            max_corner_drift=args.max_homography_drift)
                        replacement = CudaPanoramaRenderer(
                            renderer_library, pair[0].shape, pair[1].shape,
                            updated_metrics.homography, blend_mode=args.blend_mode,
                            feather_radius=args.feather_radius)
                    except (PipelineDataError, CudaRendererError, ValueError) as update_error:
                        print(f"Homography rejected; keeping stable calibration: {update_error}")
                    else:
                        renderer.close()
                        renderer = replacement
                        np.save(output_dir / "homography.npy", updated_metrics.homography)
                        print(f"Homography updated: {updated_metrics.matches} matches, "
                              f"{updated_metrics.ransac_inliers} inliers "
                              f"({update_inlier_ratio:.0%}), {update_overlap_ratio:.0%} overlap, "
                              f"{corner_drift:.1f}px drift")

            panorama = fixed_canvas.place(renderer.render(*pair), renderer)
            frames_rendered += 1
            elapsed = max(time.monotonic() - started, 1e-6)
            cv2.putText(panorama,
                        f"FPS {frames_rendered / elapsed:.1f} GPU {renderer.last_elapsed_ms:.1f} ms",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            if writer is not None:
                writer.write(panorama)
            if not args.no_display:
                cv2.imshow("PanoCall realtime panorama", panorama)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty("PanoCall realtime panorama", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

            next_deadline += frame_period
            delay = next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -frame_period:
                next_deadline = time.monotonic()
    except KeyboardInterrupt:
        stopped_by_interrupt = True
        print("Stopping realtime pipeline and finalizing output video...")
    finally:
        calibrator.close()
        renderer.close()
        camera0.release()
        camera1.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # Headless OpenCV builds can expose HighGUI symbols without an
                # implementation.  Video finalization must still succeed.
                pass
    if writer is not None:
        print(f"Finalized output video: {args.output_video.resolve()}")
    if stopped_by_interrupt:
        return 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, PipelineDataError, CudaRendererError, OSError, cv2.error) as error:
        print(f"REALTIME PIPELINE FAILED: {error}")
        raise SystemExit(1)
