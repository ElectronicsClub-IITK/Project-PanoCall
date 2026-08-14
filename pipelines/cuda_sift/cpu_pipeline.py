"""CPU post-processing stage for the CUDA SIFT panorama pipeline.

Performance-sensitive geometry and image operations use OpenCV's compiled
implementations.  The small reference helpers remain available for validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


class PipelineDataError(RuntimeError):
    """Raised when CUDA output cannot safely be consumed by the CPU stage."""


@dataclass(frozen=True)
class PipelineMetrics:
    descriptors_camera0: int
    descriptors_camera1: int
    matches: int
    ransac_inliers: int
    mean_reprojection_error: float
    mean_symmetric_error: float
    timings: dict[str, float]
    homography: np.ndarray


def _read_count(handle, path: Path, label: str) -> int:
    line = handle.readline().strip()
    try:
        count = int(line)
    except ValueError as error:
        raise PipelineDataError(f"Malformed {label} header in {path}: {line!r}") from error
    if count < 0:
        raise PipelineDataError(f"Malformed {label} header in {path}: negative count")
    return count


def read_cuda_descriptors(path: Path) -> list[dict]:
    descriptors = []
    try:
        with path.open(encoding="utf-8") as handle:
            count = _read_count(handle, path, "descriptor")
            for line_number in range(2, count + 2):
                values = handle.readline().split()
                if len(values) < 5:
                    raise PipelineDataError(f"Malformed descriptor at {path}:{line_number}")
                try:
                    descriptors.append({"x": int(values[0]), "y": int(values[1]),
                                        "octave": int(values[2]), "scale": int(values[3]),
                                        "angle": float(values[4])})
                except ValueError as error:
                    raise PipelineDataError(f"Malformed descriptor at {path}:{line_number}") from error
    except OSError as error:
        raise PipelineDataError(f"Cannot read descriptor file {path}: {error}") from error
    return descriptors


def read_cuda_matches(path: Path) -> list[dict]:
    matches = []
    try:
        with path.open(encoding="utf-8") as handle:
            count = _read_count(handle, path, "match")
            for line_number in range(2, count + 2):
                values = handle.readline().split()
                if len(values) < 9:
                    raise PipelineDataError(f"Malformed match at {path}:{line_number}; expected 9 fields")
                try:
                    matches.append({"query_index": int(values[0]), "train_index": int(values[1]),
                                    "distance": float(values[2]), "query_x": int(values[3]),
                                    "query_y": int(values[4]), "query_octave": int(values[5]),
                                    "train_x": int(values[6]), "train_y": int(values[7]),
                                    "train_octave": int(values[8])})
                except ValueError as error:
                    raise PipelineDataError(f"Malformed match at {path}:{line_number}") from error
    except OSError as error:
        raise PipelineDataError(f"Cannot read match file {path}: {error}") from error
    return matches


_COMPACT_MATCH_MAGIC = 0x43534D50
_COMPACT_MATCH_VERSION = 1
_COMPACT_MATCH_DTYPE = np.dtype([
    ("query_x", "<i4"), ("query_y", "<i4"), ("query_octave", "<i4"),
    ("train_x", "<i4"), ("train_y", "<i4"), ("train_octave", "<i4"),
    ("distance", "<f4"),
])


def read_compact_cuda_matches(path: Path, coordinate_scales) -> tuple[int, int, list[tuple], list[tuple], list[tuple]]:
    """Read the binary match payload emitted by ``sift_stitcher.cu``.

    It contains only accepted correspondence geometry, not full 128-value
    descriptors.  This avoids multi-megabyte device downloads and text parsing
    while retaining the identical points used by the RANSAC stage.
    """
    try:
        with path.open("rb") as handle:
            header = np.fromfile(handle, dtype="<u4", count=5)
            if len(header) != 5:
                raise PipelineDataError(f"Truncated compact match header: {path}")
            magic, version, descriptor_count0, descriptor_count1, match_count = map(int, header)
            if magic != _COMPACT_MATCH_MAGIC or version != _COMPACT_MATCH_VERSION:
                raise PipelineDataError(f"Unsupported compact match file: {path}")
            records = np.fromfile(handle, dtype=_COMPACT_MATCH_DTYPE, count=match_count)
    except OSError as error:
        raise PipelineDataError(f"Cannot read compact match file {path}: {error}") from error
    if len(records) != match_count:
        raise PipelineDataError(f"Truncated compact match payload: {path}")
    if match_count < 4:
        raise PipelineDataError(f"At least 4 valid matches are required for RANSAC; found {match_count}")

    kp0 = [None] * descriptor_count0
    kp1 = [None] * descriptor_count1
    matches = []
    scale0_x, scale0_y = coordinate_scales[0]
    scale1_x, scale1_y = coordinate_scales[1]
    for index, record in enumerate(records):
        # A descriptor match's query index is its deterministic record index
        # within this compact payload; RANSAC requires only paired coordinates.
        qx = float(record["query_x"] * (2 ** int(record["query_octave"])) * scale0_x)
        qy = float(record["query_y"] * (2 ** int(record["query_octave"])) * scale0_y)
        tx = float(record["train_x"] * (2 ** int(record["train_octave"])) * scale1_x)
        ty = float(record["train_y"] * (2 ** int(record["train_octave"])) * scale1_y)
        kp0[index] = (qx, qy, int(record["query_octave"]), 0, 0.0)
        kp1[index] = (tx, ty, int(record["train_octave"]), 0, 0.0)
        matches.append((index, index, float(record["distance"])))
    return descriptor_count0, descriptor_count1, kp0[:match_count], kp1[:match_count], matches


def bridge_cuda_to_cpu(camera0_descriptors: list[dict], camera1_descriptors: list[dict],
                       cuda_matches: list[dict],
                       coordinate_scales=((1.0, 1.0), (1.0, 1.0))) -> tuple[list[tuple], list[tuple], list[tuple]]:
    # The fixed notebook establishes that x/y are octave coordinates in this
    # CUDA export; geometry must use full-resolution coordinates.
    def keypoints(descriptors, coordinate_scale):
        scale_x, scale_y = coordinate_scale
        return [(d["x"] * (2 ** d["octave"]) * scale_x,
                 d["y"] * (2 ** d["octave"]) * scale_y,
                 d["octave"], d["scale"], d["angle"]) for d in descriptors]

    kp1 = keypoints(camera0_descriptors, coordinate_scales[0])
    kp2 = keypoints(camera1_descriptors, coordinate_scales[1])
    matches = []
    invalid = []
    for number, match in enumerate(cuda_matches, start=1):
        qi, ti = match["query_index"], match["train_index"]
        expected_q = (match["query_x"] * (2 ** match["query_octave"]) * coordinate_scales[0][0],
                      match["query_y"] * (2 ** match["query_octave"]) * coordinate_scales[0][1])
        expected_t = (match["train_x"] * (2 ** match["train_octave"]) * coordinate_scales[1][0],
                      match["train_y"] * (2 ** match["train_octave"]) * coordinate_scales[1][1])
        coordinates_match = (qi >= 0 and ti >= 0 and qi < len(kp1) and ti < len(kp2)
                             and np.allclose(kp1[qi][:2], expected_q)
                             and np.allclose(kp2[ti][:2], expected_t))
        if not coordinates_match:
            invalid.append(number)
        else:
            matches.append((qi, ti, match["distance"]))
    if invalid:
        sample = ", ".join(map(str, invalid[:5]))
        raise PipelineDataError("CUDA matches do not correspond to the current descriptor files "
                                f"({len(invalid)} invalid; first line(s): {sample}). Re-run CUDA to export all files together.")
    if len(matches) < 4:
        raise PipelineDataError(f"At least 4 valid matches are required for RANSAC; found {len(matches)}")
    return kp1, kp2, matches


def reprojection_error(H, src_pt, dst_pt):
    projected = H @ np.array([src_pt[0], src_pt[1], 1.0])
    if abs(projected[2]) < 1e-10:
        return np.inf
    projected /= projected[2]
    return np.linalg.norm(projected[:2] - dst_pt)


def compute_homography_dlt(src_pts, dst_pts):
    if len(src_pts) < 4:
        raise ValueError("At least 4 point correspondences are required.")
    A = []
    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.extend(([-x, -y, -1, 0, 0, 0, x * u, y * u, u],
                  [0, 0, 0, -x, -y, -1, x * v, y * v, v]))
    _, _, vt = np.linalg.svd(np.array(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise np.linalg.LinAlgError("Degenerate homography")
    return H / H[2, 2]


def custom_ransac_filter(kp1, kp2, matches, iterations=2000, threshold=5.0, seed=0):
    # OpenCV performs the sampling and scoring in native code.  The previous
    # implementation performed thousands of Python SVDs and Python-level
    # reprojections, which dominated the complete pipeline.
    src = np.asarray([kp1[qi][:2] for qi, _, _ in matches], dtype=np.float64)
    dst = np.asarray([kp2[ti][:2] for _, ti, _ in matches], dtype=np.float64)
    cv2.setRNGSeed(seed)
    best_H, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold,
                                     maxIters=iterations, confidence=0.995)
    if best_H is None or mask is None:
        raise PipelineDataError("RANSAC failed to find a homography with at least 4 inliers")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), best_H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    best_inliers = [(qi, ti, float(errors[index]))
                    for index, (qi, ti, _) in enumerate(matches) if mask[index, 0]]
    if len(best_inliers) < 4:
        raise PipelineDataError("RANSAC failed to find a homography with at least 4 inliers")
    return best_inliers, best_H


def refine_homography(kp1, kp2, inliers):
    src = np.asarray([kp1[i][:2] for i, _, _ in inliers], dtype=np.float64)
    dst = np.asarray([kp2[j][:2] for _, j, _ in inliers], dtype=np.float64)
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise PipelineDataError("Could not refine the panorama homography")
    return homography


def symmetric_reprojection_error(H, src_pt, dst_pt):
    forward = reprojection_error(H, src_pt, dst_pt)
    backward = reprojection_error(np.linalg.inv(H), dst_pt, src_pt)
    return forward + backward


def bilinear_sample(image, x, y):
    height, width = image.shape[:2]
    if x < 0 or x >= width - 1 or y < 0 or y >= height - 1:
        return np.zeros(3, dtype=np.float32)
    x0, y0 = int(np.floor(x)), int(np.floor(y)); x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0
    return ((1 - dy) * ((1 - dx) * image[y0, x0] + dx * image[y0, x1]) +
            dy * ((1 - dx) * image[y1, x0] + dx * image[y1, x1])).astype(np.float32)


def inverse_warp(source_image, homography, output_width, output_height):
    return cv2.warpPerspective(source_image, homography, (output_width, output_height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def inverse_warp_with_mask(source_image, homography, output_width, output_height):
    warped = inverse_warp(source_image, homography, output_width, output_height)
    source_mask = np.full(source_image.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(source_mask, homography, (output_width, output_height),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, mask


def transform_image_corners(image, homography):
    height, width = image.shape[:2]
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float64)
    transformed = []
    for x, y in corners:
        point = homography @ np.array([x, y, 1.0]); transformed.append(point[:2] / point[2])
    return np.array(transformed)


def compute_panorama_canvas(transformed_corners, reference_image):
    height, width = reference_image.shape[:2]
    min_x, max_x = int(np.floor(min(transformed_corners[:, 0].min(), 0))), int(np.ceil(max(transformed_corners[:, 0].max(), width)))
    min_y, max_y = int(np.floor(min(transformed_corners[:, 1].min(), 0))), int(np.ceil(max(transformed_corners[:, 1].max(), height)))
    return max_x - min_x, max_y - min_y, -min_x, -min_y


def place_reference_image(reference, width, height, translation_x, translation_y):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = reference.shape[:2]
    canvas[translation_y:translation_y + h, translation_x:translation_x + w] = reference
    return canvas


def create_valid_mask(image):
    return np.any(image != 0, axis=2).astype(np.uint8)


def weighted_blend(warped, reference, warped_mask=None, reference_mask=None):
    if warped_mask is None:
        warped_mask = create_valid_mask(warped)
    if reference_mask is None:
        reference_mask = create_valid_mask(reference)
    warped_mask = (warped_mask != 0).astype(np.uint8)
    reference_mask = (reference_mask != 0).astype(np.uint8)
    warped_distance = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)
    reference_distance = cv2.distanceTransform(reference_mask, cv2.DIST_L2, 3)
    total = warped_distance + reference_distance + 1e-8
    return (warped_distance[..., None] / total[..., None] * warped.astype(np.float32) +
            reference_distance[..., None] / total[..., None] * reference.astype(np.float32)).astype(np.uint8)


class RealtimePanoramaRenderer:
    """Reusable renderer for fixed cameras after one SIFT calibration pass.

    Warp geometry, validity masks, and blend weights are computed once.  Video
    code should keep one instance alive and call ``render`` for every frame.
    """

    def __init__(self, camera0_shape, camera1_shape, homography, *, blend_mode="fast"):
        if blend_mode not in {"fast", "feather"}:
            raise ValueError("blend_mode must be 'fast' or 'feather'")
        self.blend_mode = blend_mode
        self.camera0_shape = tuple(camera0_shape[:2])
        self.camera1_shape = tuple(camera1_shape[:2])
        self.homography = np.asarray(homography, dtype=np.float64)
        h0, w0 = self.camera0_shape
        corners = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]],
                           dtype=np.float64).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, self.homography).reshape(-1, 2)
        h1, w1 = self.camera1_shape
        min_x = int(np.floor(min(transformed[:, 0].min(), 0)))
        max_x = int(np.ceil(max(transformed[:, 0].max(), w1)))
        min_y = int(np.floor(min(transformed[:, 1].min(), 0)))
        max_y = int(np.ceil(max(transformed[:, 1].max(), h1)))
        self.width, self.height = max_x - min_x, max_y - min_y
        self.translation_x, self.translation_y = -min_x, -min_y
        if self.width <= 0 or self.height <= 0:
            raise PipelineDataError(f"Invalid panorama dimensions: {self.width} x {self.height}")
        translation = np.array([[1.0, 0.0, self.translation_x],
                                [0.0, 1.0, self.translation_y],
                                [0.0, 0.0, 1.0]])
        self.panorama_homography = translation @ self.homography

        source_mask = np.full((h0, w0), 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(source_mask, self.panorama_homography,
                                          (self.width, self.height), flags=cv2.INTER_NEAREST)
        reference_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        reference_mask[self.translation_y:self.translation_y + h1,
                       self.translation_x:self.translation_x + w1] = 255
        if blend_mode == "feather":
            warped_weight = cv2.distanceTransform((warped_mask != 0).astype(np.uint8), cv2.DIST_L2, 3)
            reference_weight = cv2.distanceTransform((reference_mask != 0).astype(np.uint8), cv2.DIST_L2, 3)
            total = warped_weight + reference_weight
            self.warped_weight = np.divide(warped_weight, total, out=np.zeros_like(warped_weight), where=total > 0)
            self.reference_weight = np.divide(reference_weight, total, out=np.zeros_like(reference_weight), where=total > 0)
        else:
            warped_roi = warped_mask[self.translation_y:self.translation_y + h1,
                                     self.translation_x:self.translation_x + w1]
            self.overlap_mask = ((warped_roi != 0) * 255).astype(np.uint8)
            self.reference_only_mask = ((warped_roi == 0) * 255).astype(np.uint8)

    def render(self, camera0_frame, camera1_frame):
        if tuple(camera0_frame.shape[:2]) != self.camera0_shape or tuple(camera1_frame.shape[:2]) != self.camera1_shape:
            raise ValueError("Frame dimensions changed after panorama renderer initialization")
        warped = cv2.warpPerspective(camera0_frame, self.panorama_homography,
                                     (self.width, self.height), flags=cv2.INTER_LINEAR)
        h1, w1 = self.camera1_shape
        if self.blend_mode == "fast":
            roi = warped[self.translation_y:self.translation_y + h1,
                         self.translation_x:self.translation_x + w1]
            overlap = cv2.addWeighted(roi, 0.5, camera1_frame, 0.5, 0.0)
            cv2.copyTo(overlap, self.overlap_mask, roi)
            cv2.copyTo(camera1_frame, self.reference_only_mask, roi)
            return warped
        reference = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        reference[self.translation_y:self.translation_y + h1,
                  self.translation_x:self.translation_x + w1] = camera1_frame
        return (warped.astype(np.float32) * self.warped_weight[..., None]
                + reference.astype(np.float32) * self.reference_weight[..., None]).astype(np.uint8)


def stitch_two_cameras(images, cuda_directory: Path, output_path: Path, *, ransac_iterations=2000,
                       ransac_threshold=5.0, seed=0,
                       coordinate_scales=((1.0, 1.0), (1.0, 1.0)), blend_mode="fast"):
    if len(images) != 2:
        raise ValueError("The verified CPU panorama stage currently requires exactly two images.")
    timings = {}
    started = perf_counter()
    descriptor_count0, descriptor_count1, kp1, kp2, matches = read_compact_cuda_matches(
        cuda_directory / "matches.bin", coordinate_scales)
    timings["load_cuda_results"] = perf_counter() - started
    started = perf_counter()
    inliers, _ = custom_ransac_filter(kp1, kp2, matches, ransac_iterations, ransac_threshold, seed)
    H = refine_homography(kp1, kp2, inliers)
    src = np.asarray([kp1[i][:2] for i, _, _ in inliers], dtype=np.float64)
    dst = np.asarray([kp2[j][:2] for _, j, _ in inliers], dtype=np.float64)
    forward = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    backward = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    reprojection = float(np.linalg.norm(forward - dst, axis=1).mean())
    symmetric = float((np.linalg.norm(forward - dst, axis=1) +
                       np.linalg.norm(backward - src, axis=1)).mean())
    timings["homography"] = perf_counter() - started
    started = perf_counter()
    renderer = RealtimePanoramaRenderer(images[0].shape, images[1].shape, H, blend_mode=blend_mode)
    panorama = renderer.render(images[0], images[1])
    timings["warp_and_blend"] = perf_counter() - started
    started = perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panorama) or not output_path.is_file() or output_path.stat().st_size == 0:
        raise PipelineDataError(f"Could not write final panorama to {output_path}")
    timings["encode_output"] = perf_counter() - started
    return PipelineMetrics(descriptor_count0, descriptor_count1, len(matches), len(inliers), reprojection, symmetric,
                           timings, H.copy())
