"""CPU post-processing stage for the CUDA SIFT panorama pipeline.

This is a module form of the verified post-processing notebook.  It deliberately
keeps the existing DLT, RANSAC, inverse-warp, and feather-blend algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

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


def bridge_cuda_to_cpu(camera0_descriptors: list[dict], camera1_descriptors: list[dict],
                       cuda_matches: list[dict]) -> tuple[list[tuple], list[tuple], list[tuple]]:
    # The fixed notebook establishes that x/y are octave coordinates in this
    # CUDA export; geometry must use full-resolution coordinates.
    def keypoints(descriptors):
        return [(d["x"] * (2 ** d["octave"]), d["y"] * (2 ** d["octave"]),
                 d["octave"], d["scale"], d["angle"]) for d in descriptors]

    kp1, kp2 = keypoints(camera0_descriptors), keypoints(camera1_descriptors)
    matches = []
    invalid = []
    for number, match in enumerate(cuda_matches, start=1):
        qi, ti = match["query_index"], match["train_index"]
        expected_q = (match["query_x"] * (2 ** match["query_octave"]),
                      match["query_y"] * (2 ** match["query_octave"]))
        expected_t = (match["train_x"] * (2 ** match["train_octave"]),
                      match["train_y"] * (2 ** match["train_octave"]))
        if qi < 0 or ti < 0 or qi >= len(kp1) or ti >= len(kp2) or kp1[qi][:2] != expected_q or kp2[ti][:2] != expected_t:
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
    rng = random.Random(seed)
    best_inliers, best_H = [], None
    for _ in range(iterations):
        sample = rng.sample(matches, 4)
        try:
            H = compute_homography_dlt(np.array([kp1[m[0]][:2] for m in sample]),
                                       np.array([kp2[m[1]][:2] for m in sample]))
        except np.linalg.LinAlgError:
            continue
        inliers = []
        for qi, ti, _ in matches:
            error = reprojection_error(H, np.array(kp1[qi][:2]), np.array(kp2[ti][:2]))
            if error < threshold:
                inliers.append((qi, ti, error))
        if len(inliers) > len(best_inliers):
            best_inliers, best_H = inliers, H
    if best_H is None or len(best_inliers) < 4:
        raise PipelineDataError("RANSAC failed to find a homography with at least 4 inliers")
    return best_inliers, best_H


def refine_homography(kp1, kp2, inliers):
    return compute_homography_dlt(np.array([kp1[i][:2] for i, _, _ in inliers]),
                                  np.array([kp2[j][:2] for _, j, _ in inliers]))


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
    warped = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    inverse = np.linalg.inv(homography)
    for y in range(output_height):
        for x in range(output_width):
            source = inverse @ np.array([x, y, 1.0]); source /= source[2]
            sx, sy = source[:2]
            if 0 <= sx < source_image.shape[1] - 1 and 0 <= sy < source_image.shape[0] - 1:
                warped[y, x] = bilinear_sample(source_image, sx, sy)
    return warped


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


def weighted_blend(warped, reference):
    warped_distance = cv2.distanceTransform(create_valid_mask(warped), cv2.DIST_L2, 3)
    reference_distance = cv2.distanceTransform(create_valid_mask(reference), cv2.DIST_L2, 3)
    total = warped_distance + reference_distance + 1e-8
    return (warped_distance[..., None] / total[..., None] * warped.astype(np.float32) +
            reference_distance[..., None] / total[..., None] * reference.astype(np.float32)).astype(np.uint8)


def stitch_two_cameras(images, cuda_directory: Path, output_path: Path, *, ransac_iterations=2000, ransac_threshold=5.0, seed=0):
    if len(images) != 2:
        raise ValueError("The verified CPU panorama stage currently requires exactly two images.")
    d0 = read_cuda_descriptors(cuda_directory / "camera0_descriptors.txt")
    d1 = read_cuda_descriptors(cuda_directory / "camera1_descriptors.txt")
    raw_matches = read_cuda_matches(cuda_directory / "matches.txt")
    kp1, kp2, matches = bridge_cuda_to_cpu(d0, d1, raw_matches)
    inliers, _ = custom_ransac_filter(kp1, kp2, matches, ransac_iterations, ransac_threshold, seed)
    H = refine_homography(kp1, kp2, inliers)
    reprojection = float(np.mean([reprojection_error(H, np.array(kp1[i][:2]), np.array(kp2[j][:2])) for i, j, _ in inliers]))
    symmetric = float(np.mean([symmetric_reprojection_error(H, np.array(kp1[i][:2]), np.array(kp2[j][:2])) for i, j, _ in inliers]))
    width, height, tx, ty = compute_panorama_canvas(transform_image_corners(images[0], H), images[1])
    if width <= 0 or height <= 0:
        raise PipelineDataError(f"Invalid panorama dimensions: {width} x {height}")
    panorama_H = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]]) @ H
    panorama = weighted_blend(inverse_warp(images[0], panorama_H, width, height),
                              place_reference_image(images[1], width, height, tx, ty))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panorama) or not output_path.is_file() or output_path.stat().st_size == 0:
        raise PipelineDataError(f"Could not write final panorama to {output_path}")
    return PipelineMetrics(len(d0), len(d1), len(matches), len(inliers), reprojection, symmetric)
