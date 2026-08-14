# SIFT From Scratch: Feature Detection, Matching, Homography Estimation and Panorama Stitching

## Overview

This project implements the complete Scale-Invariant Feature Transform (SIFT) pipeline entirely from scratch using Python and NumPy, without relying on OpenCV's built-in SIFT detector, descriptor generator, homography estimation, or RANSAC implementation.

The goal of this project was not merely to reproduce SIFT, but to understand every mathematical and algorithmic component involved in modern computer vision pipelines used in:

* Virtual Reality (VR)
* Augmented Reality (AR)
* Visual Odometry
* Simultaneous Localization and Mapping (SLAM)
* Structure from Motion (SfM)
* Panorama Stitching

The pipeline begins with two input images and ends with a stitched panorama generated using custom feature detection, matching, homography estimation, inverse warping, and blending.

---

# Project Pipeline

Input Images
↓
Gaussian Pyramid
↓
Difference of Gaussian (DoG) Pyramid
↓
Scale-Space Extrema Detection
↓
Contrast Filtering
↓
Edge Response Filtering
↓
Non-Maximum Suppression (NMS)
↓
Orientation Assignment
↓
128-Dimensional SIFT Descriptor
↓
Descriptor Matching
↓
Lowe Ratio Test
↓
Custom DLT Homography
↓
Custom RANSAC
↓
Homography Refinement
↓
Inverse Warping
↓
Panorama Canvas Generation
↓
Feather Blending
↓
Final Panorama

---

# Why Build SIFT From Scratch?

Most implementations simply use:

```python
cv2.SIFT_create()
cv2.findHomography()
cv2.RANSAC
```

While convenient, these functions hide the mathematics and engineering decisions behind the algorithm.

This project was built to understand:

* How scale invariance is achieved
* Why Difference of Gaussian is used
* How keypoints are localized
* How orientation invariance is achieved
* How descriptors are constructed
* How image correspondences are established
* How RANSAC removes incorrect matches
* How homographies are estimated
* How image stitching systems work internally

---

# Stage 1: Scale Space Construction

## Gaussian Pyramid

A Gaussian pyramid is created by repeatedly blurring the image using progressively larger sigma values.

Purpose:

* Detect features at different scales
* Achieve scale invariance
* Capture both fine and coarse image structures

Without a Gaussian pyramid, features would only be detectable at a single image scale.

---

## Difference of Gaussian (DoG)

Instead of using the Laplacian of Gaussian directly, SIFT uses:

DoG = G(i+1) − G(i)

Reasons:

* Computationally efficient
* Excellent approximation of Laplacian of Gaussian
* Detects blob-like structures

The DoG pyramid forms the foundation of keypoint detection.

---

## Octaves and Scales

Initially, the detector was implemented with limited scale levels.

To improve robustness:

* Multiple octaves were introduced
* Multiple scales per octave were used

Benefits:

* Improved scale invariance
* Better keypoint localization
* Increased matching robustness

---

# Stage 2: Keypoint Detection

## Scale Space Extrema Detection

Every DoG pixel is compared against its:

* 8 neighbors in the current scale
* 9 neighbors in the scale above
* 9 neighbors in the scale below

Total comparisons:

26 neighboring samples

Pixels that are local maxima or minima become candidate keypoints.

---

## Contrast Filtering

Many extrema originate from noise.

A contrast threshold is applied to remove unstable responses.

Experimented Thresholds:

0.1 → 6032 keypoints

0.3 → 4302 keypoints

0.5 → 3692 keypoints

0.8 → 3340 keypoints

1.0 → 3181 keypoints

1.5 → 2785 keypoints

This stage dramatically improved keypoint stability.

---

## Edge Response Filtering

Edge-like responses often produce unstable matches.

Using the Hessian matrix:

H = [Dxx Dxy]
[Dxy Dyy]

we compute principal curvature ratios and reject edge-dominated extrema.

Results:

After Contrast Filtering: 6557

After Edge Filtering: 1659

This step significantly improved match quality.

---

## Non-Maximum Suppression (NMS)

Multiple detections often occur around the same structure.

NMS was introduced to:

* Remove duplicate detections
* Retain strongest responses
* Improve descriptor quality

Results:

After Edge Filter: 1659

After NMS: 1560

---

# Stage 3: Orientation Assignment

Each keypoint receives a dominant orientation.

Purpose:

* Achieve rotation invariance
* Allow descriptors to be computed in a canonical coordinate frame

For each keypoint:

* Gradient magnitudes are computed
* Gradient orientations are computed
* Orientation histograms are formed
* Dominant orientation is selected

---

# Stage 4: SIFT Descriptor Generation

## Initial Descriptor Implementation

The initial descriptor implementation produced poor matching performance and low inlier counts.

This revealed that descriptor construction is the most critical component of SIFT.

---

## Trilinear Interpolation

The descriptor was redesigned using trilinear interpolation.

Descriptor Structure:

4 × 4 spatial cells

8 orientation bins

Total:

4 × 4 × 8 = 128 dimensions

Interpolation performed across:

* Row bins
* Column bins
* Orientation bins

Benefits:

* Reduced quantization artifacts
* Improved descriptor stability
* Better matching accuracy

---

## Gaussian Weighting

Pixels closer to the keypoint center receive higher weight.

Benefits:

* Improved robustness
* Reduced influence of noisy boundaries

---

## Descriptor Normalization

Each descriptor undergoes:

1. L2 Normalization
2. Value clipping (0.2 threshold)
3. Renormalization

This improves illumination robustness.

---

# Stage 5: Descriptor Matching

Descriptors are matched using Euclidean distance.

For every descriptor:

* Find nearest neighbor
* Find second nearest neighbor

---

## Lowe Ratio Test

A match is accepted only if:

Best Distance / Second Best Distance < Threshold

Purpose:

* Reject ambiguous matches
* Improve correspondence reliability

Results:

Keypoints Image 1: 1496

Keypoints Image 2: 1515

Matches After Ratio Test: 156

---

# Stage 6: Homography Estimation

## Direct Linear Transform (DLT)

A custom DLT solver was implemented from scratch.

No OpenCV homography functions were used.

Input:

(x,y) ↔ (u,v)

Output:

3×3 Homography Matrix

---

# Stage 7: Custom RANSAC

A complete RANSAC implementation was built from scratch.

For every iteration:

1. Randomly select 4 matches
2. Compute candidate homography
3. Project all matches
4. Compute reprojection error
5. Count inliers
6. Retain best model

---

## Symmetric Reprojection Error

To improve evaluation quality:

Forward Error:
Image A → Image B

Backward Error:
Image B → Image A

Combined into:

Symmetric Reprojection Error

This provides a more reliable estimate of geometric consistency.

---

# Results

Matches: 156

Inliers: 102

Inlier Ratio: 65.4%

Average Reprojection Error: 1.23 px

Average Symmetric Error: 1.74 px

Maximum Error: 4.15 px

Final Homography:

[[ 1.2168  0.0409 -336.65]
[ 0.1590  1.1098  -41.29]
[ 0.00033 -0.00003  1.00 ]]

---

# Stage 8: Inverse Warping

Instead of forward warping, inverse warping was used.

Reason:

Forward warping creates holes.

Inverse warping ensures every destination pixel receives a value.

---

# Stage 9: Panorama Stitching

## Panorama Canvas Generation

A custom panorama canvas was generated using transformed image corners.

Canvas Size:

1057 × 1486

---

## Feather Blending

A custom feather blending algorithm was implemented using distance transforms.

Benefits:

* Smooth transitions
* Reduced seam visibility
* Improved panorama appearance

---

# Challenges Encountered

## Challenge 1: Extremely Low Inlier Counts

Initial inliers:

~11

Cause:

Weak descriptor representation

Solution:

* Improved orientation assignment
* Added trilinear interpolation
* Improved descriptor normalization

Result:

102 inliers

---

## Challenge 2: Duplicate Keypoints

Cause:

Multiple detections around the same structure

Solution:

Non-Maximum Suppression

---

## Challenge 3: Edge-Dominated Responses

Cause:

Unstable keypoints on strong edges

Solution:

Hessian-based edge filtering

---

## Challenge 4: Poor Panorama Alignment

Cause:

Simple warping and blending

Solution:

* Custom inverse warping
* Feather blending

---

# Key Learnings

This project provided a deep understanding of:

* Scale-space theory
* Feature detection
* Descriptor engineering
* Robust matching
* Outlier rejection
* Geometric estimation
* Image warping
* Panorama stitching

Most importantly, it demonstrated how modern VR, AR, SLAM, and visual odometry systems transform raw images into geometric understanding of the world.

---

# Future Work

* Multi-Band Blending
* CUDA Acceleration
* GPU-based Gaussian Pyramid Construction
* GPU-based Descriptor Generation
* Real-Time Video Stitching
* Multi-Image Panorama Stitching
* Visual Odometry Pipeline
* Integration into VR Calling System

---

# Technologies Used

* Python
* NumPy
* OpenCV (Image I/O and Visualization Only)
* Matplotlib
* SciPy

---

# Author

Built as part of a larger VR Calling and Computer Vision research project focused on understanding and implementing foundational vision algorithms from first principles.
