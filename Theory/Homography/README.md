# Understanding Feature Matching, Homography, and RANSAC for VR & SLAM

## Introduction

In VR, AR, Visual Odometry, and SLAM systems, one of the most important tasks is estimating how the camera moved between two frames.

A common misconception is that the 128-dimensional SIFT descriptors are directly used to compute the camera motion or homography matrix.

They are not.

Descriptors are only used to establish correspondences between images. Once correspondences are found, all geometric estimation is performed using image coordinates.

This document explains the complete pipeline from feature extraction to RANSAC-based homography estimation.

---

# 1. Detect Keypoints

Suppose we have two consecutive camera frames.

## Frame 1

```text
A = (120, 80)
B = (200, 90)
C = (130,160)
D = (220,170)
```

These points are detected by a feature detector such as:

- SIFT
- SURF
- ORB
- AKAZE

The detector identifies distinctive image locations such as:

- Corners
- Blobs
- Highly textured regions

These points are called keypoints.

---

# 2. Compute Descriptors

For each keypoint, SIFT computes a 128-dimensional descriptor.

Example:

```text
Keypoint A

[0.12, 0.04, 0.33, ... 128 values ...]

Keypoint B

[0.22, 0.18, 0.10, ... 128 values ...]
```

The descriptor summarizes the local image appearance around the keypoint.

Think of it as a fingerprint.

Different image locations should ideally have different fingerprints.

---

# 3. Detect Features in the Next Frame

Now the camera moves slightly.

## Frame 2

```text
A' = (150,100)
B' = (230,110)
C' = (160,180)
D' = (250,190)
```

Again, descriptors are computed.

```text
A'

[0.11, 0.05, 0.32, ...]

B'

[0.30, 0.10, 0.22, ...]
```

---

# 4. Descriptor Matching

For each descriptor in Frame 1, we compare it against all descriptors in Frame 2.

The most common distance measure is Euclidean distance.

```math
d = \sqrt{
(a_1-b_1)^2 +
(a_2-b_2)^2 +
...
(a_{128}-b_{128})^2
}
```

Example:

```text
dist(A, A') = 0.5
dist(A, B') = 8.1
dist(A, C') = 7.4
dist(A, D') = 9.0
```

Since A' has the smallest distance:

```text
A → A'
```

becomes a candidate match.

Repeat for every keypoint.

Result:

```text
A → A'
B → B'
C → C'
D → D'
E → X'   (wrong)
F → Y'   (wrong)
```

---

# 5. Lowe Ratio Test

Not every nearest neighbor is reliable.

For each descriptor:

1. Find the best match.
2. Find the second-best match.

Example:

```text
Best distance = 0.4
Second distance = 0.9
```

Compute:

```text
0.4 / 0.9 = 0.44
```

Accept if:

```text
ratio < 0.75
```

Reject if:

```text
ratio > 0.75
```

Why?

A descriptor should clearly prefer one match.

If two matches look equally good, the descriptor is ambiguous.

After ratio filtering:

```text
50 matches remain
```

But some incorrect matches still survive.

---

# 6. Convert Matches into Coordinate Pairs

At this point descriptors are no longer needed.

We keep only coordinates.

```text
(120,80)  → (150,100)
(200,90)  → (230,110)
(130,160) → (160,180)
(220,170) → (250,190)
```

Suppose:

```text
Total matches   = 50
Correct matches = 40
Wrong matches   = 10
```

Now we need to estimate the geometric transformation.

---

# 7. What is a Homography?

A homography is a 3×3 matrix.

```math
H =
\begin{bmatrix}
h_{11} & h_{12} & h_{13}\\
h_{21} & h_{22} & h_{23}\\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
```

It maps points from Image 1 to Image 2.

Input:

```math
\begin{bmatrix}
x\\
y\\
1
\end{bmatrix}
```

Output:

```math
\begin{bmatrix}
x'\\
y'\\
w
\end{bmatrix}
```

Final coordinates:

```math
\left(
\frac{x'}{w},
\frac{y'}{w}
\right)
```

A homography contains 8 independent unknown parameters.

---

# 8. Why Four Matches Are Enough

Each point correspondence provides two equations.

One correspondence:

```text
(x,y) → (u,v)
```

produces:

```text
Equation 1
Equation 2
```

Therefore:

```text
4 correspondences
×
2 equations
=
8 equations
```

Exactly enough to solve for the 8 unknown homography parameters.

This is why RANSAC samples four matches at a time.

---

# 9. RANSAC Begins

Suppose we have:

```text
50 matches
```

RANSAC randomly selects four matches.

```text
A → A'
B → B'
C → C'
D → D'
```

Using these four pairs:

```text
Compute candidate homography H1
```

---

# 10. Evaluate Candidate Homography

Apply H1 to every source point.

Example:

Actual destination:

```text
(150,100)
```

Predicted destination:

```text
(149.5,100.3)
```

Error:

```math
\sqrt{
(150-149.5)^2 +
(100-100.3)^2
}
```

```math
= 0.58 \text{ pixels}
```

Small error.

This match is an inlier.

Another match:

Predicted:

```text
(300,400)
```

Actual:

```text
(500,100)
```

Error:

```text
360 pixels
```

Large error.

This match is an outlier.

After evaluating all 50 matches:

```text
Inliers = 38
```

Store this result.

---

# 11. Repeat Hundreds of Times

RANSAC repeats the process.

```text
Iteration 1 → 38 inliers
Iteration 2 → 7 inliers
Iteration 3 → 12 inliers
Iteration 4 → 41 inliers
Iteration 5 → 39 inliers
...
Iteration 1000 → 42 inliers
```

The model with the largest inlier count wins.

---

# 12. Why RANSAC Works

Suppose:

```text
40 correct matches
10 wrong matches
```

Probability a random match is correct:

```math
40/50 = 0.8
```

Probability that all four randomly selected matches are correct:

```math
0.8^4
=
0.4096
```

About 41%.

After many iterations, RANSAC is very likely to sample a clean set of correct matches.

That clean set produces the correct homography.

---

# 13. Final Refinement

Many people think RANSAC returns the homography computed from the winning four matches.

That is not what happens.

Suppose the winning model has:

```text
42 inliers
```

OpenCV discards the original random four matches.

Instead it uses:

```text
ALL 42 inliers
```

to recompute the homography.

---

# 14. Why Recompute?

Measurements contain noise.

Example:

True point:

```text
(150,100)
```

Measured point:

```text
(150.4,99.7)
```

Another:

```text
(229.8,110.2)
```

Another:

```text
(159.9,180.5)
```

No observation is perfect.

Using only four points makes the estimate sensitive to noise.

Using all 42 inliers averages out errors.

Result:

```text
More accurate homography
```

---

# 15. Overdetermined System

Each inlier provides two equations.

With 42 inliers:

```text
42 × 2 = 84 equations
```

But homography has only 8 unknowns.

```text
84 equations
8 unknowns
```

There is no exact solution.

Instead we find the homography that minimizes total reprojection error.

Conceptually:

```math
\min_H
\sum_i
\left\|
p_i' - Hp_i
\right\|^2
```

This is solved using least-squares optimization.

---

# Complete Pipeline

```text
Camera Frame 1
        ↓
Feature Detection
        ↓
Descriptor Computation
        ↓
Camera Frame 2
        ↓
Feature Detection
        ↓
Descriptor Computation
        ↓
Descriptor Matching
        ↓
Lowe Ratio Test
        ↓
Coordinate Correspondences
        ↓
RANSAC
        ↓
Best Inlier Set
        ↓
Refine Using All Inliers
        ↓
Final Homography
        ↓
Camera Motion Estimation
```

---

# Key Takeaway

Descriptors answer:

> Which points correspond?

RANSAC answers:

> Which correspondences are geometrically consistent?

Homography estimation answers:

> What transformation explains those consistent correspondences?

In short:

```text
Descriptors find matches
        ↓
RANSAC removes bad matches
        ↓
Homography computes the transformation
```

This pipeline forms the foundation of modern:

- VR Tracking
- AR Tracking
- Visual Odometry
- Structure from Motion (SfM)
- SLAM Systems

---
