# Understanding SIFT From Scratch: A Complete Walkthrough

## Introduction

This repository implements a simplified version of the SIFT (Scale Invariant Feature Transform) pipeline from scratch using Python, NumPy, and OpenCV.

Most tutorials either:

- Explain only the mathematics
- Show only the code
- Skip important intuition

The purpose of this document is different.

We will follow the exact journey of a pixel through the pipeline and understand:

- Why each stage exists
- What problem it solves
- How the mathematics works
- How the code implements the mathematics

By the end of this document you should understand not only **what SIFT does**, but also **why it does it**.

---

# The Problem We Are Trying To Solve

Suppose we take two photographs of the same scene.

```text
Image A
```

and

```text
Image B
```

The camera may have:

- Rotated
- Zoomed
- Moved closer
- Moved farther
- Changed viewpoint
- Experienced different lighting

Humans instantly recognize that:

```text
This rock
```

in Image A is the same rock in Image B.

Computers cannot.

To a computer, an image is only a matrix of numbers.

The challenge is:

> How can a computer recognize the same physical object in two different images?

This is exactly what SIFT was designed to solve.

---

# What Makes A Good Feature?

Consider two image regions.

## Example 1

```text
Blue Sky
```

Every pixel looks similar.

If I crop a small patch of sky and ask:

> Where does this patch appear in another image?

There is no reliable answer.

This is a bad feature.

---

## Example 2

```text
Building Corner
```

A corner is unique.

If we find the same corner elsewhere, we can confidently identify it.

This is a good feature.

---

A useful feature should be:

- Distinctive
- Repeatable
- Robust to rotation
- Robust to scale changes
- Robust to illumination changes

SIFT was specifically designed to find such points.

---

# Complete Pipeline

The implementation follows the pipeline below:

```text
Input Image
      ↓
Gaussian Pyramid
      ↓
Difference of Gaussian (DoG)
      ↓
Scale Space Extrema Detection
      ↓
Contrast Filter
      ↓
Edge Filter
      ↓
Non Maximum Suppression
      ↓
Orientation Assignment
      ↓
Descriptor Generation
```

Every stage removes bad candidates and keeps only stable and distinctive keypoints.

# Step 1: Gaussian Pyramid

## Why Do We Need Multiple Scales?

Imagine looking at a building.

From very close:

```text
Large Building
```

From very far:

```text
Small Building
```

The same object appears at different sizes.

If we only search at one scale, we may completely miss the object.

Therefore SIFT must be scale invariant.

---

## The Core Idea

Instead of physically moving the camera backward and forward, we simulate distance by blurring the image.

Small blur:

```text
Object appears close
```

Large blur:

```text
Object appears far away
```

By observing the image at many blur levels, we can detect structures that remain stable across scales.

---

## Gaussian Blur

A Gaussian blur replaces each pixel with a weighted average of nearby pixels.

Mathematically:

```math
L(x,y,\sigma)
=
G(x,y,\sigma)
*
I(x,y)
```

where:

```text
I(x,y)
=
Original Image

G(x,y,σ)
=
Gaussian Kernel

L(x,y,σ)
=
Blurred Image
```

The larger the value of σ:

```text
More Blur
```

The smaller the value of σ:

```text
Less Blur
```

---

## Implementation

The code uses:

```python
SIGMAS = [
    1.6,
    2.0,
    2.52,
    3.17,
    4.0,
    5.04,
    6.35,
    8.0
]
```

For each sigma:

```python
blurred = cv2.GaussianBlur(
    current,
    (0,0),
    sigma
)
```

This generates multiple blurred versions of the same image.

---

## What Is An Octave?

After creating several scales, the image is downsampled:

```python
current = cv2.resize(
    current,
    (
        current.shape[1] // 2,
        current.shape[0] // 2
    ),
    interpolation=cv2.INTER_AREA
)
```

This creates a new octave.

Example:

```text
Octave 0
1920 × 1080

Octave 1
960 × 540

Octave 2
480 × 270

Octave 3
240 × 135
```

Each octave contains the same sigma values but at a lower resolution.

---

## Why Use Octaves?

Suppose a feature becomes extremely large.

Rather than using huge Gaussian kernels on the original image, it is computationally cheaper to shrink the image itself.

This dramatically reduces computation while preserving scale information.

---

## Output Structure

The pyramid is stored as:

```python
pyramid[octave][scale]
```

Example:

```text
pyramid[0][0]
=
Octave 0
Sigma 1.6

pyramid[0][1]
=
Octave 0
Sigma 2.0

...

pyramid[1][0]
=
Octave 1
Sigma 1.6
```

The Gaussian Pyramid forms the foundation of the entire SIFT algorithm.

# Step 2: Difference of Gaussian (DoG)

Now we have a Gaussian Pyramid containing many blurred versions of the image.

The question is:

> How do we find interesting structures inside these blurred images?

This is where the Difference of Gaussian (DoG) Pyramid comes in.

---

## Why Not Detect Features Directly On The Image?

Consider a completely flat region:

```text
██████████████
██████████████
██████████████
██████████████
```

Nothing interesting exists here.

Now consider a corner:

```text
██████
██████
██
██
██
```

This corner stands out.

A good feature detector should suppress:

```text
Flat Regions
```

and highlight:

```text
Corners
Blobs
Distinctive Structures
```

---

## The Core Idea

Suppose we have two blurred images:

```text
σ = 2.0
```

and

```text
σ = 2.52
```

Both represent the same image.

The second image is simply more blurred.

Now subtract them:

```math
DoG
=
G(x,y,\sigma_2)
-
G(x,y,\sigma_1)
```

or in code:

```python
dog = octave[i+1] - octave[i]
```

---

## What Happens During Subtraction?

### Flat Regions

Suppose an area is completely uniform.

Image 1:

```text
100 100 100
100 100 100
100 100 100
```

Image 2:

```text
100 100 100
100 100 100
100 100 100
```

Subtract:

```text
0 0 0
0 0 0
0 0 0
```

Nothing remains.

---

### Strong Structures

Suppose a corner exists.

Blur level 1:

```text
100 100 100
100 255 255
100 255 255
```

Blur level 2:

```text
120 130 140
130 180 190
140 190 200
```

Subtract:

```text
-20 -30 -40
-30  75  65
-40  65  55
```

Large values survive.

The corner becomes visible.

---

## Intuition

Think of DoG as asking:

> Which structures changed significantly when I increased the blur?

If a structure survives multiple blur levels:

```text
Important Feature
```

If it disappears immediately:

```text
Noise
```

---

## Why Not Use Laplacian of Gaussian Directly?

The mathematically ideal detector is:

```text
Laplacian of Gaussian (LoG)
```

which measures:

```text
Rapid intensity changes
```

and finds blob-like structures.

Unfortunately LoG is computationally expensive.

---

David Lowe discovered something important:

```text
Difference of Gaussian
```

is an excellent approximation of:

```text
Laplacian of Gaussian
```

while being much faster.

Therefore SIFT uses:

```text
DoG
```

instead of:

```text
LoG
```

---

## Building The DoG Pyramid

Your Gaussian Pyramid contains:

```text
Octave 0

σ1
σ2
σ3
σ4
σ5
σ6
σ7
σ8
```

The DoG Pyramid is built by subtracting neighboring scales:

```text
DoG1 = σ2 - σ1
DoG2 = σ3 - σ2
DoG3 = σ4 - σ3
DoG4 = σ5 - σ4
DoG5 = σ6 - σ5
DoG6 = σ7 - σ6
DoG7 = σ8 - σ7
```

So:

```text
8 Gaussian Images
↓
7 DoG Images
```

---

## Code Walkthrough

The implementation:

```python
for i in range(len(octave)-1):

    dog = (
        octave[i+1]
        -
        octave[i]
    )

    dog_octave.append(dog)
```

For every pair of neighboring Gaussian images:

1. Subtract them
2. Store the result
3. Build a DoG pyramid

---

## Output Structure

The DoG Pyramid is stored as:

```python
dog_pyramid[octave][dog_level]
```

Example:

```text
dog_pyramid[0][0]

=
σ2 - σ1
```

```text
dog_pyramid[0][1]

=
σ3 - σ2
```

and so on.

---

## Visual Interpretation

Bright pixels:

```text
Positive Response
```

Dark pixels:

```text
Negative Response
```

Gray pixels:

```text
Near Zero Response
```

A strong feature usually appears as:

```text
Bright Blob
```

or

```text
Dark Blob
```

inside the DoG image.

These blobs become candidates for keypoints.

---

## Why This Step Is Important

Without the DoG Pyramid:

```text
Millions of Pixels
```

would need to be examined.

After DoG:

```text
Only Regions With Strong Change
```

remain.

This dramatically narrows the search space.

---

## Summary

Input:

```text
Gaussian Pyramid
```

Operation:

```text
Subtract Neighboring Scales
```

Output:

```text
Difference of Gaussian Pyramid
```

Purpose:

```text
Suppress Flat Regions

Highlight Stable Structures

Approximate Laplacian of Gaussian

Prepare For Scale-Space Extrema Detection
```

The DoG Pyramid is where SIFT begins transforming an image into a collection of candidate keypoints.

# Step 3: Scale Space Extrema Detection

At this point we have:

```text
Input Image
      ↓
Gaussian Pyramid
      ↓
Difference of Gaussian Pyramid
```

The Difference of Gaussian images contain many bright and dark blobs.

The next question is:

> Which of these blobs correspond to actual keypoints?

SIFT answers this using a process called:

```text
Scale Space Extrema Detection
```

---

## What Is An Extremum?

An extremum is simply:

```text
A Local Maximum
```

or

```text
A Local Minimum
```

---

Example:

```text
10 12 11
13 25 12
11 10  9
```

The center value:

```text
25
```

is larger than every surrounding value.

Therefore:

```text
25
```

is a local maximum.

---

Similarly:

```text
10 12 11
13 -8 12
11 10  9
```

The center value:

```text
-8
```

is smaller than every surrounding value.

Therefore:

```text
-8
```

is a local minimum.

---

SIFT considers both:

```text
Maxima
```

and

```text
Minima
```

as candidate keypoints.

---

## Why Not Search In A Single Image?

Suppose we only search inside one DoG image.

We might find:

```text
A bright blob
```

But how do we know it is stable across scale?

Maybe it only exists at:

```text
σ = 2.0
```

and completely disappears at:

```text
σ = 2.52
```

Such features are unstable.

We want features that remain distinctive across multiple scales.

Therefore we search in:

```text
X Direction
Y Direction
Scale Direction
```

simultaneously.

---

## Thinking In 3D

Most people imagine images as:

```text
2D
```

objects.

SIFT thinks differently.

Imagine stacking all DoG images on top of each other.

```text
Scale 5
──────────

Scale 4
──────────

Scale 3
──────────

Scale 2
──────────

Scale 1
──────────
```

Now we have a:

```text
3D Volume
```

instead of a single image.

The dimensions are:

```text
X
Y
Scale
```

This structure is called:

```text
Scale Space
```

---

## The Famous 26 Neighbors

Suppose we are examining one pixel.

Current scale:

```text
3 × 3 neighborhood
```

```text
A B C
D X E
F G H
```

where:

```text
X
```

is the center pixel.

---

Inside the same scale there are:

```text
8 neighbors
```

---

Now examine:

```text
Scale Above
```

```text
3 × 3
```

which contributes:

```text
9 neighbors
```

---

Then examine:

```text
Scale Below
```

another:

```text
9 neighbors
```

---

Total:

```text
9
+
9
+
9
=
27
```

Remove the center pixel itself:

```text
27 - 1
=
26 neighbors
```

This is why SIFT always talks about:

```text
26 Neighbor Comparison
```

---

## Visualizing The Search

Imagine a cube.

```text
Previous Scale

● ● ●
● ● ●
● ● ●

Current Scale

● ● ●
● X ●
● ● ●

Next Scale

● ● ●
● ● ●
● ● ●
```

The center pixel:

```text
X
```

must be compared against:

```text
26 surrounding values
```

---

## The Decision Rule

Suppose:

```text
Center Value = 45
```

Neighbor values:

```text
12
14
20
18
7
11
...
```

If:

```text
45
```

is larger than every neighbor:

```text
Local Maximum
```

---

Suppose:

```text
Center Value = -30
```

and all neighbors are larger.

Then:

```text
Local Minimum
```

---

Either case becomes:

```text
Candidate Keypoint
```

---

## Code Walkthrough

The code loops through every octave:

```python
for octave_idx in range(
    len(dog_pyramid)
):
```

---

Then every scale:

```python
for scale_idx in range(
    1,
    len(octave)-1
):
```

Notice:

```text
Start = 1
End = len(octave)-1
```

---

Why?

Because:

```text
First Scale
```

has no scale below it.

and

```text
Last Scale
```

has no scale above it.

Therefore they cannot perform a full:

```text
3 × 3 × 3
```

comparison.

---

## Collecting Neighbors

Previous scale:

```python
neighbors.extend(
    prev_img[
        y-1:y+2,
        x-1:x+2
    ].flatten()
)
```

adds:

```text
9 neighbors
```

---

Current scale:

```python
neighbors.extend(
    curr_img[
        y-1:y+2,
        x-1:x+2
    ].flatten()
)
```

adds:

```text
9 neighbors
```

---

Next scale:

```python
neighbors.extend(
    next_img[
        y-1:y+2,
        x-1:x+2
    ].flatten()
)
```

adds:

```text
9 neighbors
```

---

Total:

```text
27 values
```

---

Then:

```python
neighbors.remove(value)
```

removes the center pixel.

Now:

```text
26 neighbors remain
```

---

## Extremum Test

The actual test:

```python
if (
    value > max(neighbors)
    or
    value < min(neighbors)
):
```

means:

```text
Greater Than Everyone
```

or

```text
Smaller Than Everyone
```

---

If true:

```python
keypoints.append(
(
x,
y,
octave_idx,
scale_idx
)
)
```

The point is stored.

---

## Why This Gives Scale Invariance

Suppose a building corner appears:

```text
Small
```

in one image.

and:

```text
Large
```

in another image.

The corner will create an extremum at different scales.

---

Example:

Image A:

```text
Scale 2
```

produces strongest response.

---

Image B:

```text
Scale 5
```

produces strongest response.

---

Because SIFT searches across:

```text
Scale Space
```

it detects the feature in both cases.

This is the main reason SIFT is:

```text
Scale Invariant
```

---

## What Comes Out Of This Step?

The output format is:

```python
(
x,
y,
octave,
scale
)
```

Example:

```text
(125,87,1,3)

(420,211,0,4)

(72,35,2,5)
```

Each tuple means:

```text
Feature Found

At Position (x,y)

Inside Octave

At Specific Scale
```

---

## Why We Cannot Stop Here

The extrema detector finds:

```text
Thousands of points
```

Most are junk.

Examples:

```text
Noise

Weak Responses

Unstable Structures

Edge Responses
```

Many detected points will disappear if the image changes slightly.

Therefore SIFT applies several filters next:

```text
Contrast Filter
```

```text
Edge Filter
```

```text
Non Maximum Suppression
```

to remove unstable points and keep only reliable landmarks.

---

## Summary

Input:

```text
Difference of Gaussian Pyramid
```

Operation:

```text
3 × 3 × 3 Neighbor Comparison
```

or:

```text
26 Neighbor Test
```

Output:

```text
(x,y,octave,scale)
```

candidate keypoints.

Purpose:

```text
Find locations that are
distinctive across both

Space
and
Scale
```

This is the stage where SIFT first discovers potential landmarks in the image.

# Step 4: Contrast Filtering

After Scale Space Extrema Detection we obtain a large number of candidate keypoints.

For example, in our implementation:

```text
Extrema Found: 6563
```

At first glance this looks great.

More keypoints should mean more information.

Unfortunately, many of these points are completely useless.

---

## Why Do Weak Extrema Exist?

Consider a completely flat wall.

```text
100 100 100
100 100 100
100 100 100
```

In a perfect world:

```text
No Keypoints
```

should exist.

---

Real images are never perfect.

Camera sensors introduce noise.

Compression introduces artifacts.

Lighting variations create tiny fluctuations.

A real image might look like:

```text
100 100 101
100 102 100
100 100 100
```

Notice:

```text
102
```

is technically larger than all its neighbors.

Therefore the extrema detector might identify it as:

```text
Local Maximum
```

even though it is just noise.

---

## The Problem

The extrema detector only checks:

```text
Relative Difference
```

It asks:

> Am I larger than my neighbors?

or

> Am I smaller than my neighbors?

---

It does NOT ask:

> How important am I?

---

Example:

```text
0.01
```

can still be a local maximum.

Likewise:

```text
-0.01
```

can still be a local minimum.

Neither is useful.

---

## What Does The DoG Value Mean?

Remember:

```text
DoG
=
Difference Of Gaussian
```

The DoG value measures:

```text
How much the image changed
between two scales
```

---

Large magnitude:

```text
Strong Feature
```

Example:

```text
DoG = 25
```

---

Small magnitude:

```text
Weak Feature
```

Example:

```text
DoG = 0.2
```

---

Very small magnitude:

```text
Likely Noise
```

Example:

```text
DoG = 0.01
```

---

## Intuition

Imagine standing in a city.

A skyscraper is easy to notice.

```text
Strong Contrast
```

---

A tiny pebble on the road is harder to notice.

```text
Weak Contrast
```

---

SIFT wants landmarks that stand out strongly.

It removes points that barely rise above the background.

---

## The Contrast Threshold

Lowe introduced a simple idea:

If the DoG response is too small:

```text
Throw It Away
```

---

Mathematically:

```math
|D(x,y,\sigma)| > T
```

where:

```text
D
=
DoG Response

T
=
Threshold
```

---

If:

```text
|DoG| > Threshold
```

keep the point.

Otherwise:

```text
Discard
```

---

## Example

Suppose threshold:

```text
T = 1.0
```

---

Candidate 1:

```text
DoG = 5.2
```

```text
5.2 > 1
```

Keep.

---

Candidate 2:

```text
DoG = -3.8
```

```text
|-3.8| = 3.8
```

Keep.

---

Candidate 3:

```text
DoG = 0.4
```

```text
0.4 < 1
```

Discard.

---

Candidate 4:

```text
DoG = -0.3
```

```text
|-0.3| = 0.3
```

Discard.

---

## Why Absolute Value?

Notice the code:

```python
value = abs(
    dog_pyramid[octave][scale][y,x]
)
```

---

Why use:

```text
abs()
```

?

Because both:

```text
Strong Positive Blob
```

and

```text
Strong Negative Blob
```

are useful.

---

Example:

```text
+15
```

Strong maximum.

Keep.

---

Example:

```text
-15
```

Strong minimum.

Keep.

---

We only care about:

```text
Magnitude
```

not sign.

---

## Code Walkthrough

The implementation:

```python
for x, y, octave, scale in extrema:
```

loops through every candidate point.

---

Fetch DoG response:

```python
value = abs(
    dog_pyramid[octave][scale][y,x]
)
```

---

Check threshold:

```python
if value > threshold:
```

---

If true:

```python
filtered.append(
(
x,
y,
octave,
scale
)
)
```

The keypoint survives.

---

Otherwise:

```text
Discard
```

---

## Why This Step Is Important

Without contrast filtering:

```text
Thousands of Noise Points
```

would survive.

---

These weak points:

- Match poorly
- Change under illumination
- Change under noise
- Create incorrect correspondences

---

Removing them early makes the later stages:

```text
More Stable

More Accurate

More Efficient
```

---

## What Happened In Our Results?

Before filtering:

```text
Extrema Found

6563
```

---

After contrast filtering:

```text
3181
```

---

Meaning:

```text
6563 - 3181

=
3382
```

points were removed.

---

More than half the candidate points disappeared.

This is expected.

Most extrema are weak responses.

---

## Why We Still Cannot Stop

Even after removing weak points:

```text
3181 Keypoints
```

remain.

Many of them still lie on:

```text
Edges
```

instead of:

```text
Corners
```

---

Edges create unstable matches.

A point can slide along an edge and still look identical.

Therefore SIFT performs another test:

```text
Edge Response Elimination
```

using the Hessian Matrix.

This is the next stage of the pipeline.

---

## Summary

Input:

```text
Candidate Keypoints
```

from extrema detection.

---

Operation:

```text
Measure DoG Magnitude
```

---

Rule:

```math
|DoG| > Threshold
```

---

Output:

```text
Strong Keypoints Only
```

---

Purpose:

```text
Remove Weak Responses

Remove Noise

Keep Only High-Contrast Features
```

This stage reduces the number of candidate keypoints dramatically and prepares them for edge filtering.

# Step 5: Edge Response Elimination (Hessian Matrix Filter)

After contrast filtering we still have:

```text
3181 Keypoints
```

At first glance these all seem useful.

Unfortunately many of them lie on edges.

SIFT intentionally removes these points.

This may seem strange because edges are visually strong structures.

To understand why, we first need to understand the difference between:

```text
Corners
```

and

```text
Edges
```

---

## Why Are Edges Bad Features?

Consider a vertical edge.

```text
████████████
████████████
████████████
------------
------------
------------
```

Suppose we place a keypoint here:

```text
████████████
████X███████
████████████
------------
------------
------------
```

Now slide the keypoint slightly up:

```text
████████████
████████████
████X███████
------------
------------
------------
```

The local appearance barely changes.

---

Slide it down:

```text
████████████
████████████
████████████
------------
████X--------
------------
```

Again the appearance looks almost identical.

---

This creates ambiguity.

The point can move significantly along the edge and still look the same.

This means:

```text
Poor Localization
```

---

Now consider a corner.

```text
████████
████████
██
██
██
```

Move slightly:

```text
Corner Shape Changes Immediately
```

The position becomes much more unique.

This means:

```text
Good Localization
```

---

SIFT wants:

```text
Corners
```

not:

```text
Edges
```

---

# How Can We Detect An Edge?

We need a mathematical way to measure:

```text
Curvature
```

in different directions.

---

Imagine a mountain.

If the mountain curves strongly:

```text
Both X and Y Directions
```

it looks like:

```text
Peak
```

or

```text
Corner
```

---

If it curves strongly in only one direction:

```text
Edge
```

---

The Hessian Matrix measures exactly this behavior.

---

# Second Derivatives

The Hessian Matrix is built from:

```text
Second Derivatives
```

These measure curvature.

---

## Dxx

Curvature in x-direction.

```math
D_{xx}
=
I(x+1,y)
+
I(x-1,y)
-
2I(x,y)
```

---

Your code:

```python
Dxx = (
    img[y,x+1]
    +
    img[y,x-1]
    -
    2*img[y,x]
)
```

---

Interpretation:

```text
Large Dxx
=
Strong Curvature Along X
```

---

## Dyy

Curvature in y-direction.

```math
D_{yy}
=
I(x,y+1)
+
I(x,y-1)
-
2I(x,y)
```

---

Your code:

```python
Dyy = (
    img[y+1,x]
    +
    img[y-1,x]
    -
    2*img[y,x]
)
```

---

Interpretation:

```text
Large Dyy
=
Strong Curvature Along Y
```

---

## Dxy

Mixed derivative.

Measures interaction between x and y.

```math
D_{xy}
=
\frac{
I(x+1,y+1)
-
I(x+1,y-1)
-
I(x-1,y+1)
+
I(x-1,y-1)
}{4}
```

---

Your code:

```python
Dxy = (
    img[y+1,x+1]
    -
    img[y+1,x-1]
    -
    img[y-1,x+1]
    +
    img[y-1,x-1]
) / 4.0
```

---

# Constructing The Hessian Matrix

These derivatives form:

```math
H=
\begin{bmatrix}
D_{xx} & D_{xy}\\
D_{xy} & D_{yy}
\end{bmatrix}
```

---

The Hessian describes how the image surface bends around the keypoint.

Think of it as describing the shape of a tiny hill around the feature.

---

# Corner vs Edge Using Eigenvalues

Suppose the Hessian has two eigenvalues:

```text
λ₁
λ₂
```

---

## Corner

Strong curvature in both directions.

```text
λ₁ = 50

λ₂ = 40
```

Both are large.

This means:

```text
Corner
```

---

## Edge

Strong curvature in one direction.

Weak curvature in the other.

```text
λ₁ = 100

λ₂ = 1
```

One is huge.

One is tiny.

This means:

```text
Edge
```

---

# The Problem

Computing eigenvalues for every candidate keypoint is expensive.

David Lowe found a shortcut.

---

# Trace And Determinant

For a 2×2 matrix:

```math
H=
\begin{bmatrix}
D_{xx} & D_{xy}\\
D_{xy} & D_{yy}
\end{bmatrix}
```

---

Trace:

```math
Tr(H)
=
D_{xx}
+
D_{yy}
```

---

Your code:

```python
trace = Dxx + Dyy
```

---

Determinant:

```math
Det(H)
=
D_{xx}D_{yy}
-
D_{xy}^2
```

---

Your code:

```python
det =
(
Dxx*Dyy
-
Dxy*Dxy
)
```

---

# Why Determinant Must Be Positive

Suppose:

```text
det <= 0
```

This usually indicates:

```text
Unstable Region
```

or

```text
Saddle Point
```

---

Therefore:

```python
if det <= 0:
    continue
```

The point is discarded.

---

# The Edge Ratio Test

Suppose:

```text
λ₁ = Large

λ₂ = Small
```

This is an edge.

---

Lowe showed that the ratio:

```math
\frac{
(\lambda_1+\lambda_2)^2
}
{
\lambda_1\lambda_2
}
```

can be computed using:

```math
\frac{
Tr(H)^2
}
{
Det(H)
}
```

without explicitly finding eigenvalues.

---

This is exactly what your code computes:

```python
ratio =
(
trace*trace
)
/ det
```

---

# Understanding The Ratio

## Corner

Suppose:

```text
λ₁ = 10

λ₂ = 10
```

Then:

```math
(10+10)^2
/
(10×10)
=
4
```

Small value.

Good feature.

---

## Edge

Suppose:

```text
λ₁ = 100

λ₂ = 1
```

Then:

```math
(100+1)^2
/
(100×1)
=
10201
/
100
=
102
```

Huge value.

Bad feature.

---

The larger the ratio:

```text
More Edge-Like
```

---

# Lowe's Threshold

Let:

```text
r = 10
```

This means:

```text
Maximum Allowed Eigenvalue Ratio
=
10
```

---

Threshold becomes:

```math
\frac{(r+1)^2}{r}
```

---

For:

```text
r = 10
```

```math
=
\frac{121}{10}
=
12.1
```

---

Your code:

```python
threshold =
(
(r+1)**2
)
/ r
```

---

# Final Decision

If:

```python
ratio < threshold
```

then:

```text
Keep Keypoint
```

---

Otherwise:

```text
Discard Keypoint
```

because it is likely an edge.

---

# What Happened In Our Results?

Before edge filtering:

```text
3181 Keypoints
```

---

After edge filtering:

```text
1659 Keypoints
```

---

Removed:

```text
3181 - 1659

=
1522 Points
```

---

Almost half the remaining keypoints were edge responses.

---

# Why This Step Is So Important

Without edge filtering:

```text
Many Unstable Features
```

would survive.

These points:

- Match poorly
- Produce incorrect correspondences
- Reduce RANSAC accuracy
- Create noisy descriptors

---

By removing edge responses we retain:

```text
Corners

Blobs

Highly Distinctive Structures
```

which are much more reliable for matching.

---

# Summary

Input:

```text
Contrast Filtered Keypoints
```

---

Operation:

```text
Compute Hessian Matrix

Measure Curvature

Reject Edge-Like Structures
```

---

Output:

```text
Corner-Like Keypoints
```

---

Purpose:

```text
Remove Strong Edges

Keep Stable Landmarks

Improve Matching Accuracy
```

This stage transforms a large collection of high-contrast points into a much cleaner set of distinctive geometric features that are suitable for descriptor generation.

# Step 6: Non-Maximum Suppression (NMS)

After edge filtering we still have:

```text
1659 Keypoints
```

These keypoints are much better than before.

They are:

- High contrast
- Not edge-like
- Stable across scale

However, there is still a problem.

Many keypoints are actually describing the exact same physical feature.

---

# Why Does This Happen?

Consider a strong corner.

```text
████████
████████
██
██
██
```

When the Difference of Gaussian detector examines this corner, several nearby pixels may all produce strong responses.

Example:

```text
15 18 14
17 20 16
13 19 12
```

Notice:

```text
20
```

is the strongest response.

However:

```text
19
18
17
16
```

are also strong.

---

As a result, the detector may produce:

```text
Keypoint A

Keypoint B

Keypoint C

Keypoint D
```

all clustered around the same corner.

---

Visually:

```text
Corner

● ●
 ●
● ●
```

instead of:

```text
Corner

●
```

---

# Why Is This A Problem?

Suppose we keep all of them.

Then:

```text
One Physical Corner
```

creates:

```text
Five Different Keypoints
```

---

This causes several issues.

### Problem 1: Redundant Features

Multiple descriptors describe almost the same location.

Example:

```text
Feature 1
```

```text
Feature 2
```

```text
Feature 3
```

all correspond to the same corner.

We gain no new information.

---

### Problem 2: Increased Computation

Every keypoint eventually generates:

```text
128-D Descriptor
```

More keypoints means:

```text
More Memory

More Matching

More Computation
```

---

### Problem 3: Feature Clustering

Without suppression:

```text
Important Corner
```

may produce:

```text
20 Keypoints
```

while another important region produces:

```text
1 Keypoint
```

This creates uneven feature distribution.

---

# The Goal

Instead of:

```text
One Corner
↓
Many Keypoints
```

we want:

```text
One Corner
↓
One Representative Keypoint
```

---

# The Basic Idea

Suppose nearby keypoints have responses:

```text
20
18
15
14
12
```

The strongest response:

```text
20
```

is kept.

The weaker responses are discarded.

---

This process is called:

```text
Non-Maximum Suppression
```

because:

```text
Only Local Maximum Responses Survive
```

---

# How Our Implementation Works

The first step is assigning a score to every keypoint.

---

For each keypoint:

```python
response = abs(
    dog_pyramid[octave][scale][y,x]
)
```

The score is simply:

```text
Absolute DoG Response
```

---

Example:

```text
Point A = 25

Point B = 18

Point C = 10

Point D = 7
```

---

Larger value:

```text
Stronger Feature
```

---

Smaller value:

```text
Weaker Feature
```

---

# Sorting By Strength

The keypoints are sorted:

```python
scored.sort(
    reverse=True
)
```

---

After sorting:

```text
25
22
20
19
18
17
15
...
```

The strongest features appear first.

---

# Greedy Selection

The algorithm now processes points one by one.

---

Take strongest point:

```text
Response = 25
```

Keep it.

---

Now examine next point:

```text
Response = 22
```

Compute distance from previously selected keypoints.

---

# Distance Calculation

The code uses:

```python
dist = np.sqrt(
    (x-sx)**2
    +
    (y-sy)**2
)
```

This is simply:

```text
Euclidean Distance
```

between two keypoints.

---

Example:

Point A:

```text
(100,100)
```

Point B:

```text
(108,105)
```

Distance:

```math
\sqrt{
(108-100)^2
+
(105-100)^2
}
```

```math
=
\sqrt{
64+25
}
```

```math
=
9.43
```

pixels

---

# Radius Rule

Suppose:

```python
radius = 15
```

---

If:

```text
Distance < 15
```

then:

```text
Too Close
```

and the weaker point is removed.

---

If:

```text
Distance > 15
```

then:

```text
Far Enough Away
```

and the point survives.

---

# Example

Suppose responses:

```text
Point A = 25

Point B = 20

Point C = 18
```

---

Coordinates:

```text
A = (100,100)

B = (105,102)

C = (200,220)
```

---

Keep A first.

---

Distance:

```text
A ↔ B
```

```math
=
5.38
```

Less than:

```text
15
```

Therefore:

```text
Discard B
```

---

Distance:

```text
A ↔ C
```

Large.

Therefore:

```text
Keep C
```

---

Final result:

```text
A
C
```

Only the strongest nearby feature survives.

---

# Why Check Octaves?

Notice the code:

```python
if octave != so:
    continue
```

---

This means:

```text
Only Compare Features
Within The Same Octave
```

---

Why?

A feature detected at:

```text
Octave 0
```

and

```text
Octave 2
```

represents different scales.

They should not suppress each other.

---

Therefore suppression occurs only inside the same octave.

---

# Visualization

Before NMS:

```text
● ● ●
 ● ●
● ● ●
```

Many points clustered around one corner.

---

After NMS:

```text
    ●
```

Only the strongest representative survives.

---

# What Happened In Our Results?

Before NMS:

```text
1659 Keypoints
```

---

After NMS:

```text
738 Keypoints
```

---

Removed:

```text
1659 - 738
=
921 Keypoints
```

---

More than half the remaining points were redundant.

---

# Why This Step Is Important

Without NMS:

```text
Too Many Features

Redundant Descriptors

Slower Matching
```

---

With NMS:

```text
Compact Feature Set

Better Spatial Distribution

Faster Descriptor Generation

Faster Matching
```

---

# What Comes Next?

At this point we have:

```text
Stable

High Contrast

Non-Edge

Well Distributed

Scale Invariant
```

keypoints.

However they still lack:

```text
Rotation Invariance
```

If we rotate the camera:

```text
Descriptor Changes
```

and matching may fail.

The next stage solves this problem:

```text
Orientation Assignment
```

where every keypoint is given its own dominant direction before descriptor generation begins.

---

# Summary

Input:

```text
Edge Filtered Keypoints
```

---

Operation:

```text
Sort By Response

Keep Strongest Points

Remove Nearby Weaker Points
```

---

Output:

```text
Sparse High Quality Keypoints
```

---

Purpose:

```text
Reduce Redundancy

Improve Spatial Distribution

Decrease Computation

Keep One Representative Per Feature
```

This stage transforms a dense cluster of detections into a clean set of unique landmarks that are ready for orientation assignment.

# Step 7: Orientation Assignment

After Non-Maximum Suppression we have:

```text
738 Keypoints
```

These keypoints are:

- High Contrast
- Not Edge-Like
- Scale Invariant
- Well Distributed

However there is still a major problem.

---

# The Rotation Problem

Suppose we detect a corner.

Image A:

```text
████████
████████
██
██
██
```

Now rotate the camera:

```text
██
██
████████
████████
```

The corner is physically the same.

Humans immediately recognize it.

However the pixel arrangement has changed.

---

Without additional processing:

```text
Descriptor Changes
```

which means:

```text
Matching Fails
```

---

To solve this, SIFT gives every keypoint its own:

```text
Orientation
```

Think of it as attaching a tiny compass to every keypoint.

---

# The Core Idea

Instead of describing the feature relative to:

```text
Image Coordinates
```

we describe it relative to:

```text
Keypoint Orientation
```

---

Example:

Suppose the strongest local direction is:

```text
60°
```

We define:

```text
60°
=
New Zero Degrees
```

for that keypoint.

Now every descriptor is built relative to that orientation.

---

Result:

```text
Rotate Camera
↓
Orientation Changes
↓
Descriptor Remains Similar
```

---

# First Step: Compute Image Gradients

To determine orientation we need to know:

```text
Which Direction
Intensity Changes
```

around the keypoint.

---

Your code computes:

```python
dx = cv2.Sobel(
    img,
    cv2.CV_32F,
    1,
    0,
    ksize=3
)
```

and

```python
dy = cv2.Sobel(
    img,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)
```

---

# What Is A Gradient?

Consider a row of pixels.

```text
10 10 10 10 10
```

No change.

Gradient:

```text
0
```

---

Now:

```text
10 10 10 200 200
```

Huge change.

Gradient:

```text
Large
```

---

Gradients tell us:

```text
Where Edges Exist
```

and

```text
Which Direction They Point
```

---

# Sobel Operator

The Sobel operator estimates derivatives.

---

## Horizontal Gradient

```text
dx
```

measures:

```text
Change Along X Direction
```

---

Example:

```text
Dark → Bright
```

produces:

```text
Large dx
```

---

## Vertical Gradient

```text
dy
```

measures:

```text
Change Along Y Direction
```

---

Example:

```text
Top Dark
Bottom Bright
```

produces:

```text
Large dy
```

---

# Gradient Magnitude

After computing:

```text
dx
```

and

```text
dy
```

we calculate:

```math
Magnitude
=
\sqrt{
dx^2
+
dy^2
}
```

---

Your code:

```python
magnitude =
np.sqrt(
dx**2
+
dy**2
)
```

---

Interpretation:

```text
Large Magnitude
=
Strong Edge
```

---

```text
Small Magnitude
=
Weak Edge
```

---

# Gradient Orientation

Now we compute:

```text
Direction Of Change
```

---

Formula:

```math
\theta
=
atan2(
dy,
dx
)
```

---

Your code:

```python
orientation =
np.degrees(
np.arctan2(
dy,
dx
)
)
```

---

Example:

```text
dx = 1
dy = 0
```

Direction:

```text
0°
```

---

Example:

```text
dx = 0
dy = 1
```

Direction:

```text
90°
```

---

Example:

```text
dx = -1
dy = 0
```

Direction:

```text
180°
```

---

Example:

```text
dx = 0
dy = -1
```

Direction:

```text
270°
```

---

Now every pixel has:

```text
Magnitude
```

and

```text
Orientation
```

stored inside:

```python
mag_pyr
```

and

```python
ori_pyr
```

---

# Building The Orientation Histogram

For each keypoint we create:

```text
36 Bin Histogram
```

---

Why 36 bins?

Because:

```text
360°
÷
36
=
10°
```

per bin.

---

Example:

```text
Bin 0
=
0° to 10°

Bin 1
=
10° to 20°

Bin 2
=
20° to 30°

...

Bin 35
=
350° to 360°
```

---

Your code:

```python
hist = np.zeros(36)
```

creates this histogram.

---

# Choosing A Neighborhood

We do not look at the entire image.

Only a region around the keypoint.

Radius:

```python
radius =
int(
round(
3*1.5*sigma
)
)
```

---

Larger scale:

```text
Larger Neighborhood
```

---

Smaller scale:

```text
Smaller Neighborhood
```

---

This keeps orientation estimation scale invariant.

---

# Why Gaussian Weighting?

Suppose we are assigning orientation to a keypoint.

Nearby pixels should matter more.

Far away pixels should matter less.

---

Therefore SIFT uses:

```math
Weight
=
e^{-r^2/(2\sigma^2)}
```

---

Your code:

```python
weight =
np.exp(
-(dx*dx + dy*dy)
/
(2*(1.5*sigma)**2)
)
```

---

Result:

```text
Close Pixels
=
Large Weight
```

---

```text
Far Pixels
=
Small Weight
```

---

# Voting Into Histogram

Each pixel contributes:

```text
Weight × Magnitude
```

to its orientation bin.

---

Example:

Pixel:

```text
Magnitude = 20

Orientation = 74°
```

---

Which bin?

```text
74 / 10

=
7
```

---

Vote:

```text
Histogram[7]
+=
20
```

---

Your code:

```python
bin_idx =
int(
np.floor(
ang/10
)
)
%36
```

---

and

```python
hist[bin_idx]
+=
weight*mag
```

---

# Finding The Dominant Orientation

After all votes:

Example histogram:

```text
0°    15
10°   20
20°   50
30°   120
40°   180
50°   320
60°   500
70°   290
80°   160
```

---

Largest peak:

```text
60°
```

---

Therefore:

```text
Keypoint Orientation
=
60°
```

---

# Why Multiple Orientations?

Consider a T-Junction.

```text
█████████
    █
    █
    █
```

Two strong directions exist.

---

Histogram:

```text
60°  = 500

150° = 430
```

---

Notice:

```text
430
>
0.8 × 500
```

---

The second peak is almost as important as the first.

---

If we keep only one orientation:

```text
Information Lost
```

---

Therefore Lowe introduced:

```text
80% Rule
```

---

# The 80% Rule

Let:

```text
Maximum Peak
=
500
```

---

Threshold:

```text
0.8 × 500

=
400
```

---

Any peak larger than:

```text
400
```

creates:

```text
New Keypoint
```

with that orientation.

---

Your code:

```python
if hist[bin_idx]
>=
0.8 * max_peak:
```

---

creates additional orientations.

---

# Example

Original keypoint:

```text
(100,50)
```

---

Strong peaks:

```text
60°

150°
```

---

Output:

```text
(100,50,60°)

(100,50,150°)
```

---

Same location.

Different orientations.

---

# Why Did The Number Of Keypoints Increase?

Before orientation assignment:

```text
738 Keypoints
```

---

After orientation assignment:

```text
1057 Keypoints
```

---

Many keypoints generated:

```text
Multiple Orientations
```

---

This is expected.

It improves robustness significantly.

---

# Why This Step Is Important

Without orientation assignment:

```text
Rotate Camera
↓
Descriptor Changes
↓
Matching Fails
```

---

With orientation assignment:

```text
Rotate Camera
↓
Orientation Updated
↓
Descriptor Remains Similar
↓
Matching Works
```

---

This is the reason SIFT is:

```text
Rotation Invariant
```

---

# Output Format

Before:

```text
(
x,
y,
octave,
scale
)
```

---

After:

```text
(
x,
y,
octave,
scale,
orientation
)
```

Example:

```text
(
125,
87,
1,
3,
60°
)
```

---

# Summary

Input:

```text
NMS Keypoints
```

---

Operation:

```text
Compute Gradients

Build 36-Bin Histogram

Find Dominant Orientation

Apply 80% Rule
```

---

Output:

```text
(
x,
y,
octave,
scale,
orientation
)
```

---

Purpose:

```text
Achieve Rotation Invariance

Create Stable Reference Frame

Prepare For Descriptor Generation
```

The next stage uses this orientation to build the famous:

```text
128-Dimensional SIFT Descriptor
```

which acts as a unique fingerprint for every keypoint.

# Step 8: Descriptor Generation

After orientation assignment we have keypoints of the form:

```text
(
x,
y,
octave,
scale,
orientation
)
```

Example:

```text
(
125,
87,
1,
3,
60°
)
```

This tells us:

```text
Where The Feature Is

What Scale It Exists At

Which Direction It Faces
```

However this is still not enough for matching.

Two different corners may have:

```text
Similar Location

Similar Scale

Similar Orientation
```

but still represent completely different structures.

We need a unique fingerprint for every keypoint.

This fingerprint is called:

```text
SIFT Descriptor
```

---

# What Is A Descriptor?

Think of a descriptor as:

```text
A Numerical Fingerprint
```

for a keypoint.

---

Humans recognize a face using:

```text
Eyes

Nose

Mouth

Face Shape
```

combined together.

---

SIFT recognizes a feature using:

```text
Local Gradient Patterns
```

combined together.

---

The final descriptor contains:

```text
128 Numbers
```

which summarize the local appearance around the keypoint.

---

# The Goal

We want a descriptor that is:

```text
Distinctive

Rotation Invariant

Scale Invariant

Robust To Illumination Changes
```

---

# Step 1: Extract A Neighborhood

For every keypoint:

```text
(x,y)
```

we examine a:

```text
16 × 16
```

pixel window around it.

---

Your code:

```python
radius = 8
```

creates:

```text
-8 to +8
```

around the center.

Result:

```text
16 × 16 Region
```

---

Visualization:

```text
□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□
□□□□□□●□□□□□□□□
□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□
```

where:

```text
●
```

is the keypoint.

---

# Why Use A Local Region?

The descriptor should describe:

```text
The Feature
```

not the entire image.

---

If we used the whole image:

```text
Descriptor Changes
Whenever Camera Moves
```

---

A local neighborhood remains much more stable.

---

# Step 2: Rotate Into The Keypoint Frame

Remember:

Every keypoint has an assigned orientation.

Example:

```text
60°
```

---

Suppose the camera rotates.

Without compensation:

```text
Descriptor Changes
```

---

Instead we rotate the coordinate system itself.

---

Imagine a feature facing:

```text
60°
```

We redefine:

```text
60°
=
New Zero Degrees
```

---

Your code:

```python
x_rot =
cos_t * dx
+
sin_t * dy
```

```python
y_rot =
-sin_t * dx
+
cos_t * dy
```

---

This is a standard 2D rotation.

---

Result:

```text
Feature Always Faces Up
```

regardless of camera rotation.

---

This is one of the reasons SIFT is rotation invariant.

---

# Step 3: Divide Into 4×4 Cells

The 16×16 window is split into:

```text
4 × 4
```

cells.

---

Visualization:

```text
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
|    |    |    |    |
+----+----+----+----+
```

---

Total:

```text
16 Cells
```

---

Why?

Because a descriptor should preserve:

```text
Spatial Layout
```

of gradients.

---

Without cells:

```text
All Information Mixed Together
```

---

With cells:

```text
We Know

Where Gradients Occur
```

---

# Step 4: Create Orientation Histograms

Inside every cell we create:

```text
8 Orientation Bins
```

---

Each bin covers:

```text
360°
÷
8
=
45°
```

---

Example:

```text
Bin 0
0°-45°

Bin 1
45°-90°

Bin 2
90°-135°

...

Bin 7
315°-360°
```

---

Every pixel votes into one of these bins.

---

Suppose:

```text
Pixel Magnitude = 20

Orientation = 70°
```

---

70° belongs to:

```text
Bin 1
```

because:

```text
45°-90°
```

---

Vote:

```text
Histogram[1]
+= 20
```

---

# Why Use Histograms?

Individual pixel values are unstable.

A one-pixel shift can completely change intensity.

---

Histograms capture:

```text
Overall Gradient Structure
```

which is much more stable.

---

# Descriptor Size

Each cell:

```text
8 Values
```

---

Number of cells:

```text
4 × 4
=
16
```

---

Total descriptor size:

```text
16 × 8
=
128
```

---

This is where the famous:

```text
128-Dimensional Descriptor
```

comes from.

---

# Converting Coordinates Into Cells

After rotation:

Your code computes:

```python
col_bin =
(x_rot / 4.0)
+
1.5
```

---

and

```python
row_bin =
(y_rot / 4.0)
+
1.5
```

---

Why divide by:

```text
4
```

?

Because each cell covers:

```text
4 Pixels
```

inside the:

```text
16×16 Window
```

---

Example:

```text
x_rot = 6
```

becomes:

```text
6 / 4
=
1.5
```

meaning:

```text
Between Cell 1 And Cell 2
```

---

This becomes important for interpolation.

---

# Orientation Bin Calculation

The descriptor uses:

```text
8 Orientation Bins
```

---

Your code:

```python
ori_bin =
angle / 45.0
```

---

Example:

```text
90°
```

becomes:

```text
90/45
=
2
```

---

Meaning:

```text
Orientation Bin 2
```

---

Example:

```text
112°
```

becomes:

```text
112/45
=
2.49
```

---

This lies:

```text
Between Bin 2
and
Bin 3
```

which leads to interpolation.

---

# The Problem With Hard Assignment

Suppose:

```text
Orientation

44.9°
```

---

Bin:

```text
0
```

---

Move one pixel.

Orientation becomes:

```text
45.1°
```

---

Now:

```text
Bin 1
```

---

Descriptor changes dramatically.

This is bad.

---

SIFT solves this using:

```text
Trilinear Interpolation
```

---

# Trilinear Interpolation

Every pixel distributes its vote across:

```text
Row Dimension

Column Dimension

Orientation Dimension
```

simultaneously.

---

Instead of:

```text
100%
into one bin
```

it shares the vote.

---

Example:

```text
Row Position
=
1.2
```

---

Contributions:

```text
80%
Cell 1

20%
Cell 2
```

---

Similarly:

```text
Orientation Bin
=
2.7
```

---

Contributions:

```text
30%
Bin 2

70%
Bin 3
```

---

This makes descriptors much more stable.

---

# What The Code Is Doing

Row interpolation:

```python
r0
r1

wr0
wr1
```

---

Column interpolation:

```python
c0
c1

wc0
wc1
```

---

Orientation interpolation:

```python
o0
o1

wo0
wo1
```

---

The pixel's magnitude is distributed among neighboring bins using:

```python
magnitude
*
wr
*
wc
*
wo
```

---

This is called:

```text
Trilinear Interpolation
```

because interpolation occurs along:

```text
Row

Column

Orientation
```

---

# Flattening The Descriptor

Currently:

```text
4 × 4 × 8
```

---

Shape:

```python
descriptor.shape

=
(4,4,8)
```

---

Machine learning and matching algorithms prefer vectors.

---

Therefore:

```python
descriptor =
descriptor.flatten()
```

---

Result:

```text
128-D Vector
```

---

# Descriptor Normalization

Suppose:

Image A:

```text
Bright Day
```

---

Image B:

```text
Cloudy Day
```

---

All gradient magnitudes become larger or smaller.

---

We remove this effect using:

```python
descriptor /= norm
```

---

Now:

```text
Only Relative Structure Matters
```

not overall brightness.

---

# Illumination Clipping

Suppose sunlight creates a bright reflection.

One histogram bin becomes:

```text
Huge
```

and dominates everything.

---

SIFT limits every value:

```python
descriptor =
np.clip(
descriptor,
0,
0.2
)
```

---

Maximum contribution:

```text
20%
```

---

This prevents:

```text
Glare

Reflections

Extreme Contrast
```

from dominating the descriptor.

---

# Why Normalize Again?

Clipping changes the vector length.

Therefore SIFT performs:

```python
descriptor /= np.linalg.norm(descriptor)
```

again.

---

Final result:

```text
Unit Length Descriptor
```

with controlled values.

---

# Final Output

For our image:

```text
Keypoints After Orientation

1057
```

---

Descriptors:

```text
(1057,128)
```

Meaning:

```text
1057 Keypoints

Each Keypoint
↓
128 Numbers
```

---

# What Does A Descriptor Represent?

A descriptor does NOT represent:

```text
Color

Object Type

Meaning
```

---

A descriptor represents:

```text
Local Gradient Structure
```

around a keypoint.

---

Think of it as:

```text
A Fingerprint
```

for that feature.

---

# Summary

Input:

```text
(
x,
y,
octave,
scale,
orientation
)
```

---

Operations:

```text
Extract 16×16 Region

Rotate Into Keypoint Frame

Divide Into 4×4 Cells

Build 8-Bin Histograms

Apply Trilinear Interpolation

Flatten

Normalize

Clip

Normalize Again
```

---

Output:

```text
128-Dimensional Descriptor
```

Purpose:

```text
Create A Unique

Rotation Invariant

Scale Invariant

Illumination Robust

Fingerprint
```

for every detected feature.

These descriptors are the final output of the SIFT feature extraction pipeline and will later be used for feature matching, homography estimation, panorama stitching, SLAM, visual odometry, and 3D reconstruction.
