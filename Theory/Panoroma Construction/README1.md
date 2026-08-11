# Panorama Construction After Homography

## Introduction

In the previous document, we learned how feature matching, RANSAC, and homography estimation work.

At the end of that pipeline, we obtained the final Homography Matrix:

```math
H =
\begin{bmatrix}
h_{11} & h_{12} & h_{13}\\
h_{21} & h_{22} & h_{23}\\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
```

Many beginners assume the panorama is finished once the homography matrix is found.

In reality, the homography is only the beginning of the image stitching stage.

This document explains how the homography matrix is used to:

1. Warp images
2. Create a panorama canvas
3. Align overlapping regions
4. Blend seams
5. Produce a final panorama

---

# 1. What Does the Homography Matrix Actually Do?

The homography matrix does not transform colors.

It transforms coordinates.

For example:

```text
Pixel coordinate in Image A

(10,20)
```

After applying the homography:

```text
(10,20)
↓
H
↓
(150,180)
```

The pixel color originally stored at coordinate `(10,20)` is moved to coordinate `(150,180)`.

The matrix describes how geometry changes between the two images.

It knows nothing about:

- Trees
- Buildings
- Cars
- Colors

It only knows spatial relationships.

---

# 2. Why Direct Pixel Mapping Fails

A naive approach would be:

```text
For every source pixel:

1. Apply H
2. Compute destination coordinate
3. Paint pixel
```

This method is called:

```text
Forward Mapping
```

Unfortunately it produces holes.

Example:

```text
Source Pixels

A
B
C
D
```

After warping:

```text
A      B

    C      D
```

Some destination locations receive no pixels.

These become visible black cracks.

The stronger the perspective distortion, the worse the problem becomes.

---

# 3. Forward Mapping vs Inverse Warping

## Forward Mapping

```text
Source Pixel
      ↓
Apply H
      ↓
Destination Pixel
```

Problem:

```text
Holes
Cracks
Missing Pixels
```

---

## Inverse Warping

Instead of asking:

> Where should this source pixel go?

We ask:

> Which source pixel belongs here?

Every destination pixel works backward.

```text
Destination Pixel
      ↓
Apply H⁻¹
      ↓
Locate Source Pixel
      ↓
Copy Color
```

Result:

```text
No Holes
No Missing Pixels
```

This is why almost all modern panorama systems use inverse warping.

---

# 4. Why We Need the Inverse Matrix

The GPU does not use H directly.

Instead it uses:

```text
H⁻¹
```

The inverse homography matrix.

The inverse is computed once on the CPU and uploaded to the GPU.

For every destination coordinate:

```math
x_{src}
=
H^{-1}
x_{dest}
```

This tells us exactly where to fetch the color from the original image.

---

# 5. Building the Panorama Canvas

Before warping begins, we must determine the size of the final panorama.

The original image dimensions are usually too small.

---

## Step 1: Project Image Corners

Take the four corners of Image A:

```text
(0,0)

(width,0)

(0,height)

(width,height)
```

Apply the homography.

Example:

```text
(-500,50)

(1200,30)

(-450,900)

(1300,950)
```

---

## Step 2: Compute Bounding Box

Find:

```text
minX
maxX

minY
maxY
```

Example:

```text
minX = -500
maxX = 2500

minY = -100
maxY = 1200
```

---

## Step 3: Allocate Canvas

Canvas size:

```text
width  = maxX - minX
height = maxY - minY
```

This guarantees enough space for both images.

---

# 6. The Negative Coordinate Problem

Suppose a warped pixel lands at:

```text
(-500,100)
```

Negative coordinates cannot be stored inside an image buffer.

This would cause memory errors.

---

# 7. Translation Offset

To solve this problem we shift everything.

Example:

```text
minX = -500
minY = -100
```

Translation:

```text
+500 in X

+100 in Y
```

Now:

```text
(-500,100)
```

becomes:

```text
(0,200)
```

which is valid.

The entire panorama is translated into positive coordinate space.

---

# 8. Inverse Warping Mathematics

For a destination pixel:

```text
(x_dest, y_dest)
```

Convert to homogeneous coordinates:

```math
\begin{bmatrix}
x_{dest}\\
y_{dest}\\
1
\end{bmatrix}
```

Apply the inverse homography:

```math
x_{src}
=
H^{-1}
x_{dest}
```

Result:

```math
\begin{bmatrix}
x_{src}\\
y_{src}\\
w_{src}
\end{bmatrix}
```

Normalize:

```math
x = x_{src}/w_{src}
```

```math
y = y_{src}/w_{src}
```

These coordinates tell us where to fetch the pixel from the original image.

---

# 9. Bilinear Interpolation

The source coordinate is usually fractional.

Example:

```text
(104.3 , 55.8)
```

No such pixel exists.

Nearest pixels:

```text
(104,55)

(105,55)

(104,56)

(105,56)
```

We blend them using weighted averaging.

This process is called:

```text
Bilinear Interpolation
```

Benefits:

- Smooth results
- No blockiness
- No jagged edges

---

# 10. Image Alignment Complete

After inverse warping:

```text
Image A
```

has been transformed and placed onto the panorama canvas.

Image B is also present on the same canvas.

Common landmarks now occupy the same coordinates.

Example:

```text
Mountain Peak
```

Image A:

```text
(800,300)
```

After warping:

```text
(600,50)
```

Image B:

```text
(100,50)
```

After translation:

```text
(600,50)
```

Both images now agree on the mountain location.

The geometric alignment problem is solved.

---

# 11. Why Blending Is Necessary

Although alignment is correct, seams remain.

Example:

```text
Image A pixel = 180

Image B pixel = 150
```

Both images want to paint the same location.

If one image overwrites the other:

```text
Visible Seam
```

appears.

We therefore need blending.

# 12. Feathering (Distance Weighted Alpha Blending)

After inverse warping, both images now occupy the same panorama canvas.

The geometric alignment problem has already been solved.

Common landmarks from both images now appear at the same coordinates.

For example:

```text
Mountain Peak
```

Image A originally contained:

```text
(800,300)
```

After warping:

```text
(600,50)
```

Image B originally contained:

```text
(100,50)
```

After translation onto the panorama canvas:

```text
(600,50)
```

Both images now agree that the mountain peak belongs at:

```text
(600,50)
```

This sounds perfect.

Unfortunately a new problem appears.

---

## The Overlap Problem

Consider two overlapping images.

```text
Image A
|-----------A-----------|
              |------Overlap------|

Image B
              |------Overlap------|
                            |-----------B-----------|
```

Inside the overlap region:

```text
Both Images Want To Paint
The Same Pixels
```

---

Suppose a mountain appears in both images.

Image A records:

```text
Pixel Value = 180
```

Image B records:

```text
Pixel Value = 150
```

Remember:

```text
0   = Black

255 = White
```

---

Now both images want to paint:

```text
Canvas Coordinate

(600,50)
```

Image A says:

```text
180
```

Image B says:

```text
150
```

The obvious question becomes:

```text
Which Pixel Should Win?
```

---

## Naive Solution: Overwrite One Image

Suppose we paint:

```text
Image B First
```

and then:

```text
Image A On Top
```

The final pixel becomes:

```text
180
```

because Image A overwrites Image B.

---

At the overlap boundary:

```text
150 150 150 150 | 180 180 180 180
```

A sudden jump appears.

Human vision immediately notices:

```text
SEAM
```

between the two images.

---

## Why Do Seams Exist?

Even when alignment is perfect, images rarely have identical brightness.

Reasons include:

```text
Different Exposure

Different Camera Gain

Lens Vignetting

Changing Sunlight

Changing Shadows

Automatic Camera Adjustments
```

---

As a result:

```text
Same Physical Object
```

may have:

```text
Different Pixel Values
```

in the two photographs.

---

Therefore simply stacking images creates visible boundaries.

---

## The Core Idea Behind Feathering

Instead of choosing one image, we combine both images.

Think of it like a movie transition.

When one movie scene fades into another:

```text
Scene A Slowly Disappears

Scene B Slowly Appears
```

For a brief moment:

```text
Both Exist Simultaneously
```

---

Feathering does exactly the same thing.

Instead of:

```text
Choose A

or

Choose B
```

we use:

```text
Part A

Part B
```

---

## Simple Example

Suppose:

```text
Image A Pixel = 180

Image B Pixel = 150
```

Use:

```text
50% A

50% B
```

Then:

```math
0.5(180)
+
0.5(150)
=
165
```

The resulting pixel becomes:

```text
165
```

---

Instead of a sudden jump:

```text
150 → 180
```

we obtain:

```text
150 → 165 → 180
```

which appears much smoother.

---

## Why Not Use 50%-50% Everywhere?

At first this sounds reasonable.

Why not simply average everything?

```math
I_{final}
=
\frac{
I_A + I_B
}{2}
```

---

Because the entire overlap region would become blurry.

Imagine a 200-pixel overlap.

Every pixel would become:

```text
Half Image A

Half Image B
```

The panorama would lose sharpness.

---

Instead we gradually change the weights.

---

## Weight Transition

Suppose overlap width:

```text
100 Pixels
```

At the left side:

```text
Image A Weight = 1.0

Image B Weight = 0.0
```

Only Image A contributes.

---

At the center:

```text
Image A Weight = 0.5

Image B Weight = 0.5
```

Both contribute equally.

---

At the right side:

```text
Image A Weight = 0.0

Image B Weight = 1.0
```

Only Image B contributes.

---

Visualization:

```text
Image A Weight

1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
```

```text
Image B Weight

0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
```

This creates a smooth crossfade between the images.

---

## Feathering Formula

For every overlapping pixel:

```math
I_{final}
=
w_A I_A
+
w_B I_B
```

where:

```text
I_A
=
Pixel From Image A

I_B
=
Pixel From Image B

w_A
=
Weight Of Image A

w_B
=
Weight Of Image B
```

and:

```math
w_A + w_B = 1
```

always.

---

## Numerical Example

Suppose:

```text
Image A = 180

Image B = 150
```

Current weights:

```text
w_A = 0.7

w_B = 0.3
```

Then:

```math
I_{final}
=
0.7(180)
+
0.3(150)
```

```math
=
126 + 45
```

```math
=
171
```

Final pixel:

```text
171
```

---

Suppose later in the overlap:

```text
w_A = 0.2

w_B = 0.8
```

Then:

```math
I_{final}
=
0.2(180)
+
0.8(150)
```

```math
=
156
```

Now the pixel is much closer to Image B.

---

This gradual transition removes the harsh seam.

---

## What Feathering Really Is

Conceptually:

```text
Image A Opacity

100%
↓
0%
```

while:

```text
Image B Opacity

0%
↓
100%
```

across the overlap region.

---

The two images fade into one another.

Exactly like a movie crossfade.

---

## Why Feathering Is Popular

Advantages:

```text
Easy To Implement

Extremely Fast

GPU Friendly

Produces Smooth Transitions
```

---

For this reason feathering is usually the first blending method implemented in custom panorama systems.

---

## Limitation

Feathering only blends:

```text
Pixel Intensities
```

It does not understand:

```text
Edges

Textures

Objects

Structures
```

If images are not perfectly aligned:

```text
Ghosting
```

appears.

Understanding this limitation leads directly to the next topic:

```text
Distance Transform Weight Maps
```

which explains where the blending weights actually come from.

---

# 13. Why Not Use Constant 50%-50% Blending?

After learning feathering, a common question arises:

```text
Why Not Simply Average Both Images?
```

In other words:

```math
I_{final}
=
\frac{
I_A + I_B
}{2}
```

for every pixel inside the overlap region.

At first glance this seems reasonable.

The seam disappears.

The implementation is simple.

The computation is extremely fast.

So why do panorama systems not use this approach?

---

## Constant Averaging

Suppose:

```text
Image A Pixel = 180

Image B Pixel = 150
```

Simple averaging gives:

```math
I_{final}
=
\frac{
180 + 150
}{2}
=
165
```

This appears to work.

---

Now imagine the overlap region contains:

```text
1000 × 1000 Pixels
```

Every pixel becomes:

```text
50% Image A

50% Image B
```

regardless of where it is located.

---

Visualization:

```text
Image A Contribution

50%
50%
50%
50%
50%
50%
50%
```

```text
Image B Contribution

50%
50%
50%
50%
50%
50%
50%
```

The weights never change.

---

## The Fundamental Problem

Not all pixels are equally reliable.

Consider a photograph.

The center of the image is usually:

```text
Well Exposed

Less Distorted

Sharper

More Reliable
```

---

Near the edges of the image we often observe:

```text
Lens Distortion

Perspective Errors

Vignetting

Blur

Exposure Changes
```

---

Therefore:

```text
Center Pixels
```

should generally be trusted more than:

```text
Boundary Pixels
```

---

Constant averaging ignores this completely.

It treats:

```text
Best Pixels
```

and

```text
Worst Pixels
```

exactly the same.

---

## Example: Exposure Difference

Suppose the overlap region looks like:

```text
Image A

180 180 180 180 180
```

```text
Image B

150 150 150 150 150
```

Constant averaging produces:

```text
165 165 165 165 165
```

---

Notice what happened.

Neither image survives.

We create an entirely new brightness value.

The overlap region becomes a compromise.

---

Sometimes this looks acceptable.

Sometimes it creates a visible haze.

---

## Example: Slight Misalignment

Suppose a rock appears in both images.

Image A:

```text
Rock Center = x=100
```

Image B:

```text
Rock Center = x=103
```

The alignment is close but not perfect.

---

Constant averaging produces:

```text
50% Rock A

50% Rock B
```

Result:

```text
Blurred Rock
```

---

Visualization:

```text
Rock      Rock
```

becomes:

```text
Ghost Rock
```

---

The entire overlap region appears soft.

---

## Another Problem: Sharp Features

Consider text.

Image A:

```text
OPEN
```

Image B:

```text
OPEN
```

shifted by a few pixels.

---

Averaging creates:

```text
Blurry Text
```

because both versions contribute equally.

---

The human eye notices this immediately.

---

## What We Actually Want

Ideally:

```text
Image A Dominates
```

where Image A is strongest.

---

And:

```text
Image B Dominates
```

where Image B is strongest.

---

Instead of:

```text
50%-50%
```

everywhere.

---

We want:

```text
100%-0%
```

at one side,

then gradually transition to:

```text
0%-100%
```

at the other side.

---

Visualization:

```text
Image A Weight

1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
```

```text
Image B Weight

0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
```

Now each image contributes most strongly where it is most trustworthy.

---

## Crossfade Analogy

Think of two songs.

Suppose Song A is ending.

Song B is beginning.

A music player does not usually do:

```text
50% Song A

50% Song B
```

for the entire transition.

---

Instead:

```text
Song A Slowly Fades Out
```

while:

```text
Song B Slowly Fades In
```

---

Visualization:

```text
Song A

100%
90%
80%
70%
...
0%
```

```text
Song B

0%
10%
20%
30%
...
100%
```

This sounds natural.

---

Image feathering works exactly the same way.

---

## The Missing Piece

We now understand:

```text
Weights Should Change Gradually
```

rather than remain fixed.

The next question becomes:

```text
How Do We Compute Those Weights?
```

Where do values such as:

```text
1.0

0.8

0.6

0.4

0.2

0.0
```

actually come from?

The answer is:

```text
Distance Transform Weight Maps
```

which are used by most feathering-based panorama systems to automatically generate smooth blending weights.

# 14. Distance Transform Based Weight Maps

In the previous section we learned that:

```text id="t4azk8"
Constant 50%-50% Blending
```

is usually a bad idea.

Instead we want:

```text id="upm7n8"
Image A
```

to dominate near the side where Image A is strongest.

And:

```text id="9tq4oe"
Image B
```

to dominate near the side where Image B is strongest.

This raises an important question:

```text id="v8lj5x"
Where Do The Blending Weights Come From?
```

---

## The Goal

We want to automatically generate:

```text id="vnl1qx"
Smoothly Changing Weights
```

such as:

```text id="6fw1w4"
1.0
0.9
0.8
0.7
0.6
0.5
...
0.0
```

without manually specifying them.

---

The most common solution is:

```text id="7o0rhv"
Distance Transform
```

---

## Intuition

Imagine a photograph.

```text id="e0qec9"
+----------------------+
|                      |
|                      |
|        Center        |
|                      |
|                      |
+----------------------+
```

---

The center of the image is usually:

```text id="kjxxhh"
More Reliable

Sharper

Less Distorted

Better Exposed
```

---

Near the boundaries we often observe:

```text id="99ib9m"
Lens Distortion

Vignetting

Motion Blur

Perspective Error
```

---

Therefore we would like:

```text id="mgxcpv"
Center Pixels
```

to receive:

```text id="t8r9b5"
High Weight
```

and:

```text id="j5w6ew"
Edge Pixels
```

to receive:

```text id="a4s57f"
Low Weight
```

---

## Measuring Distance From The Edge

Consider a one-dimensional image strip.

```text id="yfq9v5"
|----------------------|
```

Suppose width:

```text id="1isjlwm"
100 Pixels
```

---

The left boundary:

```text id="7h6uwm"
Distance = 0
```

---

The center:

```text id="7mnf8t"
Distance = 50
```

---

The right boundary:

```text id="k4jk0i"
Distance = 0
```

again.

---

Visualization:

```text id="3txa8k"
0
10
20
30
40
50
40
30
20
10
0
```

These values represent:

```text id="ekjkxh"
Distance To Nearest Edge
```

---

Notice something interesting.

The center automatically gets:

```text id="v6nkv8"
Largest Distance
```

while the boundaries receive:

```text id="z7nl9i"
Smallest Distance
```

---

Exactly what we want.

---

## Converting Distance Into Weight

Distances are usually normalized.

Suppose:

```text id="g3a5ch"
Maximum Distance = 50
```

---

Then:

```text id="1s9g5w"
Weight
=
Distance
/
Maximum Distance
```

---

Examples:

```text id="6jtlc4"
Distance = 50

Weight = 1.0
```

---

```text id="kjlwm0"
Distance = 25

Weight = 0.5
```

---

```text id="4rddkz"
Distance = 5

Weight = 0.1
```

---

```text id="ap2vq0"
Distance = 0

Weight = 0.0
```

---

This automatically generates a smooth ramp.

---

Visualization:

```text id="vptnj8"
1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
```

---

## Two Images Means Two Weight Maps

Remember:

```text id="1r3zns"
Image A
```

and

```text id="o7j2x5"
Image B
```

each occupy different regions of the panorama.

---

Therefore each image receives its own:

```text id="9lj1ra"
Distance Map
```

---

Image A:

```text id="n4ywtx"
Center → High Weight

Edges → Low Weight
```

---

Image B:

```text id="q1lt6q"
Center → High Weight

Edges → Low Weight
```

---

These weights are generated independently.

---

## What Happens In The Overlap Region?

Suppose overlap width:

```text id="drg8tt"
100 Pixels
```

---

Image A weights:

```text id="weuy9w"
1.0
0.9
0.8
0.7
0.6
0.5
0.4
0.3
0.2
0.1
0.0
```

---

Image B weights:

```text id="ajg1jq"
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
```

---

Notice:

```text id="v4mq8y"
One Image Fades Out

The Other Fades In
```

automatically.

---

No manual tuning is required.

---

## Why Distance Transform Works So Well

Distance transform produces weights that are:

```text id="9dy0dw"
Continuous

Smooth

Automatic

Geometry Aware
```

---

Pixels near the center naturally become:

```text id="m5lfca"
Trusted Pixels
```

---

Pixels near boundaries naturally become:

```text id="4zkjhp"
Less Trusted Pixels
```

---

The algorithm behaves much like a human editor.

If two photographs disagree:

```text id="g4rw4k"
Trust The Interior

Distrust The Boundaries
```

---

## Example

Suppose:

```text id="z44wdf"
Image A Pixel = 180

Weight = 0.8
```

---

Image B:

```text id="3f9ysq"
Pixel = 150

Weight = 0.2
```

---

Normalize:

```math id="lmhqod"
w_A
=
\frac{0.8}{0.8+0.2}
=
0.8
```

```math id="vjlwm1"
w_B
=
\frac{0.2}{0.8+0.2}
=
0.2
```

---

Final pixel:

```math id="h2mxfr"
I_{final}
=
0.8(180)
+
0.2(150)
```

```math id="7sqyhl"
=
174
```

---

Since Image A is farther from its boundary, it contributes more.

---

## Visual Interpretation

Imagine transparency masks.

Image A opacity:

```text id="1s4v0n"
100%
90%
80%
70%
...
0%
```

---

Image B opacity:

```text id="jjajmf"
0%
10%
20%
30%
...
100%
```

---

Distance transform automatically creates these opacity maps.

---

## Computational Cost

Distance transforms are relatively inexpensive.

They are usually computed:

```text id="k1l5xt"
Once Per Image
```

before blending begins.

---

The resulting weight maps are then reused for every pixel.

---

This makes feathering:

```text id="0jz3v4"
Very Fast

Very GPU Friendly

Easy To Implement
```

---

## Limitation

Even though distance-based feathering removes seams very effectively:

```text id="vv8d6j"
It Still Assumes
The Images Are Perfectly Aligned
```

---

If objects are slightly offset:

```text id="hlb5hy"
Tree A

Tree B
```

or:

```text id="a8dfcc"
Rock A

Rock B
```

then feathering will average them together.

The result is:

```text id="tux1jc"
Ghosting

Blur

Double Edges
```

Understanding this limitation leads to the next topic:

```text id="dbd5ol"
Why Feathering Works
```

and eventually:

```text id="nqjjqj"
Why Feathering Ultimately Fails
```

for challenging panoramas.

# 15. Why Feathering Works

At this point we understand:

```text
Feathering
=
Weighted Averaging
```

using distance-based weight maps.

We also know that feathering removes visible seams.

The natural question is:

```text
Why Does Feathering Look So Good?
```

After all, we are simply averaging pixel values.

Why does something so simple often produce surprisingly good panoramas?

---

## Understanding The Human Eye

The answer lies in:

```text
Human Visual Perception
```

Humans do not inspect images pixel by pixel.

Instead, our visual system is extremely sensitive to:

```text
Sudden Changes
```

and much less sensitive to:

```text
Gradual Changes
```

---

Consider this intensity profile:

```text
150 150 150 150 | 180 180 180 180
```

At the center:

```text
150 → 180
```

there is an abrupt jump.

---

Your brain immediately notices:

```text
SEAM
```

even though the difference is only:

```text
30 Intensity Levels
```

---

## Sharp Transitions Attract Attention

Human vision evolved to detect:

```text
Edges

Boundaries

Discontinuities
```

because they often correspond to:

```text
Objects

Obstacles

Movement
```

in the real world.

---

As a result:

```text
Sudden Brightness Change
```

immediately attracts attention.

---

In a panorama:

```text
Visible Seam
```

appears unnatural.

The brain instantly recognizes that two images were stitched together.

---

## What Feathering Changes

Instead of:

```text
150 150 150 150 | 180 180 180 180
```

feathering produces:

```text
150
153
156
159
162
165
168
171
174
177
180
```

---

Now there is no abrupt boundary.

The brightness changes gradually.

---

The eye perceives:

```text
One Continuous Surface
```

rather than:

```text
Two Separate Images
```

---

## Visualizing The Transition

Without feathering:

```text
████████████|████████████
```

There is a visible border.

---

With feathering:

```text
████████▒▒▒▒▓▓▓▓████████
```

The transition becomes smooth.

---

Instead of seeing:

```text
Image A

Image B
```

the viewer sees:

```text
One Panorama
```

---

## Why The Center Of The Overlap Looks Natural

Suppose:

```text
Image A Pixel = 180

Image B Pixel = 150
```

At the center of the overlap:

```text
w_A = 0.5

w_B = 0.5
```

Result:

```math
I_{final}
=
0.5(180)
+
0.5(150)
=
165
```

---

Notice something important.

The resulting value:

```text
165
```

lies between:

```text
150
```

and

```text
180
```

---

This creates a smooth bridge between the images.

The eye interprets this as:

```text
Natural Variation
```

rather than:

```text
Artificial Boundary
```

---

## Why Distance-Based Weights Help

Suppose two images overlap.

The center of each image usually contains:

```text
Best Exposure

Lowest Distortion

Highest Sharpness
```

---

Near image boundaries we often see:

```text
Lens Distortion

Vignetting

Perspective Error
```

---

Distance transforms automatically produce:

```text
High Weight
```

for reliable pixels

and

```text
Low Weight
```

for less reliable pixels.

---

This means feathering does not merely smooth the seam.

It also tends to favor:

```text
Better Pixels
```

during blending.

---

## A Real-World Analogy

Imagine two painters painting the same wall.

Painter A did a better job on the left side.

Painter B did a better job on the right side.

---

Instead of drawing a hard line:

```text
Left Painter | Right Painter
```

we gradually mix their work.

---

The transition becomes almost invisible.

---

Feathering behaves the same way.

---

## Why Feathering Is Popular

Feathering has several advantages:

### Simple

Only weighted averaging is required.

---

### Fast

Each pixel requires:

```math
w_A I_A
+
w_B I_B
```

which is extremely inexpensive.

---

### Parallelizable

Every pixel can be processed independently.

This makes feathering ideal for:

```text
CUDA

OpenCL

GPU Processing
```

---

### Good Enough For Many Cases

If:

```text
Alignment Is Accurate

Exposure Differences Are Small
```

feathering often produces visually pleasing panoramas.

---

## When Feathering Looks Excellent

Feathering performs very well when:

```text
Camera Motion Is Small
```

---

```text
Overlap Is Large
```

---

```text
Objects Are Far Away
```

---

```text
Images Are Well Aligned
```

---

In these situations:

```text
Seams Become Nearly Invisible
```

with minimal computational cost.

---

## The Hidden Assumption

Feathering quietly assumes:

```text
Both Images Show
The Same Structure
At The Same Location
```

---

Example:

```text
Tree Trunk
```

in Image A

must align with:

```text
Tree Trunk
```

in Image B.

---

If this assumption is true:

```text
Feathering Works Beautifully
```

---

If this assumption is violated:

```text
Problems Begin
```

---

## The Beginning Of The Ghosting Problem

Suppose:

```text
Tree In Image A
```

appears at:

```text
x = 100
```

---

The same tree in Image B appears at:

```text
x = 103
```

because of imperfect alignment.

---

Feathering does not understand:

```text
Trees

Buildings

People

Edges
```

It only understands:

```text
Pixel Values
```

---

Therefore it simply averages both versions together.

---

Instead of:

```text
One Tree
```

you obtain:

```text
Two Semi-Transparent Trees
```

This artifact is called:

```text
Ghosting
```

and it is the primary weakness of feathering.

---

## Summary

Feathering works because:

```text
Human Vision Hates
Sudden Changes
```

but tolerates:

```text
Gradual Changes
```

---

Distance-based blending converts:

```text
Sharp Seams
```

into:

```text
Smooth Transitions
```

that appear natural to the viewer.

---

Advantages:

```text
Simple

Fast

GPU Friendly

Visually Effective
```

---

However feathering assumes:

```text
Perfect Alignment
```

between overlapping images.

When that assumption fails:

```text
Ghosting Appears
```

Understanding ghosting is the next step toward understanding why professional panorama systems use:

```text
Multi-Band Blending
```

instead of simple feathering.

# 16. The Ghosting Problem

In the previous section we learned that feathering works surprisingly well because it converts:

```text id="r6p0wo"
Abrupt Changes
```

into:

```text id="ndzvwi"
Gradual Changes
```

that are much less noticeable to the human eye.

---

When the images are perfectly aligned:

```text id="5o92a2"
Feathering Produces Excellent Results
```

The seam disappears.

The panorama looks natural.

Everything appears continuous.

---

Unfortunately real images are rarely perfect.

Even after:

```text id="g1tyk5"
Feature Matching

RANSAC

Homography Estimation

Inverse Warping
```

small alignment errors almost always remain.

---

## Understanding Misalignment

Consider a rock that appears in both images.

Image A:

```text id="k56fqp"
Rock Center

x = 100
```

---

Image B:

```text id="a2yxjn"
Rock Center

x = 103
```

---

The difference is only:

```text id="r0vg8o"
3 Pixels
```

which seems tiny.

---

To a human observer:

```text id="kqxyjy"
3 Pixels
```

often appears insignificant.

---

However feathering blends pixels mathematically.

Even tiny shifts become visible.

---

## Perfect Alignment Case

Suppose the rock occupies:

```text id="dx3grw"
Exactly The Same Location
```

in both images.

Visualization:

```text id="v0n8h7"
Rock
Rock
```

---

Feathering combines:

```text id="6wkg5h"
Rock A

+

Rock B
```

---

Result:

```text id="jz7cv5"
Rock
```

The output remains sharp.

Everything looks correct.

---

## Slight Misalignment Case

Now suppose:

```text id="v9nsu8"
Rock A
```

appears slightly left.

---

And:

```text id="zm1x6s"
Rock B
```

appears slightly right.

Visualization:

```text id="7x97c2"
Rock      Rock
```

---

Feathering still performs:

```text id="b7n7t5"
50% Rock A

+

50% Rock B
```

---

Result:

```text id="ah34nh"
Blurred Rock
```

or

```text id="3mkwy5"
Double Rock
```

---

Instead of one object:

```text id="xwz9jv"
Two Semi-Transparent Copies
```

appear.

---

This artifact is called:

```text id="n0sl4s"
Ghosting
```

---

## Why Is It Called Ghosting?

Because the duplicated object appears:

```text id="ytrqq3"
Faint

Semi Transparent

Shadow Like
```

---

Example:

```text id="3jhk95"
Tree
```

becomes:

```text id="ry1xnk"
Tree    Tree
```

with both copies partially visible.

---

The second copy looks like:

```text id="7wd06v"
A Ghost Image
```

which is where the name originates.

---

## Feathering Does Not Understand Objects

This is the most important thing to understand.

Feathering only knows:

```text id="n7fr0e"
Pixel Intensities
```

---

It does not know:

```text id="0jl7lf"
Trees

Cars

Buildings

People

Roads
```

---

It has no concept of:

```text id="j2q0l5"
Object Boundaries
```

or:

```text id="2aw9z0"
Image Content
```

---

The algorithm simply sees:

```text id="7nnl1i"
Numbers
```

and computes:

```math id="vv65zc"
w_A I_A
+
w_B I_B
```

for every pixel.

---

Because of this:

```text id="xnh9f3"
Misaligned Objects
```

are blended together instead of corrected.

---

## Example With A Tree

Suppose:

```text id="0u4q5p"
Tree Center In Image A

x = 500
```

---

And:

```text id="ysr5zy"
Tree Center In Image B

x = 505
```

---

Feathering computes:

```text id="3whf5u"
50% Tree A

+

50% Tree B
```

---

The result becomes:

```text id="6mjlwm"
Double Tree
```

---

The leaves become blurry.

The branches appear duplicated.

The image loses sharpness.

---

## Why Ghosting Is Most Visible Near Edges

Ghosting becomes especially noticeable around:

```text id="gj8rd7"
High Contrast Boundaries
```

such as:

```text id="mjlwm9"
Tree Branches

Building Edges

Text

Street Signs

Window Frames
```

---

Why?

Because the human eye is extremely sensitive to:

```text id="9ldk4y"
Sharp Edges
```

---

When two edges become slightly offset:

```text id="zvt5hy"
Double Edge
```

appears immediately.

---

Even a small misalignment becomes obvious.

---

## Example With Text

Image A:

```text id="m7ot98"
OPEN
```

---

Image B:

```text id="8hl2ze"
OPEN
```

shifted slightly.

---

Feathering creates:

```text id="3nmzzr"
Blurry Text
```

---

Human observers notice this instantly.

---

## Why Homography Cannot Always Fix This

Many beginners assume:

```text id="ft5h1v"
Better Homography
=
No Ghosting
```

---

This is not always true.

Homography assumes:

```text id="2rzjhb"
One Global Geometric Transformation
```

between the images.

---

Real scenes often violate this assumption.

---

Examples:

```text id="7buwxq"
Moving Cars

Walking People

Tree Branches Moving In Wind

Parallax

Depth Differences
```

---

Even a perfect homography cannot align all of these simultaneously.

---

Therefore some ghosting remains unavoidable.

---

## Why Feathering Eventually Reaches Its Limit

Feathering solves:

```text id="wrb2i2"
Brightness Differences
```

extremely well.

---

It solves:

```text id="gwb1i7"
Exposure Changes
```

quite well.

---

It solves:

```text id="qjuzwg"
Visible Seams
```

very well.

---

However it does not solve:

```text id="u3t3v2"
Structural Misalignment
```

---

When objects differ between images:

```text id="h1n37j"
Feathering Simply Averages Them
```

---

The result becomes:

```text id="13bmkf"
Blur

Double Edges

Ghosting
```

---

## The Need For Something Smarter

To reduce ghosting we need a blending method that understands:

```text id="wvqgrm"
Large Smooth Regions
```

and

```text id="v95m3o"
Fine Sharp Details
```

should not be blended in exactly the same way.

---

Human vision is very sensitive to:

```text id="r8f24n"
Brightness Changes
```

but surprisingly tolerant of:

```text id="j9wybc"
Small Changes In Fine Detail
```

---

Professional panorama systems exploit this fact.

Instead of blending the entire image uniformly, they separate the image into:

```text id="6e1hbm"
Different Frequency Bands
```

and blend each band differently.

---

This approach is called:

```text id="3ttzwm"
Multi-Band Blending
```

and it is one of the primary reasons professional panorama software produces nearly invisible seams.

