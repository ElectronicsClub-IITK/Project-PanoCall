# 17. Why Human Vision Notices Seams

Before understanding multi-band blending, we first need to understand:

```text id="l4h7m9"
How Human Vision Works
```

The reason multi-band blending is so effective has very little to do with image processing and a lot to do with:

```text id="x8v3n2"
Human Perception
```

---

## A Surprising Observation

Consider two situations.

---

### Situation 1

A sudden brightness change:

```text id="r5k8w1"
150 150 150 150 | 180 180 180 180
```

---

Most people immediately notice:

```text id="m2c6z7"
SEAM
```

even when they are not actively looking for it.

---

Their eyes are automatically drawn to:

```text id="v9n4k3"
The Boundary
```

between the two regions.

---

### Situation 2

A gradual brightness change:

```text id="j1t8y5"
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

Most people perceive this as:

```text id="f7p3w8"
A Smooth Transition
```

rather than:

```text id="d4x2r6"
Two Different Regions
```

---

This observation is the foundation of modern blending algorithms.

---

## The Human Eye Is An Edge Detector

Human vision evolved to recognize:

```text id="k8q1m7"
Objects

Shapes

Boundaries

Movement
```

in the environment.

---

To accomplish this, our brains are extremely sensitive to:

```text id="c3z7n4"
Sudden Changes
```

in brightness and color.

---

A sudden transition often indicates:

```text id="r6t2v8"
Object Boundary

Obstacle

Danger

Movement
```

---

As a result:

```text id="p4x8n2"
Sharp Intensity Changes
```

immediately attract attention.

---

## Why Seams Are Easy To Notice

Consider a panorama seam.

```text id="m8v1c5"
Image A
|
Image B
```

---

Even if the images are aligned perfectly:

```text id="g2k7w4"
Brightness Difference
```

may still exist.

---

Example:

```text id="n3t6p8"
Image A = 150

Image B = 180
```

---

The seam creates:

```text id="s1y4m7"
Abrupt Contrast Change
```

which our visual system interprets as:

```text id="e9q8w2"
A Boundary
```

---

The panorama suddenly looks stitched together.

---

## Why Feathering Helps

Feathering replaces:

```text id="h7c2n6"
Abrupt Transition
```

with:

```text id="a5m9v3"
Gradual Transition
```

---

Instead of:

```text id="w3r7t1"
150 | 180
```

we obtain:

```text id="u8p4j5"
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

The edge disappears.

---

The brain no longer detects:

```text id="k5n8m4"
A Sudden Boundary
```

and therefore perceives:

```text id="j9v2r7"
One Continuous Image
```

---

## Not All Image Information Is Equally Important

Now consider two types of image information.

---

### Large Smooth Regions

Examples:

```text id="v4m8p2"
Blue Sky

Road Surface

Wall

Grass Field

Lighting Gradient
```

---

These regions change slowly.

---

Small errors are often very noticeable.

---

For example:

```text id="s2q9w6"
Sky Brightness
```

changing abruptly creates an obvious seam.

---

### Fine Details

Examples:

```text id="n7k4p8"
Tree Leaves

Grass Texture

Building Edges

Hair

Text
```

---

These change rapidly.

---

Interestingly:

```text id="p8t1v5"
Human Vision Is More Forgiving
```

with respect to small inconsistencies in these details.

---

## A Thought Experiment

Imagine two panoramas.

---

### Panorama A

Perfect detail alignment.

But:

```text id="y3n7m2"
Brightness Jumps Suddenly
```

across the seam.

---

Most people immediately notice:

```text id="q5r8w1"
The Seam
```

---

### Panorama B

Brightness transitions smoothly.

But some leaf details differ slightly.

---

Most viewers perceive:

```text id="c9m2p6"
A Much Better Panorama
```

even though technically the alignment is less accurate.

---

Why?

Because human vision is far more sensitive to:

```text id="d7v3k9"
Low Frequency Brightness Changes
```

than to:

```text id="m4t8q2"
Small High Frequency Errors
```

---

This observation is the key insight behind:

```text id="f1w6r8"
Multi-Band Blending
```

---

## Another Analogy: Looking From Far Away

Imagine viewing a panorama from:

```text id="v8n5m1"
50 Meters Away
```

---

Fine details disappear.

---

You can no longer see:

```text id="r4k2p7"
Leaves

Text

Grass
```

---

What remains visible?

```text id="q7m3t8"
Overall Brightness

Large Shapes

Color Differences
```

---

If the seam is visible at this distance:

```text id="k2v9m5"
The Panorama Looks Bad
```

regardless of how sharp the details are.

---

This is why large-scale brightness consistency is so important.

---

## What This Means For Blending

Ideally:

```text id="z6r8n3"
Lighting
```

should be blended:

```text id="c4m7p1"
Very Gradually
```

across a wide region.

---

But:

```text id="y1t5v8"
Edges

Textures

Fine Details
```

should remain:

```text id="x3p8m4"
Sharp
```

---

Feathering does not make this distinction.

It blends:

```text id="n8k3w7"
Everything
```

the same way.

---

This is the fundamental reason feathering eventually fails.

---

## The Key Insight

Human vision is extremely sensitive to:

```text id="p7v2m9"
Large Scale Brightness Changes
```

but relatively tolerant of:

```text id="r5m8k1"
Small Detail Differences
```

---

If we could somehow separate:

```text id="q1t6n4"
Lighting
```

from:

```text id="m9w3p7"
Details
```

we could blend them differently.

---

This is exactly what modern panorama systems do.

They separate the image into:

```text id="v4k7n2"
Low Frequencies

Medium Frequencies

High Frequencies
```

and process each independently.

---

This idea leads directly to:

```text id="n6r8w5"
Image Frequencies
```

which is the foundation of:

```text id="y2m5p9"
Multi-Band Blending
```

# 18. Understanding Image Frequencies

In the previous section we learned something surprising:

```text id="u3m8v2"
Human Vision Is More Sensitive
To Large Brightness Changes
Than Small Detail Changes
```

This idea is the foundation of:

```text id="w7n2p5"
Multi-Band Blending
```

To understand multi-band blending, we first need to understand:

```text id="r5k8m1"
Image Frequencies
```

---

## What Does Frequency Mean?

Most students hear:

```text id="x9v4q7"
Frequency
```

and immediately think about:

```text id="j2m7t4"
Signals

Fourier Transforms

Sinusoids
```

---

The same concept exists in images.

---

Instead of asking:

```text id="v1n5r8"
How Fast Does A Signal Change Over Time?
```

we ask:

```text id="q8m3p2"
How Fast Does Intensity Change Across Space?
```

---

## Low Frequency Example

Consider a smooth sky.

```text id="n7k4w9"
150
151
152
153
154
155
156
157
158
159
160
```

Notice:

```text id="z5t8p1"
Brightness Changes Slowly
```

from one pixel to the next.

---

This is called:

```text id="c2m9v7"
Low Frequency Content
```

---

Examples:

```text id="d4w8n3"
Sky

Walls

Roads

Lighting Gradients

Large Smooth Surfaces
```

---

Low frequencies represent:

```text id="y7k3m6"
Overall Appearance
```

of the image.

---

## High Frequency Example

Now consider an edge.

```text id="j9v2r5"
0 0 0 0 0 255 255 255 255
```

---

Brightness changes extremely quickly.

---

One pixel:

```text id="t4m8p3"
0
```

---

Next pixel:

```text id="g7n1w9"
255
```

---

This is:

```text id="q6v5m2"
High Frequency Content
```

---

Examples:

```text id="m8p4r7"
Text

Tree Branches

Leaves

Hair

Building Edges

Sharp Corners
```

---

High frequencies represent:

```text id="f2w7n5"
Fine Details
```

inside the image.

---

## Medium Frequencies

Between the two extremes we have:

```text id="r8m1p4"
Medium Frequencies
```

---

Examples:

```text id="u5n9v2"
Building Shapes

Rock Boundaries

Large Tree Branches

Road Markings
```

---

These represent structures larger than texture but smaller than lighting variations.

---

## A Useful Analogy: Music

Imagine a song.

A song contains:

```text id="m3v8n7"
Bass

Vocals

Treble
```

---

Each occupies a different frequency range.

---

Bass:

```text id="p6w2m9"
Slow Oscillations
```

---

Treble:

```text id="y4n8r3"
Rapid Oscillations
```

---

You can adjust each independently.

---

Images work exactly the same way.

---

Low frequencies:

```text id="v7k5m1"
Lighting

Color

Large Shapes
```

---

High frequencies:

```text id="r1p8w6"
Edges

Textures

Details
```

---

Multi-band blending manipulates each frequency range separately.

---

## Why Frequencies Matter For Stitching

Suppose two images overlap.

Image A:

```text id="z9m3n7"
Brightness = 180
```

---

Image B:

```text id="w5k8p2"
Brightness = 150
```

---

This difference belongs mostly to:

```text id="c8v2m4"
Low Frequencies
```

because brightness changes slowly.

---

Now consider a tree branch.

Image A:

```text id="x2n9w5"
Branch Edge
```

---

Image B:

```text id="f7m4p1"
Branch Edge
```

shifted slightly.

---

This difference belongs mostly to:

```text id="t8w5n3"
High Frequencies
```

because edges change rapidly.

---

## The Problem With Feathering

Feathering treats everything equally.

---

It blends:

```text id="y1v7m4"
Sky
```

the same way it blends:

```text id="r4k9p8"
Tree Branches
```

---

It blends:

```text id="n3w8m1"
Lighting
```

the same way it blends:

```text id="v5p2r7"
Text
```

---

This is often not what we want.

---

Suppose:

```text id="m7n4v2"
Sky Brightness Difference
```

exists between images.

---

We want that difference to disappear gradually.

---

However suppose:

```text id="d2w9p5"
Tree Edge
```

exists.

---

We want that edge to remain sharp.

---

Feathering cannot distinguish between the two.

---

## What We Really Want

Ideally:

```text id="z8m5r3"
Lighting
```

should be blended over:

```text id="j4n7v1"
Very Wide Regions
```

because the eye is sensitive to brightness changes.

---

Meanwhile:

```text id="q5p9m2"
Edges
```

should be blended over:

```text id="r7v3n8"
Very Narrow Regions
```

so that they remain sharp.

---

This immediately suggests a strategy.

---

Instead of blending:

```text id="x1k8m6"
One Image
```

we separate the image into:

```text id="c6v2p9"
Low Frequencies

Medium Frequencies

High Frequencies
```

and blend each differently.

---

## Frequency Separation

Imagine breaking an image into layers.

Layer 1:

```text id="w9m4n2"
Fine Details
```

---

Layer 2:

```text id="r2v8p7"
Medium Structures
```

---

Layer 3:

```text id="n7m1w5"
Large Shapes
```

---

Layer 4:

```text id="y5p8v3"
Lighting And Color
```

---

Now we can choose:

```text id="j8n4m7"
Different Blending Widths
```

for every layer.

---

This is the core idea behind:

```text id="x3v7p2"
Multi-Band Blending
```

---

## How Do We Separate Frequencies?

This is where image pyramids enter the picture.

We build:

```text id="m5n9w1"
Gaussian Pyramids
```

to isolate low frequencies.

---

Then we build:

```text id="r8v2p4"
Laplacian Pyramids
```

to isolate different detail levels.

---

These pyramids allow us to blend:

```text id="k1m7v9"
Lighting Slowly
```

while blending:

```text id="q4p8n2"
Fine Details Sharply
```

---

This is the reason professional panorama software can produce:

```text id="d7v5m3"
Smooth Lighting

Sharp Edges

Invisible Seams
```

simultaneously.

---

## Summary

Images contain different frequency components.

---

Low Frequencies:

```text id="h2n8p5"
Lighting

Color

Large Smooth Regions
```

---

High Frequencies:

```text id="v9m3w7"
Edges

Textures

Fine Details
```

---

Feathering blends all frequencies equally.

---

Professional panorama systems instead:

```text id="r5v1p8"
Separate Frequencies

Blend Them Independently

Combine Them Back Together
```

---

The first step in this process is constructing a:

```text id="j6m8n4"
Gaussian Pyramid
```

which we will study next.

# 19. Gaussian Pyramid: Separating Large Structures From Fine Details

In the previous section we learned that images contain:

```text
Low Frequencies
```

and

```text
High Frequencies
```

---

Low frequencies correspond to:

```text
Lighting

Color

Large Smooth Regions
```

---

High frequencies correspond to:

```text
Edges

Textures

Fine Details
```

---

To blend these frequencies differently, we first need a way to separate them.

The most common tool for doing this is:

```text
Gaussian Pyramid
```

---

## Why Do We Need A Gaussian Pyramid?

Suppose we have an image:

```text
Mountain
Trees
Sky
```

All of these exist together in one image.

---

Currently:

```text
Lighting

Mountain Shape

Tree Branches

Leaf Details
```

are mixed together.

---

We need a way to gradually remove details until only large structures remain.

---

Think of it like looking at a photograph from farther and farther away.

---

Close up:

```text
Leaves

Grass

Small Rocks

Textures
```

are visible.

---

Move farther away:

```text
Leaves disappear
```

---

Move even farther:

```text
Grass disappears
```

---

Move farther:

```text
Only mountain shape remains
```

---

Move very far:

```text
Only sky brightness remains
```

---

A Gaussian Pyramid mathematically simulates this process.

---

# What Is A Gaussian Pyramid?

A Gaussian Pyramid is a collection of progressively:

```text
Blurred

And

Downsampled
```

versions of an image.

---

Visualization:

```text
Level 0
Original Image

↓

Level 1
Blurred + Smaller

↓

Level 2
More Blurred + Smaller

↓

Level 3
Even More Blurred + Smaller

↓

Level 4
Very Blurred + Very Small
```

---

Each level contains less detail than the previous one.

---

# Step 1: Gaussian Blur

We start with the original image.

Example:

```text
1920 × 1080
```

---

Apply a Gaussian filter.

---

What does Gaussian blur do?

Imagine replacing each pixel by an average of nearby pixels.

---

Example:

Original:

```text
100 100 100 255 255 255
```

---

After blur:

```text
100 120 150 200 230 255
```

---

Notice:

```text
Sharp Edges Become Softer
```

---

Tiny details begin to disappear.

---

# Why Is It Called Gaussian?

The averaging is not uniform.

Nearby pixels receive:

```text
Large Weight
```

while distant pixels receive:

```text
Small Weight
```

---

The weights follow a Gaussian distribution:

```math
G(x,y)
=
\frac{1}{2\pi\sigma^2}
e^{-\frac{x^2+y^2}{2\sigma^2}}
```

---

You do not need to memorize the equation.

The important idea is:

```text
Closer Pixels Influence More
```

than farther pixels.

---

This creates smooth blur without introducing artifacts.

---

# Step 2: Downsampling

After blurring:

```text
We Shrink The Image
```

typically by:

```text
Factor Of 2
```

---

Example:

```text
1920 × 1080
```

becomes:

```text
960 × 540
```

---

Why blur before shrinking?

---

Suppose we directly shrink:

```text
Black White Black White Black White
```

---

Without blur:

```text
Information Is Lost
```

randomly.

---

This creates:

```text
Aliasing
```

artifacts.

---

Blurring removes very fine details first.

Then shrinking becomes safe.

---

# Building The Pyramid

Suppose we start with:

```text
1920 × 1080
```

---

Level 0:

```text
1920 × 1080
```

---

Level 1:

```text
960 × 540
```

---

Level 2:

```text
480 × 270
```

---

Level 3:

```text
240 × 135
```

---

Level 4:

```text
120 × 67
```

---

Visualization:

```text
Original
 ↓
Blur + Shrink
 ↓
Blur + Shrink
 ↓
Blur + Shrink
 ↓
Blur + Shrink
```

---

This stack is called:

```text
Gaussian Pyramid
```

---

# What Happens To Details?

Imagine a tree.

Level 0:

```text
Leaves
Branches
Trunk
```

all visible.

---

Level 1:

```text
Leaves begin disappearing
```

---

Level 2:

```text
Most leaves gone

Branches visible
```

---

Level 3:

```text
Only trunk and large branches
```

---

Level 4:

```text
Only rough tree shape
```

---

Fine details disappear first.

Large structures survive longest.

---

# What Does Each Level Represent?

Upper Levels:

```text
High Resolution

More Details

More High Frequencies
```

---

Lower Levels:

```text
Less Detail

Mostly Low Frequencies
```

---

Bottom Levels:

```text
Lighting

Large Shapes

Color Gradients
```

---

Exactly the information we want to treat differently during blending.

---

# Example: Sky And Trees

Original Image:

```text
Blue Sky
Tree Leaves
Branches
```

---

Level 0:

```text
Everything Visible
```

---

Level 1:

```text
Leaves Slightly Softer
```

---

Level 2:

```text
Leaves Mostly Gone
```

---

Level 3:

```text
Branches Visible
```

---

Level 4:

```text
Only Sky Brightness
And Tree Silhouette
```

---

Notice how frequencies separate naturally.

---

# Why Gaussian Pyramid Is Useful For Blending

Suppose two images have:

```text
Different Brightness
```

---

Brightness differences belong mainly to:

```text
Low Frequencies
```

---

Those frequencies survive in:

```text
Lower Pyramid Levels
```

---

Suppose two images have:

```text
Different Leaf Details
```

---

Those belong mainly to:

```text
High Frequencies
```

---

Those exist mostly in:

```text
Upper Pyramid Levels
```

---

If we can isolate these frequencies:

```text
We Can Blend Them Differently
```

---

That is exactly what Multi-Band Blending does.

---

# But There Is A Problem

A Gaussian Pyramid only gives us:

```text
Blurred Images
```

---

It does not directly tell us:

```text
Which Details
Were Removed
```

between levels.

---

For example:

```text
Level 0
```

contains:

```text
Tree + Sky
```

---

```text
Level 1
```

contains:

```text
Blurred Tree + Sky
```

---

The difference between them contains:

```text
Tree Details
```

which is exactly what we need.

---

This leads to the next structure:

```text
Laplacian Pyramid
```

which stores the details lost between Gaussian levels and forms the foundation of true multi-band blending.

# 20. Laplacian Pyramid: Extracting Image Details

In the previous section we built a:

```text id="o6h2pd"
Gaussian Pyramid
```

which contains progressively:

```text id="b4g9rn"
Blurred

And

Downsampled
```

versions of the image.

---

The Gaussian Pyramid is excellent for storing:

```text id="g9m3tx"
Low Frequencies
```

such as:

```text id="q7v5mw"
Lighting

Color

Large Shapes
```

---

However there is a problem.

---

## What Information Is Missing?

Suppose we have:

```text id="n8t2wp"
Level 0
```

Original image:

```text id="9w4kxm"
Tree
Sky
Mountain
```

---

And:

```text id="f2m8vr"
Level 1
```

Blurred image:

```text id="p6t3zn"
Blurry Tree
Sky
Mountain
```

---

Notice something.

The blur operation removed:

```text id="v5r9qk"
Fine Details
```

such as:

```text id="u3n7mw"
Leaves

Grass

Textures

Sharp Edges
```

---

Those details did not disappear.

They were simply discarded.

---

If we want to perform:

```text id="k1v8mp"
Multi-Band Blending
```

we must know:

```text id="h4t6qr"
Exactly What Details
Were Removed
```

at every scale.

---

This is the purpose of the:

```text id="c9m2vx"
Laplacian Pyramid
```

---

# Core Idea

Suppose:

```text id="w7p4nm"
Original Image
=
Details
+
Blurred Version
```

---

Then:

```text id="x3m8vt"
Details
=
Original
−
Blurred Version
```

---

This subtraction is the entire foundation of the Laplacian Pyramid.

---

# Simple Example

Imagine a one-dimensional signal.

Original:

```text id="t8q4mk"
100
120
140
160
180
200
```

---

Blurred:

```text id="m2v7wp"
110
125
140
155
170
185
```

---

Subtract:

```text id="v9t2rm"
-10
-5
0
5
10
15
```

---

The remaining values describe:

```text id="q4n8vx"
Changes

Edges

Details
```

that were lost during blurring.

---

This difference image is what a Laplacian level stores.

---

# Building A Laplacian Pyramid

Suppose we already have a Gaussian Pyramid:

```text id="r6m3tw"
G0
G1
G2
G3
G4
```

---

Where:

```text id="n1v5qx"
G0
```

is the original image.

---

And:

```text id="z8p4tm"
G4
```

is heavily blurred.

---

We now create:

```text id="u7m2vr"
L0
L1
L2
L3
```

using subtraction.

---

# Step 1: Expand The Next Gaussian Level

Suppose:

```text id="h5v9qx"
G0
```

has size:

```text id="a2m7wr"
1920 × 1080
```

---

And:

```text id="v3p8tm"
G1
```

has size:

```text id="g7n4qx"
960 × 540
```

---

We cannot subtract them directly.

The dimensions differ.

---

Therefore:

```text id="j9m5vw"
G1
```

must first be:

```text id="m4t8qn"
Upsampled
```

back to:

```text id="f1v7mp"
1920 × 1080
```

---

This process is called:

```text id="w8m3tx"
Expansion
```

---

# Step 2: Subtract

Now both images have identical dimensions.

---

Compute:

```math id="x5t9qn"
L_0
=
G_0
-
Expand(G_1)
```

---

What remains?

---

Mostly:

```text id="k2v8mw"
Fine Details

Edges

Textures
```

---

because the large smooth structures cancel out.

---

# Repeat For Every Level

Similarly:

```math id="t7m2vx"
L_1
=
G_1
-
Expand(G_2)
```

---

```math id="p8v4tm"
L_2
=
G_2
-
Expand(G_3)
```

---

```math id="n6m7qw"
L_3
=
G_3
-
Expand(G_4)
```

---

The final Gaussian level:

```text id="g3v8tx"
G4
```

is usually kept unchanged.

---

It represents:

```text id="c8m2vw"
Very Low Frequencies
```

such as:

```text id="q7t4mx"
Lighting

Color

Large Shapes
```

---

# What Does Each Laplacian Level Contain?

Think of the pyramid as a frequency decomposition.

---

## L0

Contains:

```text id="v5m8qn"
Tiny Details

Fine Textures

Leaf Edges

Hair

Text
```

---

These are the highest frequencies.

---

## L1

Contains:

```text id="w2v7mx"
Small Structures

Branches

Rock Edges
```

---

Slightly lower frequencies.

---

## L2

Contains:

```text id="m8q3vt"
Large Branches

Building Outlines

Object Boundaries
```

---

Medium frequencies.

---

## L3

Contains:

```text id="x4m9qw"
Large Shapes

Mountain Silhouettes

Road Boundaries
```

---

Low frequencies.

---

## G4

Contains:

```text id="u8v2tm"
Overall Brightness

Color

Lighting
```

---

Very low frequencies.

---

# Visual Intuition

Suppose the original image contains:

```text id="r2m7vx"
Mountain

Trees

Leaves

Sky
```

---

L0:

```text id="h9v4qm"
Leaf Details

Tiny Textures
```

---

L1:

```text id="n3m8tx"
Branches

Rock Details
```

---

L2:

```text id="j7v2mw"
Mountain Edges
```

---

L3:

```text id="p5m9qx"
Mountain Shape
```

---

G4:

```text id="z4v8tm"
Sky Brightness

Overall Illumination
```

---

Notice how the image has been separated into:

```text id="k1m7vx"
Different Frequency Bands
```

---

Exactly what we wanted.

---

# Why Is This Useful For Blending?

Remember the problem with feathering.

---

Feathering treats:

```text id="f8v3mq"
Lighting
```

and

```text id="c5m9tx"
Leaf Edges
```

exactly the same.

---

But human vision does not.

---

We want:

```text id="t4v8qm"
Lighting
```

to blend:

```text id="g2m7vw"
Very Slowly
```

across a wide region.

---

And:

```text id="n8v4tx"
Fine Details
```

to blend:

```text id="m5q2vw"
Very Quickly
```

across a narrow region.

---

The Laplacian Pyramid allows us to do exactly that.

---

Each frequency band can now receive:

```text id="w9m3qx"
Its Own Blending Width
```

---

This is the key idea behind:

```text id="p2v7tm"
Multi-Band Blending
```

---

# Can We Recover The Original Image?

A natural question is:

```text id="y4m8vx"
Did We Lose Information?
```

---

Surprisingly:

```text id="k7v2qw"
No
```

---

The Laplacian Pyramid is not merely a compressed representation.

It is a complete decomposition.

---

Given:

```text id="n1m5tx"
L0
L1
L2
L3
G4
```

we can reconstruct:

```text id="c8v4qm"
The Exact Original Image
```

---

This reconstruction process becomes extremely important after blending.

Because once we blend every frequency band independently:

```text id="x5m9vw"
We Must Combine Them Back Together
```

to create the final panorama.

---

This reconstruction step is the final stage of:

```text id="j2v7tx"
Multi-Band Blending
```

and is what allows professional panorama software to produce:

```text id="g8m3qw"
Smooth Lighting

Sharp Details

Nearly Invisible Seams
```

all at the same time.

---

## Summary

A Gaussian Pyramid stores:

```text id="v6m8tx"
Progressively Blurred Images
```

---

A Laplacian Pyramid stores:

```text id="t3v5qm"
The Details Lost
Between Gaussian Levels
```

---

Each Laplacian level corresponds to:

```text id="r7m2vx"
A Different Frequency Band
```

---

This frequency separation allows us to:

```text id="n4v8qw"
Blend Lighting Slowly

Blend Details Sharply
```

which is impossible using ordinary feathering.

---

Now that we have both:

```text id="k9m3tx"
Gaussian Pyramid

Laplacian Pyramid
```

we are finally ready to understand the complete:

```text id="p5v7qm"
Multi-Band Blending Algorithm
```

used in professional panorama stitching systems.

# 21. Multi-Band Blending: The Complete Algorithm

At this point we have learned:

```text
Feathering
```

removes seams by gradually crossfading images.

---

We also learned its biggest weakness:

```text
Ghosting
```

because feathering treats:

```text
Lighting
```

and

```text
Fine Details
```

exactly the same way.

---

To solve this problem we introduced:

```text
Gaussian Pyramids
```

and

```text
Laplacian Pyramids
```

which separate an image into different frequency bands.

---

Now we can finally understand:

```text
Multi-Band Blending
```

which is the blending method used in most professional panorama systems.

---

# The Main Idea

Instead of blending:

```text
One Image
```

we blend:

```text
Many Frequency Bands
```

independently.

---

Think of it like audio engineering.

A song contains:

```text
Bass

Vocals

Treble
```

---

Audio engineers do not process all frequencies identically.

Each frequency range is adjusted separately.

---

Multi-band blending does the same thing for images.

---

Instead of:

```text
One Image
```

we have:

```text
Fine Details

Medium Details

Large Structures

Lighting
```

and each is blended differently.

---

# Why This Works

Human vision is extremely sensitive to:

```text
Brightness Changes

Color Changes

Large Scale Illumination Differences
```

---

These belong to:

```text
Low Frequencies
```

---

Human vision is much more tolerant of:

```text
Small Texture Differences

Tiny Edge Differences
```

---

These belong to:

```text
High Frequencies
```

---

Therefore:

```text
Low Frequencies
```

should blend:

```text
Very Gradually
```

---

While:

```text
High Frequencies
```

should blend:

```text
Very Sharply
```

---

This is exactly what multi-band blending accomplishes.

---

# Step 1: Build Gaussian Pyramids

Suppose we have:

```text
Image A

Image B
```

---

Build Gaussian pyramids for both.

---

Image A:

```text
GA0
GA1
GA2
GA3
GA4
```

---

Image B:

```text
GB0
GB1
GB2
GB3
GB4
```

---

Each lower level contains:

```text
Less Detail

More Blur

Lower Frequencies
```

---

# Step 2: Build Laplacian Pyramids

Convert both Gaussian pyramids into Laplacian pyramids.

---

Image A:

```text
LA0
LA1
LA2
LA3
GA4
```

---

Image B:

```text
LB0
LB1
LB2
LB3
GB4
```

---

Each level now stores a specific frequency band.

---

Example:

```text
LA0
```

contains:

```text
Leaf Edges

Fine Textures

Tiny Details
```

---

While:

```text
GA4
```

contains:

```text
Lighting

Color

Large Structures
```

---

# Step 3: Create A Blending Mask

Remember feathering.

We created weights:

```text
1.0
0.9
0.8
0.7
...
0.0
```

---

These weights form a:

```text
Mask
```

---

Example:

```text
Image A Weight

1.0 → 0.0
```

---

Image B Weight

```text
0.0 → 1.0
```

---

This mask determines:

```text
Which Image Dominates
```

at each location.

---

# Step 4: Build A Gaussian Pyramid Of The Mask

This step is the secret sauce.

---

Suppose the original mask is:

```text
M0
```

---

Build a Gaussian pyramid:

```text
M0
M1
M2
M3
M4
```

---

Notice:

```text
High Resolution Mask
```

at upper levels.

---

And:

```text
Very Smooth Mask
```

at lower levels.

---

This automatically creates different blending widths.

---

# Why Do We Need A Mask Pyramid?

Suppose we used the same mask at every frequency level.

Then:

```text
Lighting

Leaves

Text

Sky
```

would still blend identically.

---

That defeats the entire purpose.

---

Instead:

```text
Each Frequency Band
```

gets its own version of the mask.

---

This allows:

```text
Wide Blending
```

for low frequencies

and

```text
Narrow Blending
```

for high frequencies.

---

# Step 5: Blend Each Frequency Band

Now blend corresponding Laplacian levels.

---

For Level 0:

```math
BL_0
=
M_0 LA_0
+
(1-M_0)LB_0
```

---

For Level 1:

```math
BL_1
=
M_1 LA_1
+
(1-M_1)LB_1
```

---

For Level 2:

```math
BL_2
=
M_2 LA_2
+
(1-M_2)LB_2
```

---

Continue for every level.

---

Result:

```text
BL0
BL1
BL2
BL3
BG4
```

---

These are the:

```text
Blended Frequency Bands
```

---

# What Is Actually Happening?

Consider:

```text
Level 0
```

which contains:

```text
Tiny Details

Leaf Edges

Fine Textures
```

---

Mask:

```text
M0
```

is sharp.

---

Transition region:

```text
Very Narrow
```

---

Therefore:

```text
Edges Stay Sharp
```

---

Now consider:

```text
Level 4
```

which contains:

```text
Lighting

Brightness

Large Structures
```

---

Mask:

```text
M4
```

is heavily blurred.

---

Transition region:

```text
Very Wide
```

---

Therefore:

```text
Brightness Changes
```

become extremely smooth.

---

# Visual Intuition

Suppose seam location:

```text
AAAAA|BBBBB
```

---

Feathering:

```text
AAAABBBBB
```

Everything blends equally.

---

Multi-band:

Low Frequencies:

```text
AAAAaaaabbbbBBBB
```

Very smooth transition.

---

High Frequencies:

```text
AAAA|BBBB
```

Very sharp transition.

---

Combined result:

```text
Smooth Lighting

Sharp Details
```

simultaneously.

---

# Step 6: Reconstruct The Image

After blending all pyramid levels:

```text
BL0
BL1
BL2
BL3
BG4
```

we must rebuild the image.

---

Start at the smallest level.

---

Expand:

```text
BG4
```

---

Add:

```text
BL3
```

---

Expand again.

---

Add:

```text
BL2
```

---

Continue:

```text
Expand
+
Add
```

until reaching full resolution.

---

Mathematically:

```math
G_3
=
BL_3
+
Expand(BG_4)
```

---

Then:

```math
G_2
=
BL_2
+
Expand(G_3)
```

---

Then:

```math
G_1
=
BL_1
+
Expand(G_2)
```

---

Then:

```math
G_0
=
BL_0
+
Expand(G_1)
```

---

The final:

```text
G0
```

is the reconstructed panorama.

---

# Why It Looks Better Than Feathering

Suppose:

```text
Image A Brightness = 180

Image B Brightness = 150
```

---

Low-frequency blending smooths this difference over hundreds of pixels.

---

The seam disappears.

---

Suppose:

```text
Tree Edge
```

is slightly misaligned.

---

High-frequency blending uses a very narrow transition.

---

Instead of averaging:

```text
Tree A

+

Tree B
```

over a wide region,

it largely preserves the strongest local detail.

---

Result:

```text
Less Ghosting

Sharper Edges
```

---

# Computational Cost

Feathering:

```text
One Weight Map

One Weighted Average
```

---

Complexity:

```text
Very Cheap
```

---

Multi-band blending:

```text
Build Gaussian Pyramid

Build Laplacian Pyramid

Build Mask Pyramid

Blend Every Level

Reconstruct Pyramid
```

---

Complexity:

```text
Much Higher
```

---

Memory usage:

```text
Much Higher
```

---

Implementation difficulty:

```text
Much Higher
```

---

# Why OpenCV Uses It

When OpenCV creates professional panoramas, one of the most important components is:

```text
MultiBandBlender
```

---

Internally it performs:

```text
Frequency Separation

Pyramid Construction

Band-wise Blending

Pyramid Reconstruction
```

---

This is a major reason OpenCV panoramas often appear dramatically better than panoramas produced using simple feathering.

---

# Mental Model

Think of:

```text
Feathering
```

as:

```text
Crossfading Two Images
```

like fading between two songs.

---

Think of:

```text
Multi-Band Blending
```

as:

```text
Crossfading Bass Separately

Crossfading Vocals Separately

Crossfading Treble Separately
```

and then recombining everything.

---

Because each frequency band is blended differently, the final panorama achieves:

```text
Smooth Lighting

Sharp Details

Minimal Ghosting

Nearly Invisible Seams
```

which is why multi-band blending remains the standard blending technique in modern panorama stitching systems.

# 22. Why Multi-Band Blending Is Not Enough

After learning Multi-Band Blending, many students assume:

```text
Problem Solved
```

---

After all, Multi-Band Blending provides:

```text
Smooth Lighting

Sharp Details

Minimal Ghosting

Invisible Seams
```

---

This is true for many panoramas.

However, if you compare your implementation against:

```text
OpenCV Stitcher

AutoStitch

PTGui

Hugin
```

you will notice something surprising.

---

Even with:

```text
Perfect SIFT

Perfect RANSAC

Perfect Homography

Perfect Multi-Band Blending
```

professional panorama software often still looks better.

---

Why?

Because blending is only one piece of the puzzle.

---

# The Hidden Problems

Real photographs contain many imperfections.

Examples:

```text
Different Exposures

Different White Balance

Lens Distortion

Parallax

Moving Objects

Perspective Changes
```

---

Multi-band blending cannot solve these.

---

It only answers:

```text
How Should We Blend Colors?
```

---

It does not answer:

```text
Which Geometry Is Correct?

Which Exposure Is Correct?

Where Should The Seam Pass?
```

---

Professional panorama systems solve all of these separately.

---

# Problem 1: Exposure Differences

Suppose you capture:

```text
Image A
```

under sunlight.

---

A few seconds later:

```text
Image B
```

is captured after a cloud appears.

---

Now:

```text
Image A Average Brightness

180
```

---

```text
Image B Average Brightness

120
```

---

Even if alignment is perfect:

```text
Seam Appears
```

because the entire images have different brightness.

---

Multi-band blending helps.

But it does not completely solve the issue.

---

The correct solution is:

```text
Exposure Compensation
```

---

# Exposure Compensation

Before blending:

```text
Estimate Brightness Difference
```

between overlapping regions.

---

Example:

```text
Image A Average

180
```

---

```text
Image B Average

150
```

---

Compute correction:

```text
Scale B Up

or

Scale A Down
```

---

Now both images share a similar exposure.

---

Then blending becomes much easier.

---

This is exactly what OpenCV does internally.

---

# Problem 2: White Balance Differences

Suppose:

```text
Image A
```

appears slightly warm.

---

```text
Image B
```

appears slightly blue.

---

Even after perfect blending:

```text
Color Shift
```

remains visible.

---

Example:

```text
Sky
```

changes from:

```text
Blue
```

to

```text
Cyan
```

across the panorama.

---

Humans notice this immediately.

---

Professional panorama software performs:

```text
Color Compensation
```

in addition to brightness compensation.

---

# Problem 3: Lens Vignetting

Most camera lenses darken near their boundaries.

This effect is called:

```text
Vignetting
```

---

Visualization:

```text
Center

Bright
```

---

```text
Edges

Darker
```

---

When two images overlap:

```text
Edge Of Image A

meets

Edge Of Image B
```

---

Both regions may be darker than expected.

---

This creates visible intensity variations.

---

Professional panorama systems estimate and compensate for:

```text
Lens Falloff
```

before blending.

---

# Problem 4: Bad Seam Locations

Suppose overlap contains:

```text
Person

Car

Tree Branch
```

---

Feathering blends through them.

---

Multi-band blending also blends through them.

---

Result:

```text
Ghosting
```

---

But what if we could simply avoid those objects?

---

This idea leads to:

```text
Seam Finding
```

---

# What Is Seam Finding?

Instead of blending everywhere equally:

```text
Choose A Path
```

through the overlap.

---

The path should pass through:

```text
Sky

Grass

Road
```

---

and avoid:

```text
Faces

Cars

Buildings

Text
```

---

because humans notice errors there.

---

Visualization:

```text
Overlap Region

XXXXXXXXXX
XXXXXXXXXX
XXXXXXXXXX
```

---

Bad seam:

```text
Cuts Through Person
```

---

Good seam:

```text
Passes Through Empty Sky
```

---

This dramatically reduces visible artifacts.

---

# Graph-Cut Seam Finding

Modern panorama systems often use:

```text
Graph Cut Optimization
```

---

Each pixel becomes a node.

---

The algorithm asks:

```text
Should This Pixel Come
From Image A

or

Image B ?
```

---

The seam is chosen to minimize:

```text
Visible Error
```

---

Result:

```text
Less Ghosting

Cleaner Boundaries

Better Panoramas
```

---

# Problem 5: Parallax

Parallax is one of the hardest problems in panorama stitching.

---

Suppose:

```text
Tree

10 meters away
```

---

And:

```text
Mountain

5 kilometers away
```

---

You move the camera sideways.

---

The nearby tree appears to move more than the mountain.

---

This violates a fundamental assumption of homography.

---

# Why Homography Fails

Homography assumes:

```text
One Global Transformation
```

for the entire image.

---

Reality:

```text
Near Objects

Move Differently
```

than

```text
Far Objects
```

---

No single matrix can perfectly align both.

---

Result:

```text
Ghosting

Double Edges

Distortions
```

---

even if RANSAC is perfect.

---

# Bundle Adjustment

Professional systems often perform:

```text
Bundle Adjustment
```

after feature matching.

---

Instead of optimizing:

```text
One Homography
```

---

They optimize:

```text
All Camera Poses
```

simultaneously.

---

This significantly improves global alignment.

---

Especially for:

```text
Large Panoramas
```

containing many images.

---

# Why OpenCV Looks Better

When people compare:

```text
My Stitcher
```

versus:

```text
OpenCV Stitcher
```

they often think:

```text
OpenCV Has Better SIFT
```

---

Usually that is not true.

---

The biggest improvements come from:

```text
Exposure Compensation

Seam Finding

Bundle Adjustment

Multi-Band Blending
```

working together.

---

# OpenCV Panorama Pipeline

A simplified version looks like:

```text
Feature Detection
        ↓
Descriptor Extraction
        ↓
Feature Matching
        ↓
RANSAC
        ↓
Camera Estimation
        ↓
Bundle Adjustment
        ↓
Warping
        ↓
Exposure Compensation
        ↓
Seam Finding
        ↓
Multi-Band Blending
        ↓
Final Panorama
```

---

# Mental Model

Think of panorama stitching as three separate problems.

---

## Geometry Problem

```text
Where Should Pixels Go?
```

Solved by:

```text
Homography

Bundle Adjustment

Warping
```

---

## Photometric Problem

```text
How Bright Should Pixels Be?
```

Solved by:

```text
Exposure Compensation

White Balance Compensation
```

---

## Blending Problem

```text
How Should Overlapping Pixels Be Combined?
```

Solved by:

```text
Feathering

Multi-Band Blending

Graph-Cut Seam Finding
```

---

A professional panorama only appears seamless when:

```text
Geometry

Photometry

And Blending
```

are all solved together.

---

## Summary

Multi-band blending is extremely powerful, but it is not the final stage of panorama stitching.

Professional systems additionally use:

```text
Exposure Compensation

White Balance Correction

Vignetting Compensation

Graph-Cut Seam Finding

Bundle Adjustment
```

to eliminate the remaining artifacts.

Only after all these stages work together do we obtain the type of panorama quality seen in:

```text
OpenCV Stitcher

PTGui

AutoStitch

Hugin
```

and other professional stitching systems.
