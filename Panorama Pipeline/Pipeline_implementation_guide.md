# Run the Panorama Pipeline

This guide runs the complete pipeline in one command: CUDA SIFT feature matching, homography, blending, and panorama creation.

## 1. Copy the project folder

Copy the full `CUDA` project folder to a computer that has an NVIDIA GPU and CUDA Toolkit installed. Keep these files together:

- `run_pipeline.py`
- `cpu_pipeline.py`
- `Sift_till_descriptor_matching_FIXED.ipynb`

## 2. Install the one-time requirements

Open PowerShell inside the copied project folder and run:

```powershell
python -m pip install numpy opencv-python
nvcc --version
```

The second command must show a CUDA version. If it says that `nvcc` is not found, CUDA Toolkit is not installed or not added to PATH.

On Windows, CUDA compilation also needs the Visual Studio C++ Build Tools installed.

## 3. Add the two images

Create an `input` folder inside the project folder. Put exactly two overlapping images in it and name them exactly:

```text
input/
  camera0.jpg
  camera1.jpg
```

`jpg`, `jpeg`, `png`, `bmp`, `tif`, and `tiff` are supported. The two images must show overlapping parts of the same scene.

## 4. Run everything

In PowerShell, from the project folder, run this single command:

```powershell
python run_pipeline.py
```

That is all. The script automatically prepares CUDA inputs, extracts and compiles the CUDA code, runs descriptor matching, estimates the homography, blends the images, and saves the panorama.

## 5. Get the result

When the terminal says `PIPELINE COMPLETE`, open:

```text
output/panorama.png
```

The first run can take longer because CUDA code is compiled. Later runs reuse the compiled program automatically; replace only the two images in `input` and run the same command again.

## If an error appears

- `nvcc is not available`: install CUDA Toolkit and reopen PowerShell.
- `Expected exactly one input image`: check that the files are called `camera0` and `camera1`, with one supported image extension each.
- CUDA build error on Windows: install Visual Studio C++ Build Tools, then rerun the command.
