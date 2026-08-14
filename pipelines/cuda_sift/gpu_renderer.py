"""ctypes bridge for the persistent fused CUDA panorama renderer."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


class CudaRendererError(RuntimeError):
    pass


_BYTE_PTR = ctypes.POINTER(ctypes.c_ubyte)
_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)


class CudaPanoramaRenderer:
    """Persistent GPU warp/blend renderer with cached geometry and weights."""

    def __init__(self, library_path: str | Path, camera0_shape, camera1_shape,
                 homography, *, blend_mode="fast", feather_radius=96.0):
        path = Path(library_path).expanduser().resolve()
        if not path.is_file():
            raise CudaRendererError(f"CUDA renderer library does not exist: {path}")
        if blend_mode not in {"fast", "feather"}:
            raise ValueError("blend_mode must be 'fast' or 'feather'")
        self.camera0_shape = tuple(camera0_shape[:2])
        self.camera1_shape = tuple(camera1_shape[:2])
        try:
            self._dll = ctypes.CDLL(str(path))
        except OSError as error:
            raise CudaRendererError(f"Could not load CUDA renderer {path}: {error}") from error
        try:
            self._create = self._dll.panocall_renderer_create
            self._render = self._dll.panocall_renderer_render
            self._destroy = self._dll.panocall_renderer_destroy
        except AttributeError as error:
            raise CudaRendererError(f"Renderer API exports are missing from {path}") from error

        self._create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            _DOUBLE_PTR, ctypes.c_int, ctypes.c_float,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_char), ctypes.c_int,
        ]
        self._create.restype = ctypes.c_int
        self._render.argtypes = [
            ctypes.c_void_p, _BYTE_PTR, ctypes.c_size_t, _BYTE_PTR, ctypes.c_size_t,
            _BYTE_PTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_char), ctypes.c_int,
        ]
        self._render.restype = ctypes.c_int
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = ctypes.c_int

        h0, w0 = self.camera0_shape
        h1, w1 = self.camera1_shape
        matrix = np.ascontiguousarray(homography, dtype=np.float64).reshape(3, 3)
        self._handle = ctypes.c_void_p()
        output_width = ctypes.c_int()
        output_height = ctypes.c_int()
        error = ctypes.create_string_buffer(1024)
        status = self._create(
            w0, h0, w1, h1, matrix.ctypes.data_as(_DOUBLE_PTR),
            1 if blend_mode == "feather" else 0, ctypes.c_float(feather_radius),
            ctypes.byref(self._handle), ctypes.byref(output_width), ctypes.byref(output_height),
            error, len(error))
        if status != 0 or not self._handle.value:
            detail = error.value.decode("utf-8", errors="replace") or f"status {status}"
            raise CudaRendererError(f"Could not create CUDA renderer: {detail}")
        self.width = output_width.value
        self.height = output_height.value
        self._output = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.last_elapsed_ms = 0.0

    def render(self, frame0, frame1):
        if tuple(frame0.shape[:2]) != self.camera0_shape or tuple(frame1.shape[:2]) != self.camera1_shape:
            raise ValueError("Frame dimensions changed after CUDA renderer initialization")
        source0 = np.ascontiguousarray(frame0, dtype=np.uint8)
        source1 = np.ascontiguousarray(frame1, dtype=np.uint8)
        elapsed = ctypes.c_float()
        error = ctypes.create_string_buffer(1024)
        status = self._render(
            self._handle,
            source0.ctypes.data_as(_BYTE_PTR), source0.nbytes,
            source1.ctypes.data_as(_BYTE_PTR), source1.nbytes,
            self._output.ctypes.data_as(_BYTE_PTR), self._output.nbytes,
            ctypes.byref(elapsed), error, len(error))
        if status != 0:
            detail = error.value.decode("utf-8", errors="replace") or f"status {status}"
            raise CudaRendererError(f"CUDA rendering failed: {detail}")
        self.last_elapsed_ms = float(elapsed.value)
        return self._output

    def close(self):
        if getattr(self, "_handle", None) is not None and self._handle.value:
            self._destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
