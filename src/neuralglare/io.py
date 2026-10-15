"""Shared HDR image I/O and small helpers. Linear float32 RGB everywhere, never clamped."""

import cv2  # Radiance .hdr reading, PNG previews (pip wheels often lack EXR support)
import numpy as np
import OpenEXR  # official bindings: pip install OpenEXR


def write_exr(path, rgb):
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    OpenEXR.File(header, {"RGB": np.ascontiguousarray(rgb, dtype=np.float32)}).write(path)


def read_exr(path):
    with OpenEXR.File(path) as f:
        ch = f.channels()
        if "RGB" in ch:
            return ch["RGB"].pixels.astype(np.float32)
        return np.stack([ch[c].pixels for c in ("R", "G", "B")], -1).astype(np.float32)


def read_hdr(path):
    """Read .exr or Radiance .hdr as linear float32 RGB (H, W, 3)."""
    if path.lower().endswith(".exr"):
        return read_exr(path)
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise IOError(f"failed to read {path}")
    return img[..., ::-1].astype(np.float32)


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def write_preview(path, rgb, exposure=1.0):
    """Reinhard + gamma 2.2, for eyeballing only. Never used as data."""
    x = rgb * exposure
    x = np.clip(x / (1.0 + x), 0, 1) ** (1 / 2.2)
    cv2.imwrite(path, (x[..., ::-1] * 255).astype(np.uint8))