"""Measured camera spectral sensitivities (Jiang et al. 2013).

Loads measured camera spectral sensitivity curve
    - How strongly a real camera's R, G, and B pixels respond to light at each wavelength
    - Reference renderer uses them to turn a per-wavelength PSF into an RGB PSF

- The reference PSF is computed spectrally
- Diffraction and chromatic abberation depend on wavelength, so the lens is simulated at 48 wavelengths from 420 to 680nm
- This gives 48 grayscale PSFs
- But the renderer works in RGB, so those 48 slices have to be collapsed into 3 channels

Source: J. Jiang, D. Liu, J. Gu, S. Süsstrunk, "What is the Space of Spectral Sensitivity
Functions for Digital Color Cameras?", WACV 2013. Database of 28 cameras, 400-720 nm in
10 nm steps, from Zenodo record 3245883 (doi:10.5281/zenodo.3245883).
License CC BY-NC-SA 4.0, so the file is downloaded into the gitignored data/ folder
instead of being committed, and checked against Zenodo's MD5.
"""

import hashlib
import os
import urllib.request

import numpy as np

URL = "https://zenodo.org/api/records/3245883/files/camspec_database.txt/content"
MD5 = "dcf7ce9ec9f2c87b87014f5cc46f8c15"
PATH = "data/camspec/camspec_database.txt"
WAVELENGTHS_NM = np.arange(400, 721, 10)


def _fetch(path=PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading camera spectral database -> {path}")
        urllib.request.urlretrieve(URL, path)
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read()).hexdigest()
    if digest != MD5:
        raise RuntimeError(f"{path}: MD5 {digest} != {MD5}; delete it and retry")
    return path


def sensitivities(camera, path=PATH):
    """(33, 3) R, G, B relative sensitivities of `camera` (e.g. "Nikon D700") at
    WAVELENGTHS_NM."""
    with open(_fetch(path)) as f:
        lines = [ln.strip() for ln in f]
    i = lines.index(camera)
    return np.array([[float(v) for v in lines[i + k].split()] for k in (1, 2, 3)]).T
