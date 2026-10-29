"""Pull the Unreal check's data out of RenderDoc captures of the convolution-bloom frame.

Runs inside RenderDoc's Python (renderdoc module, Python 3.8), not the uv environment. From
the repo root:

    "C:/Program Files/RenderDoc/qrenderdoc.exe" --python engines/unreal/Baseline/extract_capture.py

For each data/captures/unreal/<scene>_convolution.rdc it writes, next to the capture:
    unreal_bloom_<scene>.exr          Bloom.FFT.ScatterDispersion (the FFT bloom texture)
    unreal_fft_constants_<scene>.json  Bloom.FFT.SceneColorApplyParameters and
                                       Bloom.FFT.FFTMulitplyParameters (Epic's spelling), and
                                       the spectral kernel texels the FFT normalises by
The tonemapper combines them as SceneColor · SceneColorApply + bloom, so no single linear
buffer holds the final image; unreal_check.py rebuilds it.
"""

import glob
import json
import os
import struct
import traceback

import renderdoc as rd

FOLDER = os.path.join(os.getcwd(), "data", "captures", "unreal")
BUFFERS = ("Bloom.FFT.SceneColorApplyParameters", "Bloom.FFT.FFTMulitplyParameters")
BLOOM = "Bloom.FFT.ScatterDispersion"
SPECTRAL = "Bloom.FFT.SpectralKernel"


def by_name(ctrl):
    names = {}
    for r in ctrl.GetResources():
        names.setdefault(r.name, []).append(r.resourceId)
    return names


def save_exr(ctrl, rid, path):
    ts = rd.TextureSave()
    ts.resourceId = rid
    ts.destType = rd.FileType.EXR
    ts.mip = 0
    ts.slice.sliceIndex = 0
    ts.alpha = rd.AlphaMapping.Preserve
    if ctrl.SaveTexture(ts, path) != rd.ResultCode.Succeeded:
        raise RuntimeError(f"could not save {path}")


def extract(path):
    scene = os.path.basename(path).replace("_convolution.rdc", "")
    cap = rd.OpenCaptureFile()
    if cap.OpenFile(path, "", None) != rd.ResultCode.Succeeded:
        raise RuntimeError(f"could not open {path}")
    _, ctrl = cap.OpenCapture(rd.ReplayOptions(), None)
    names = by_name(ctrl)
    texs = {t.resourceId: t for t in ctrl.GetTextures()}
    bufs = {b.resourceId: b for b in ctrl.GetBuffers()}
    ctrl.SetFrameEvent(10 ** 9, True)              # end of frame: buffers hold final values
    if BLOOM not in names:
        raise RuntimeError(f"{path}: no {BLOOM}; is Bloom Method's override ticked?")

    out = {"capture": os.path.relpath(path).replace("\\", "/")}
    for n in BUFFERS:
        data = ctrl.GetBufferData(names[n][0], 0, bufs[names[n][0]].length)
        out[n] = list(struct.unpack("<4f", data[:16]))
    # FFT normalisation reads these texels (GetKernelSum, GPUFastFourierTransform.usf:146).
    # Layout after the horizontal two-for-one: columns [0, N/2) hold (R, B), the rest (G, A)
    k = texs[names[SPECTRAL][0]]
    out[SPECTRAL] = {"size": [k.width, k.height], "texels": {}}
    for x in (0, k.width - 2):
        v = ctrl.PickPixel(k.resourceId, x, 0, rd.Subresource(0, 0, 0), rd.CompType.Typeless)
        out[SPECTRAL]["texels"][f"{x},0"] = list(v.floatValue)
    t = texs[names[BLOOM][0]]
    out[BLOOM] = {"size": [t.width, t.height], "format": t.format.Name()}
    save_exr(ctrl, t.resourceId, os.path.join(FOLDER, f"unreal_bloom_{scene}.exr"))
    with open(os.path.join(FOLDER, f"unreal_fft_constants_{scene}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    ctrl.Shutdown()
    cap.Shutdown()


def main():
    # qrenderdoc shows no console, so every outcome goes to the log
    with open(os.path.join(FOLDER, "extract_capture.log"), "w") as log:
        try:
            for path in sorted(glob.glob(os.path.join(FOLDER, "*_convolution.rdc"))):
                extract(path)
                log.write(f"ok {path}\n")
        except Exception:  # noqa: BLE001
            log.write(traceback.format_exc())
    os._exit(0)                                    # quit qrenderdoc without opening its UI


main()
