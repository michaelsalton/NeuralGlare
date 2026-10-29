"""Unreal 5.8 convolution bloom as it actually runs: what the engine does to the input and to
the kernel before convolving. Engine paths are relative to Engine/ in UE 5.8.4 (CL 58902052).

    unreal_fft_bloom(I, K) = Q(I) ⊛ K_eff

Q is the R11G11B10 quantisation of post-process scene colour, and K_eff the kernel after
Unreal's centre search, clamp, pass-through split, renormalisation and tint. The pass-through
weight C sits on K_eff's centre texel, so convolving with K_eff gives the whole output,
Tint · (C·I + I ⊛ K_clamped) / (C + ΣK_clamped), in one convolution.

Emulation assumes 1 kernel texel = 1 view pixel (Convolution Scale = kernel width / max(view
width, height), BloomResizeKernel.usf). Unconfirmed, to be settled by the Unreal capture:
- "Scatter dispersion energy" is the sum of the clamped kernel (BloomSumScatterDispersionEnergy.usf)
- the FFT normalises the kernel per channel (GPUFastFourierTransform.usf)
"""

import numpy as np

from neuralglare.reference.convolve import convolve_uniform

# Post-process scene colour is PF_FloatR11G11B10 unless r.PostProcessing.PropagateAlpha is on
# (Source/Runtime/Renderer/Private/PostProcess/PostProcessing.cpp:459): unsigned, 5-bit
# exponent (bias 15), mantissa bits per channel:
R11G11B10_MANTISSA = (6, 6, 5)
KERNEL_LUMA = np.array([0.3, 0.59, 0.11])   # luminance weights of the centre search


def quantize_r11g11b10(img):
    """Round linear RGB to what an R11G11B10_FLOAT render target stores: negatives become 0,
    the mantissa is truncated (rounded toward zero), and values clamp to the largest finite
    one. Truncation matches data/captures/unreal/unreal_beforebloom_ferndale.exr bit for bit
    (round-to-nearest misses half of its values by one step)."""
    out = np.empty(img.shape, np.float32)
    for c, m in enumerate(R11G11B10_MANTISSA):
        x = np.maximum(img[..., c].astype(np.float64), 0.0)
        _, e = np.frexp(x)                    # x = f · 2**e, f in [0.5, 1)
        e = np.maximum(e, -13)                # below 2**-14 the step is fixed (denormals)
        q = np.ldexp(np.floor(np.ldexp(x, m + 1 - e)), e - m - 1)
        out[..., c] = np.minimum(q, (2.0 - 2.0 ** -m) * 2.0 ** 15)
    return out


def unreal_kernel(K):
    """K_eff (k, k, 3) and the intermediate constants, for a kernel K (k, k, 3) that sums to 1."""
    K = K.astype(np.float64)
    # centre = brightest texel by luminance; the Center setting is ignored in 5.8
    # (Shaders/Private/Bloom/BloomFindKernelCenter.usf)
    cy, cx = np.unravel_index((K @ KERNEL_LUMA).argmax(), K.shape[:2])
    yy, xx = np.mgrid[:K.shape[0], :K.shape[1]]
    d = np.maximum(abs(xx - cx), abs(yy - cy))           # Chebyshev distance, view pixels
    # clamp level: max over RGB in a ring 2-4 px from the centre
    # (BloomSurveyMaxScatterDispersion.usf)
    M = K[(d >= 2) & (d <= 4)].max()
    Kc = np.minimum(K, M)                                # BloomClampKernel.usf
    # only the centre texel's energy above M survives, as a pass-through weight on scene
    # colour (BloomSurveyKernelCenterEnergy.usf); the neighbours' excess is lost
    C = np.maximum(K[cy, cx] - M, 0.0)
    # renormalise per channel, then tint = total / max(total) (BloomFinalizeApplyConstants.usf)
    T = C + Kc.sum((0, 1))
    tint = np.clip(T / T.max(), 0.0, 1.0)
    K_eff = (Kc + np.where((d == 0)[..., None], C, 0.0)) / T * tint
    info = {"centre": (int(cy), int(cx)), "clamp_level": float(M), "pass_through": C.tolist(),
            "pass_through_fraction": (C / T).tolist(), "tint": tint.tolist()}
    return K_eff.astype(np.float32), info


def unreal_fft_bloom(image, kernel):
    """Predicted Unreal output (bloom combined with scene colour) for a linear HDR frame."""
    K_eff, _ = unreal_kernel(kernel)
    return convolve_uniform(quantize_r11g11b10(image), K_eff)
