"""Unreal 5.8 convolution bloom as it actually runs: what the engine does to the input and to
the kernel before convolving. Engine paths are relative to Engine/ in UE 5.8.4 (CL 58902052).

    unreal_fft_bloom(I, K)        = Q(I) ⊛ K_eff                       prediction (b), as designed
    unreal_fft_bloom_as_run(I, K) = Q(I) · apply + Q(bloom)            prediction (c), as it runs

Q is the R11G11B10 quantisation of post-process scene colour, and K_eff the kernel after
Unreal's centre search, clamp, pass-through split, renormalisation and tint. The pass-through
weight C sits on K_eff's centre texel, so convolving with K_eff gives the whole output,
Tint · (C·I + I ⊛ K_clamped) / (C + ΣK_clamped), in one convolution.

(c) adds two channel mix-ups in the 5.8 FFT (GPUFastFourierTransform.usf), found in the
2026-10-29 captures (data/captures/unreal/), and the R11G11B10 bloom texture. It is post hoc:
it was written after seeing Unreal's output, so it explains the result but does not test it.

Assumes 1 kernel texel = 1 view pixel (Convolution Scale = kernel width / max(view width,
height), BloomResizeKernel.usf). Confirmed by the captures: the apply constants match to 6-7
digits, so "scatter dispersion energy" is the sum of the clamped kernel; the FFT normalises
per channel, but by the wrong channels' sums (FFT_NORM_CHANNEL).
"""

import numpy as np

from neuralglare.reference.convolve import convolve_uniform

# Post-process scene colour is PF_FloatR11G11B10 unless r.PostProcessing.PropagateAlpha is on
# (Source/Runtime/Renderer/Private/PostProcess/PostProcessing.cpp:459): unsigned, 5-bit
# exponent (bias 15), mantissa bits per channel. The FFT bloom texture
# (Bloom.FFT.ScatterDispersion) uses the same format.
R11G11B10_MANTISSA = (6, 6, 5)
KERNEL_LUMA = np.array([0.3, 0.59, 0.11])   # luminance weights of the centre search

# Channel c of the bloom is normalised by the kernel sum of channel FFT_NORM_CHANNEL[c].
# After the two-for-one transform, spectral texels hold (R, B) in the first half of the
# columns and (G, A) in the second (GPUFastFourierTransform.usf:569-578), so GetKernelSum
# (line 146) returns (R, B) and (G, A), but GSConvolutionWithTextureCS (line 640) uses
# them as (R, G) and (B, A): green is divided by blue's sum and blue by green's
FFT_NORM_CHANNEL = (0, 2, 1)
# Channel c of the bloom is scaled by FFTMulitplyParameters[FFT_MULTIPLY_CHANNEL[c]]: the
# inverse two-for-one scales b + i·a by Scale.xw, not Scale.zw (line 415), so blue gets red's
FFT_MULTIPLY_CHANNEL = (0, 1, 0)


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
    """K_eff (k, k, 3) and the intermediate constants, for a kernel K (k, k, 3) that sums to 1.
    info["scene_color_apply"] and info["fft_multiply"] are what Unreal writes to
    Bloom.FFT.SceneColorApplyParameters and Bloom.FFT.FFTMulitplyParameters."""
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
            "pass_through_fraction": (C / T).tolist(), "tint": tint.tolist(),
            "clamped_kernel_sum": Kc.sum((0, 1)).tolist(),
            "scene_color_apply": (tint * C / T).tolist(),
            "fft_multiply": (tint * (1.0 - C / T)).tolist()}
    return K_eff.astype(np.float32), info


def unreal_fft_bloom(image, kernel):
    """Prediction (b): Unreal's output (bloom combined with scene colour) as the 5.8 shaders
    are designed, for a linear HDR frame."""
    K_eff, _ = unreal_kernel(kernel)
    return convolve_uniform(quantize_r11g11b10(image), K_eff)


def unreal_bloom_as_run(scene_color, kernel):
    """The FFT bloom texture (Bloom.FFT.ScatterDispersion) as 5.8 computes it, from the
    already-quantised scene colour: the clamped kernel stored as fp16 (Bloom.FFT.ClampedKernel
    is PF_FloatRGBA, PostProcessFFTBloom.cpp:577; this zeroes the far tails), normalised by
    its own sums and scaled per channel through FFT_NORM_CHANNEL and FFT_MULTIPLY_CHANNEL,
    stored as R11G11B10."""
    K = kernel.astype(np.float64)
    _, info = unreal_kernel(K)
    Kc = np.minimum(K, info["clamp_level"]).astype(np.float16).astype(np.float64)
    s = Kc.sum((0, 1))
    mult = np.array(info["fft_multiply"])
    gain = mult[list(FFT_MULTIPLY_CHANNEL)] / s[list(FFT_NORM_CHANNEL)]
    return quantize_r11g11b10(convolve_uniform(scene_color, (Kc * gain).astype(np.float32)))


def unreal_fft_bloom_as_run(image, kernel):
    """Prediction (c), post hoc: the frame the tonemapper combines,
    SceneColor · SceneColorApply + bloom (PostProcessFFTBloom.cpp, Tonemap pass)."""
    _, info = unreal_kernel(kernel)
    scene = quantize_r11g11b10(image)
    return scene * np.float32(info["scene_color_apply"]) + unreal_bloom_as_run(scene, kernel)
