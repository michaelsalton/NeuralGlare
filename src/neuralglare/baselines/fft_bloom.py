"""Uniform-kernel FFT bloom: Unreal-style convolution bloom with one kernel everywhere.

    fft_bloom(I) = (1 - η) · I ⊛ K  +  η · S ⊛ I

K is a single diffraction kernel (the centre PSF, or the best-fit PSF). The scatter term
η · S ⊛ I is identical to the reference's, so the only difference from the reference is
that K does not vary across the frame. Light spread off the frame is lost, as in the
reference.
"""
import argparse
import glob
import json
import os

import numpy as np
import scipy
import torch

from neuralglare.eval.metrics import log_loss
from neuralglare.io import read_hdr
from neuralglare.reference.convolve import (
    PSFSet,
    convolve_scatter,
    convolve_uniform,
    interpolated_psf,
)


def centre_psf(ps):
    """The diffraction PSF the reference applies at the frame-centre pixel, (K, K, 3)."""
    return interpolated_psf(ps, ps.width // 2, ps.height // 2).astype(np.float32)


def scatter_term(image, ps):
    """S ⊛ I (without η). It does not depend on K, so compute it once per image and reuse it."""
    return convolve_scatter(image, ps.scatter)


def fft_bloom(image, kernel, ps, scatter=None):
    """(1 - η) · I ⊛ kernel + η · S ⊛ I. Pass a precomputed scatter_term to skip recomputing it."""
    eta = ps.scatter.energy_fraction
    if scatter is None:
        scatter = scatter_term(image, ps)
    return ((1.0 - eta) * convolve_uniform(image, kernel) + eta * scatter).astype(np.float32)


def conv_same_torch(image, kernel):
    """FFT convolution matching convolve_uniform (scipy fftconvolve, mode="same").
    image (H, W, C), kernel (kh, kw, C): torch tensors on one device. Use float64,
    for the same round-off reason as convolve_uniform."""
    H, W = image.shape[:2]
    kh, kw = kernel.shape[:2]
    fh, fw = H + kh - 1, W + kw - 1               # full linear size: zero padding, no wrap-around
    img = image.permute(2, 0, 1)                  # (C, H, W): rfft2 works on the last two dims
    ker = kernel.permute(2, 0, 1)
    spectrum = torch.fft.rfft2(img, s=(fh, fw)) * torch.fft.rfft2(ker, s=(fh, fw))
    full = torch.fft.irfft2(spectrum, s=(fh, fw))
    y0, x0 = (kh - 1) // 2, (kw - 1) // 2         # the same centred crop scipy's "same" uses
    return full[:, y0:y0 + H, x0:x0 + W].permute(1, 2, 0)


def fit_uniform_kernel(pairs, scatters, eta, init, eps=1e-3, steps=200, lr=0.1, device="cuda"):
    """Best single diffraction kernel for all (image, reference) pairs.

    Softmax over texels per channel (so the kernel is >= 0 and sums to 1, like a PSF),
    initialised from `init`, fitted with Adam to the mean log loss. Runs in float64.
    Returns (kernel (kh, kw, C) float32, loss per step).
    """
    torch.manual_seed(0)        # nothing is random today; keeps any future change reproducible
    data = [(torch.from_numpy(img).to(device, torch.float64),
             torch.log(torch.from_numpy(ref).to(device, torch.float64) + eps),
             eta * torch.from_numpy(s).to(device, torch.float64))
            for (img, ref), s in zip(pairs, scatters)]
    kh, kw, c = init.shape
    theta = torch.log(torch.from_numpy(np.maximum(init, 1e-30).astype(np.float64)))
    theta = theta.reshape(kh * kw, c).to(device).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    history = []
    for step in range(steps):
        opt.zero_grad()
        kernel = torch.softmax(theta, dim=0).reshape(kh, kw, c)
        total = 0.0
        for img, log_ref, scat in data:
            pred = (1.0 - eta) * conv_same_torch(img, kernel) + scat
            # same as metrics.log_loss, in torch so it can be differentiated
            loss = torch.mean((torch.log(pred + eps) - log_ref) ** 2) / len(data)
            loss.backward(retain_graph=True)
            total += loss.item()
        opt.step()
        history.append(total)
        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {total:.4e}")
    kernel = torch.softmax(theta, dim=0).reshape(kh, kw, c)
    return kernel.detach().cpu().numpy().astype(np.float32), history


def evaluate(names, pairs, scatters, kernel, ps, eps):
    """Per-image log loss, no-bloom loss and energy error of one uniform kernel."""
    per_image = {}
    for n, (img, ref), s in zip(names, pairs, scatters):
        out = fft_bloom(img, kernel, ps, scatter=s)
        e_in = img.astype(np.float64).sum()
        per_image[n] = {"loss": log_loss(out, ref, eps),
                        "loss_no_bloom": log_loss(img, ref, eps),
                        "energy_relative": float(out.astype(np.float64).sum() / e_in - 1)}
        print(f"  {n:<45} loss {per_image[n]['loss']:.2e} "
              f"(no bloom {per_image[n]['loss_no_bloom']:.4f}), "
              f"energy {per_image[n]['energy_relative']:+.2%}")
    mean = float(np.mean([v["loss"] for v in per_image.values()]))
    return {"mean_loss": mean, "per_image": per_image}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psf", default="data/psf/psf_quick.npz")
    ap.add_argument("--inputs", nargs="+", required=True, help="input EXRs; globs are expanded")
    ap.add_argument("--refs", default="data/reference")
    ap.add_argument("--out", default="outputs/01_gap_study/fft_bloom.json")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    with open(os.path.join(args.refs, "reference_report.json")) as fh:
        ref_report = json.load(fh)
    if os.path.normpath(ref_report["psf"]) != os.path.normpath(args.psf):
        raise ValueError(f"references were made with {ref_report['psf']}, not {args.psf}")
    ps = PSFSet(args.psf)
    if ps.placeholder:
        print("WARNING: PLACEHOLDER lens parameters. Do not report these results.\n")

    paths = sorted(f for pattern in args.inputs for f in glob.glob(pattern))
    names = [os.path.splitext(os.path.basename(f))[0] for f in paths]
    pairs = [(read_hdr(f), read_hdr(os.path.join(args.refs, f"{n}_ref.exr")))
             for f, n in zip(paths, names)]
    print("computing scatter terms...")
    scatters = [scatter_term(img, ps) for img, _ in pairs]

    kernels = {"centre": centre_psf(ps)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nfitting best-fit uniform kernel on {device}...")
    kernels["best_fit"], history = fit_uniform_kernel(
        pairs, scatters, ps.scatter.energy_fraction, kernels["centre"],
        eps=args.eps, steps=args.steps, lr=args.lr, device=device)
    variants = {}
    for name, k in kernels.items():
        print(f"\n{name} PSF")
        variants[name] = evaluate(names, pairs, scatters, k, ps, args.eps)
        print(f"  mean loss {variants[name]['mean_loss']:.2e}")

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    kernel_path = os.path.join(out_dir, "fft_kernels.npz")
    np.savez_compressed(kernel_path, **kernels)
    with open(args.out, "w") as fh:
        json.dump({"psf": args.psf, "placeholder": ps.placeholder, "numpy": np.__version__,
                    "scipy": scipy.__version__, "inputs": names, "eps": args.eps,
                    "kernels": os.path.basename(kernel_path), "variants": variants,
                    "torch": torch.__version__,
                   "fit": {"steps": args.steps, "lr": args.lr, "device": device,
                           "loss_history": history},
                   }, fh, indent=2)
    print(f"\nwrote {args.out} and {kernel_path}")


if __name__ == "__main__":
    main()
