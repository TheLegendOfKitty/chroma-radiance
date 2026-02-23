#!/usr/bin/env python3
"""Offline INT8 weight quantization for Chroma Radiance safetensors files.

Converts BF16 weight matrices to INT8 with per-group scales.
Non-weight tensors (biases, norms, embeddings) are copied unchanged.

Features:
  --asymmetric     Asymmetric quantization: maps [min,max] to [-128,127] with zero_point
  --smooth ALPHA   SmoothQuant: per-channel weight equalization (alpha in [0,1], recommend 0.5)
  --gptq           GPTQ: second-order optimal rounding using weight Hessian proxy

Usage:
    python quantize.py input.safetensors output.safetensors [--group-size 128] [--asymmetric] [--smooth 0.5] [--gptq]
"""

import argparse
import time
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from collections import OrderedDict


def compute_smooth_factors(w: torch.Tensor, alpha: float = 0.5,
                           act_absmax: torch.Tensor = None) -> torch.Tensor:
    """Compute per-input-channel smoothing factors from weight statistics.

    SmoothQuant identity: X @ W^T = (X / s) @ (s * W)^T

    With calibration data (recommended):
        s[k] = act_absmax[k]^alpha / weight_absmax[k]^(1-alpha)

    Without calibration (weight-only proxy, poor accuracy):
        s[k] = weight_absmax[k]^alpha

    Args:
        w: Weight matrix [N, K]
        alpha: Smoothing strength in [0, 1]. 0 = no smoothing, 1 = full equalization
        act_absmax: [K] per-channel activation absmax from calibration, or None
    Returns:
        s: Smoothing factors [K], always positive
    """
    weight_absmax = w.abs().amax(dim=0).clamp(min=1e-5)  # [K]
    if act_absmax is not None:
        # Full SmoothQuant: s = act^alpha / weight^(1-alpha)
        s = act_absmax.clamp(min=1e-5).pow(alpha) / weight_absmax.pow(1 - alpha)
    else:
        # Weight-only fallback (not recommended)
        s = weight_absmax.pow(alpha)
    return s.clamp(min=1e-5)


def compute_group_params(w: torch.Tensor, group_size: int, asymmetric: bool):
    """Compute per-group quantization parameters from weight values.

    Args:
        w: Weight matrix [N, K] (float32, possibly smoothed)
        group_size: Number of elements per quantization group along K
        asymmetric: If True, use asymmetric [min,max]->[-128,127] mapping
    Returns:
        scale: [N, num_groups] float32
        zero_point: [N, num_groups] int8 or None (if symmetric)
    """
    N, K = w.shape
    num_groups = (K + group_size - 1) // group_size
    K_pad = num_groups * group_size

    if K_pad > K:
        w_padded = F.pad(w, (0, K_pad - K))
    else:
        w_padded = w

    w_groups = w_padded.reshape(N, num_groups, group_size)

    if asymmetric:
        min_val = w_groups.amin(dim=2)           # [N, num_groups]
        max_val = w_groups.amax(dim=2)           # [N, num_groups]
        scale = ((max_val - min_val) / 255.0).clamp(min=1e-10)
        zero_point = (-128.0 - min_val / scale).round().clamp(-128, 127).to(torch.int8)
    else:
        absmax = w_groups.abs().amax(dim=2)      # [N, num_groups]
        scale = (absmax / 127.0).clamp(min=1e-10)
        zero_point = None

    return scale, zero_point


def standard_quantize(w: torch.Tensor, scale: torch.Tensor,
                      zero_point, group_size: int) -> torch.Tensor:
    """Standard round-to-nearest quantization with pre-computed group params.

    Args:
        w: Weight matrix [N, K] float32
        scale: [N, num_groups] float32
        zero_point: [N, num_groups] int8 or None
        group_size: Group size
    Returns:
        w_int8: [N, K] int8
    """
    N, K = w.shape
    num_groups = scale.shape[1]
    K_pad = num_groups * group_size

    if K_pad > K:
        w_padded = F.pad(w, (0, K_pad - K))
    else:
        w_padded = w

    w_groups = w_padded.reshape(N, num_groups, group_size)

    if zero_point is not None:
        w_int8 = (w_groups / scale.unsqueeze(2)
                  + zero_point.unsqueeze(2).float()).round().clamp(-128, 127).to(torch.int8)
    else:
        w_int8 = (w_groups / scale.unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)

    return w_int8.reshape(N, K_pad)[:, :K].contiguous()


def gptq_quantize(w: torch.Tensor, scale: torch.Tensor, zero_point,
                  group_size: int, blocksize: int = 128,
                  percdamp: float = 0.01) -> torch.Tensor:
    """GPTQ optimal rounding with block-wise Hessian error propagation.

    Uses H = W^T @ W / N as Hessian proxy (no calibration data needed).
    Processes columns left-to-right in blocks, propagating quantization error
    to unprocessed columns via the inverse Hessian to minimize total error.

    Args:
        w: Weight matrix [N, K] float32 (possibly smoothed)
        scale: [N, num_groups] float32 (pre-computed group scales)
        zero_point: [N, num_groups] int8 or None
        group_size: Group size for scale indexing
        blocksize: Number of columns per block for batched error propagation
        percdamp: Hessian dampening as fraction of diagonal mean
    Returns:
        w_int8: [N, K] int8 (optimally rounded)
    """
    N, K = w.shape
    dev = w.device

    # Compute Hessian proxy and its inverse
    H = (w.T @ w) / N  # [K, K]
    damp = percdamp * H.diagonal().mean()
    H.diagonal().add_(damp)

    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
        del L
    except RuntimeError:
        print("      (Cholesky failed, using pseudo-inverse)")
        H_inv = torch.linalg.pinv(H)
    del H

    w_q = w.clone()
    w_int8 = torch.zeros(N, K, dtype=torch.int8, device=dev)
    num_groups = scale.shape[1]

    for b_start in range(0, K, blocksize):
        b_end = min(b_start + blocksize, K)
        b_size = b_end - b_start

        # Working copy of this block's columns
        W1 = w_q[:, b_start:b_end].clone()  # [N, b_size]
        Q1 = torch.zeros_like(W1)            # dequantized values
        Err = torch.zeros_like(W1)           # scaled errors for batch propagation
        H1 = H_inv[b_start:b_end, b_start:b_end]  # [b_size, b_size]

        for j_local in range(b_size):
            j = b_start + j_local
            g = min(j // group_size, num_groups - 1)
            s = scale[:, g]  # [N]

            w_col = W1[:, j_local]  # [N]

            # Quantize column
            if zero_point is not None:
                zp = zero_point[:, g].float()  # [N]
                q = (w_col / s + zp).round().clamp(-128, 127)
                w_hat = (q - zp) * s
            else:
                q = (w_col / s).round().clamp(-128, 127)
                w_hat = q * s

            w_int8[:, j] = q.to(torch.int8)
            Q1[:, j_local] = w_hat

            # Error propagation within block
            d = H1[j_local, j_local].clamp(min=1e-8)
            err = (w_col - w_hat) / d  # [N]
            Err[:, j_local] = err

            if j_local + 1 < b_size:
                # Intra-block: update remaining columns in this block
                W1[:, j_local + 1:] -= err.unsqueeze(1) * H1[j_local, j_local + 1:].unsqueeze(0)

        # Replace processed block with dequantized values
        w_q[:, b_start:b_end] = Q1

        # Inter-block: batch propagate error to all remaining columns
        if b_end < K:
            w_q[:, b_end:] -= Err @ H_inv[b_start:b_end, b_end:]

    del H_inv, w_q
    return w_int8


def quantize_weights(input_path: str, output_path: str, group_size: int = 128,
                     scale_dtype: str = "f32", asymmetric: bool = False,
                     smooth_alpha: float = 0.0, use_gptq: bool = False,
                     calib_path: str = "") -> None:
    print(f"Loading {input_path}...")
    tensors = load_file(input_path)

    # Load calibration data if provided
    calib_data = {}
    if calib_path:
        calib_data = load_file(calib_path)
        print(f"  Loaded calibration data: {len(calib_data)} tensors")

    # Use GPU if available for GPTQ (Cholesky + matmul much faster)
    dev = torch.device("cuda" if use_gptq and torch.cuda.is_available() else "cpu")
    if use_gptq and dev.type == "cuda":
        print(f"  Using GPU for GPTQ computation")

    total = len(tensors)
    output = OrderedDict()
    n_quantized = 0
    n_copied = 0
    original_bytes = 0
    quantized_bytes = 0

    for i, (name, tensor) in enumerate(tensors.items()):
        original_bytes += tensor.nelement() * tensor.element_size()

        # Only quantize 2D weight matrices (not biases, norms, conv weights, etc.)
        is_weight = tensor.ndim == 2 and "weight" in name
        if is_weight:
            t0 = time.time()
            w = tensor.float()  # [N, K]
            N, K = w.shape

            # Step 1: SmoothQuant channel equalization
            smooth = None
            if smooth_alpha > 0:
                act_abs = calib_data.get(name)
                if act_abs is not None:
                    act_abs = act_abs.float()
                smooth = compute_smooth_factors(w, smooth_alpha, act_abs)  # [K]
                w = w * smooth.unsqueeze(0)  # W_smooth[n,k] = W[n,k] * s[k]

            # Step 2: Compute per-group quantization parameters
            scale, zero_point = compute_group_params(w, group_size, asymmetric)

            # Step 3: Quantize (GPTQ or standard round-to-nearest)
            if use_gptq:
                w_dev = w.to(dev)
                scale_dev = scale.to(dev)
                zp_dev = zero_point.to(dev) if zero_point is not None else None
                w_int8 = gptq_quantize(w_dev, scale_dev, zp_dev, group_size).cpu()
                del w_dev, scale_dev, zp_dev
            else:
                w_int8 = standard_quantize(w, scale, zero_point, group_size)

            dt = time.time() - t0

            # Convert scale dtype for storage
            if scale_dtype == "f32":
                scale_out = scale
            elif scale_dtype == "fp16":
                scale_out = scale.to(torch.float16)
            elif scale_dtype == "bf16":
                scale_out = scale.to(torch.bfloat16)
            else:
                raise ValueError(f"Unsupported scale_dtype: {scale_dtype}")

            output[name] = w_int8                              # [N, K] INT8
            output[name + ".scale"] = scale_out                # [N, num_groups]

            quantized_bytes += w_int8.nelement() * w_int8.element_size()
            quantized_bytes += scale_out.nelement() * scale_out.element_size()

            if asymmetric and zero_point is not None:
                output[name + ".zp"] = zero_point              # [N, num_groups] INT8
                quantized_bytes += zero_point.nelement() * zero_point.element_size()

            if smooth is not None:
                output[name + ".smooth"] = smooth              # [K] F32
                quantized_bytes += smooth.nelement() * smooth.element_size()

            n_quantized += 1
        else:
            output[name] = tensor
            quantized_bytes += tensor.nelement() * tensor.element_size()
            n_copied += 1

        # Progress bar
        pct = (i + 1) / total
        bar_len = 40
        filled = int(bar_len * pct)
        bar = "=" * filled + "-" * (bar_len - filled)
        label = "Q" if is_weight else " "
        extra = ""
        if is_weight:
            extra = f" ({dt:.1f}s)" if use_gptq else ""
        print(f"\r  [{bar}] {i+1}/{total} ({pct*100:.0f}%) {label} {name[:55]:<55}{extra}", end="", flush=True)

    mode_parts = []
    if asymmetric:
        mode_parts.append("asymmetric")
    else:
        mode_parts.append("symmetric")
    if smooth_alpha > 0:
        mode_parts.append(f"smooth={smooth_alpha}")
    if use_gptq:
        mode_parts.append("GPTQ")
    mode_str = ", ".join(mode_parts)

    print()
    print(f"\nQuantized {n_quantized} weight matrices ({mode_str}, group_size={group_size}, scale_dtype={scale_dtype}), copied {n_copied} other tensors")
    print(f"Original size:  {original_bytes / 1e9:.2f} GB")
    print(f"Quantized size: {quantized_bytes / 1e9:.2f} GB")
    print(f"Reduction:      {(1 - quantized_bytes / original_bytes) * 100:.1f}%")

    print(f"\nSaving to {output_path}...")
    metadata = {"quantization_group_size": str(group_size), "quantization_scale_dtype": scale_dtype}
    if asymmetric:
        metadata["quantization_asymmetric"] = "true"
    if smooth_alpha > 0:
        metadata["quantization_smooth_alpha"] = str(smooth_alpha)
    if use_gptq:
        metadata["quantization_gptq"] = "true"
    save_file(output, output_path, metadata=metadata)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Chroma Radiance weights to INT8 with per-group scales")
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("output", help="Output safetensors file")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")
    parser.add_argument("--scale-dtype", type=str, default="f32",
                        choices=["f32", "fp16", "bf16"],
                        help="Storage dtype for quant scales (default: f32)")
    parser.add_argument("--asymmetric", action="store_true",
                        help="Use asymmetric quantization (maps [min,max] to [-128,127] with zero_point)")
    parser.add_argument("--smooth", type=float, default=0.0, metavar="ALPHA",
                        help="SmoothQuant: per-channel weight equalization (alpha in [0,1], recommend 0.5)")
    parser.add_argument("--gptq", action="store_true",
                        help="GPTQ: second-order optimal rounding (uses weight Hessian proxy)")
    parser.add_argument("--calib", type=str, default="", metavar="FILE",
                        help="Calibration file (safetensors) with per-channel activation absmax for SmoothQuant")
    args = parser.parse_args()

    if args.group_size < 1:
        print("Error: --group-size must be >= 1")
        exit(1)
    if not 0 <= args.smooth <= 1:
        print("Error: --smooth alpha must be in [0, 1]")
        exit(1)

    quantize_weights(args.input, args.output, args.group_size, args.scale_dtype,
                     args.asymmetric, args.smooth, args.gptq, args.calib)
