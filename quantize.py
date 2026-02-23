#!/usr/bin/env python3
"""Offline INT8 weight quantization for Chroma Radiance safetensors files.

Converts BF16 weight matrices to INT8 with per-group scales.
Non-weight tensors (biases, norms, embeddings) are copied unchanged.

Features:
  --asymmetric     Asymmetric quantization: maps [min,max] to [-128,127] with zero_point
  --smooth ALPHA   SmoothQuant: per-channel weight equalization (alpha in [0,1], recommend 0.5)
  --gptq           GPTQ: second-order optimal rounding using weight Hessian proxy
  --clip mse       MSE-optimal clipping: grid-search for clip ratio minimizing per-group MSE
  --awq            AWQ: activation-aware weight quantization (searches optimal scaling alpha)

Usage:
    python quantize.py input.safetensors output.safetensors [--group-size 128] [--asymmetric] [--smooth 0.5] [--gptq] [--clip mse] [--awq]
"""

import argparse
import math
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


def hadamard_matrix(n: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix of size n (must be power of 2).

    Returns orthogonal H such that H @ H^T = I.
    Constructed via Sylvester's recursive method with 1/sqrt(2) normalization
    at each level so the final matrix is orthonormal.
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1),
                        torch.cat([H, -H], 1)], 0) / math.sqrt(2)
    return H


def block_hadamard_rotate(w: torch.Tensor, K: int) -> torch.Tensor:
    """Rotate W[N, K] with block-diagonal normalized Hadamard matrix.

    Uses the largest power-of-2 block size that divides K.
    For K=3072: block_size=1024, 3 blocks of H_1024.
    Result: W_rot = W @ blkdiag(H, H, ..., H).
    Since H is orthonormal, W_rot @ H^T = W (the rotation is reversible).
    """
    block_size = K & (-K)  # largest power of 2 dividing K
    num_blocks = K // block_size
    H = hadamard_matrix(block_size).to(w.device, w.dtype)  # [block_size, block_size]
    # Reshape to [N, num_blocks, block_size], rotate each block, reshape back
    W_blocks = w.reshape(-1, num_blocks, block_size)
    W_rot = W_blocks @ H.T
    return W_rot.reshape(-1, K)


def compute_group_params(w: torch.Tensor, group_size: int, asymmetric: bool,
                         clip_ratio: torch.Tensor = None):
    """Compute per-group quantization parameters from weight values.

    Args:
        w: Weight matrix [N, K] (float32, possibly smoothed)
        group_size: Number of elements per quantization group along K
        asymmetric: If True, use asymmetric [min,max]->[-128,127] mapping
        clip_ratio: [N, num_groups] optional per-group clip ratios (from MSE search)
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
        if clip_ratio is not None:
            center = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            clip_half = half_range * clip_ratio
            min_val = center - clip_half
            max_val = center + clip_half
        scale = ((max_val - min_val) / 255.0).clamp(min=1e-10)
        zero_point = (-128.0 - min_val / scale).round().clamp(-128, 127).to(torch.int8)
    else:
        absmax = w_groups.abs().amax(dim=2)      # [N, num_groups]
        if clip_ratio is not None:
            absmax = absmax * clip_ratio
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


def dequantize_weights(w_int8: torch.Tensor, scale: torch.Tensor,
                       zero_point, group_size: int) -> torch.Tensor:
    """Dequantize INT8 weights back to float32.

    Args:
        w_int8: [N, K] int8 quantized weights
        scale: [N, num_groups] float32 scales
        zero_point: [N, num_groups] int8 or None
        group_size: Group size
    Returns:
        w_float: [N, K] float32 reconstructed weights
    """
    N, K = w_int8.shape
    num_groups = scale.shape[1]
    K_pad = num_groups * group_size

    if K_pad > K:
        w_padded = F.pad(w_int8.float(), (0, K_pad - K))
    else:
        w_padded = w_int8.float()

    w_groups = w_padded.reshape(N, num_groups, group_size)

    if zero_point is not None:
        w_float = (w_groups - zero_point.unsqueeze(2).float()) * scale.unsqueeze(2)
    else:
        w_float = w_groups * scale.unsqueeze(2)

    return w_float.reshape(N, K_pad)[:, :K].contiguous()


def find_optimal_clip_ratio(w: torch.Tensor, group_size: int, asymmetric: bool,
                            n_candidates: int = 100) -> torch.Tensor:
    """Grid-search for per-group clip ratio that minimizes quantization MSE.

    Instead of using the full absmax range, searches for a clip ratio in [0.5, 1.0]
    that minimizes the mean squared error between original and dequantized weights
    per group. Outlier weights are clipped but the remaining values get finer
    quantization granularity.

    Args:
        w: Weight matrix [N, K] float32
        group_size: Group size
        asymmetric: Whether to use asymmetric quantization
        n_candidates: Number of clip ratios to evaluate
    Returns:
        best_ratio: [N, num_groups] optimal clip ratios
    """
    N, K = w.shape
    num_groups = (K + group_size - 1) // group_size
    K_pad = num_groups * group_size

    if K_pad > K:
        w_padded = F.pad(w, (0, K_pad - K))
    else:
        w_padded = w

    w_groups = w_padded.reshape(N, num_groups, group_size)

    best_mse = torch.full((N, num_groups), float('inf'), device=w.device)
    best_ratio = torch.ones(N, num_groups, device=w.device)

    if asymmetric:
        min_val = w_groups.amin(dim=2)
        max_val = w_groups.amax(dim=2)
        center = (min_val + max_val) / 2
        half_range = ((max_val - min_val) / 2).clamp(min=1e-10)
    else:
        absmax = w_groups.abs().amax(dim=2).clamp(min=1e-10)

    for i in range(n_candidates):
        r = 0.5 + 0.5 * i / max(n_candidates - 1, 1)

        if asymmetric:
            clip_half = (half_range * r).unsqueeze(2)
            clip_center = center.unsqueeze(2)
            clip_min = clip_center - clip_half
            clip_max = clip_center + clip_half
            w_clipped = w_groups.clamp(clip_min, clip_max)
            sc = ((clip_max - clip_min) / 255.0).clamp(min=1e-10)
            zp = (-128.0 - clip_min / sc).round().clamp(-128, 127)
            q = (w_clipped / sc + zp).round().clamp(-128, 127)
            deq = (q - zp) * sc
        else:
            clip_max = (absmax * r).unsqueeze(2)
            w_clipped = w_groups.clamp(-clip_max, clip_max)
            sc = (clip_max / 127.0).clamp(min=1e-10)
            q = (w_clipped / sc).round().clamp(-128, 127)
            deq = q * sc

        mse = (w_groups - deq).pow(2).mean(dim=2)

        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_ratio = torch.where(improved, r, best_ratio)

    return best_ratio


def compute_awq_scales(w: torch.Tensor, act_absmax: torch.Tensor,
                       group_size: int, asymmetric: bool,
                       use_clip_mse: bool, n_grid: int = 20) -> tuple:
    """Search for per-channel AWQ scaling that minimizes activation-weighted quantization error.

    AWQ (Activation-Aware Weight Quantization) searches for the optimal alpha that
    balances activation and weight magnitude when computing per-channel scaling factors:
        s[k] = act_absmax[k]^alpha / weight_absmax[k]^(1-alpha)

    The optimal alpha minimizes activation-weighted reconstruction MSE. This uses the
    same scaling artifact (.smooth) as SmoothQuant but with a searched alpha per layer.

    Args:
        w: Weight matrix [N, K] float32
        act_absmax: [K] per-channel activation absmax from calibration
        group_size: Quantization group size
        asymmetric: Whether to use asymmetric quantization
        use_clip_mse: Whether to use MSE-optimal clipping during search
        n_grid: Number of alpha candidates (alpha = i/n_grid for i=0..n_grid)
    Returns:
        (scales, best_alpha, clip_ratio):
            scales: [K] per-channel scaling factors
            best_alpha: float, optimal alpha value
            clip_ratio: [N, num_groups] or None, clip ratios for best alpha
    """
    N, K = w.shape
    w_absmax = w.abs().amax(dim=0).clamp(min=1e-5)  # [K]
    act_clamped = act_absmax.clamp(min=1e-5)

    best_mse = float('inf')
    best_alpha = 0.0
    best_scales = torch.ones(K, device=w.device, dtype=w.dtype)
    best_clip_ratio = None

    for i in range(n_grid + 1):
        alpha = i / n_grid

        s = act_clamped.pow(alpha) / w_absmax.pow(1 - alpha)
        s = s.clamp(min=1e-5)

        w_scaled = w * s.unsqueeze(0)

        # Find optimal clip ratios if requested
        cr = find_optimal_clip_ratio(w_scaled, group_size, asymmetric) if use_clip_mse else None

        # Quantize and dequantize with current scaling
        scale, zp = compute_group_params(w_scaled, group_size, asymmetric, cr)
        w_q = standard_quantize(w_scaled, scale, zp, group_size)
        w_deq = dequantize_weights(w_q, scale, zp, group_size)

        # Undo scaling to measure error in original weight space
        w_recon = w_deq / s.unsqueeze(0)

        # Activation-weighted MSE: channels with larger activations matter more
        mse = ((w - w_recon).pow(2) * act_clamped.unsqueeze(0).pow(2)).mean().item()

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_scales = s.clone()
            best_clip_ratio = cr.clone() if cr is not None else None

    return best_scales, best_alpha, best_clip_ratio


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
                     calib_path: str = "", use_awq: bool = False,
                     clip_mode: str = "absmax", hadamard: bool = False) -> None:
    print(f"Loading {input_path}...")
    tensors = load_file(input_path)

    # Load calibration data if provided
    calib_data = {}
    if calib_path:
        calib_data = load_file(calib_path)
        print(f"  Loaded calibration data: {len(calib_data)} tensors")

    use_clip_mse = (clip_mode == "mse")

    # Use GPU for compute-intensive operations
    use_gpu = (use_gptq or use_awq or use_clip_mse) and torch.cuda.is_available()
    dev = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        print(f"  Using GPU for computation")

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

            if use_gpu:
                w = w.to(dev)

            # Step 1: AWQ or SmoothQuant channel equalization
            smooth = None
            clip_ratio = None
            if use_awq:
                act_abs = calib_data.get(name)
                if act_abs is not None:
                    act_abs = act_abs.float().to(w.device)
                    smooth, best_alpha, clip_ratio = compute_awq_scales(
                        w, act_abs, group_size, asymmetric, use_clip_mse)
                    w = w * smooth.unsqueeze(0)
            elif smooth_alpha > 0:
                act_abs = calib_data.get(name)
                if act_abs is not None:
                    act_abs = act_abs.float().to(w.device)
                smooth = compute_smooth_factors(w, smooth_alpha, act_abs)  # [K]
                w = w * smooth.unsqueeze(0)  # W_smooth[n,k] = W[n,k] * s[k]

            # Step 2: Hadamard rotation (after smooth/AWQ, before clip/quantize)
            # Rotates the K dimension with a block-diagonal normalized Hadamard matrix.
            # Skip weights whose activations are quantized by fused kernels that don't
            # yet support WHT (MLP_2: K=12288, linear2: K=15360).
            if hadamard:
                skip_had = ("mlp.2." in name or "linear2." in name)
                if not skip_had:
                    w = block_hadamard_rotate(w, K)

            # Step 3: MSE-optimal clipping (if not already done by AWQ)
            if use_clip_mse and clip_ratio is None:
                clip_ratio = find_optimal_clip_ratio(w, group_size, asymmetric)

            # Step 4: Compute per-group quantization parameters
            scale, zero_point = compute_group_params(w, group_size, asymmetric, clip_ratio)

            # Step 5: Quantize (GPTQ or standard round-to-nearest)
            if use_gptq:
                w_int8 = gptq_quantize(w, scale, zero_point, group_size)
            else:
                w_int8 = standard_quantize(w, scale, zero_point, group_size)

            # Move results to CPU for storage
            w_int8 = w_int8.cpu()
            scale = scale.cpu()
            if zero_point is not None:
                zero_point = zero_point.cpu()
            if smooth is not None:
                smooth = smooth.cpu()

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
            extra = f" ({dt:.1f}s)"
        print(f"\r  [{bar}] {i+1}/{total} ({pct*100:.0f}%) {label} {name[:55]:<55}{extra}", end="", flush=True)

    mode_parts = []
    if asymmetric:
        mode_parts.append("asymmetric")
    else:
        mode_parts.append("symmetric")
    if use_awq:
        mode_parts.append("AWQ")
    elif smooth_alpha > 0:
        mode_parts.append(f"smooth={smooth_alpha}")
    if use_clip_mse:
        mode_parts.append("clip=mse")
    if use_gptq:
        mode_parts.append("GPTQ")
    if hadamard:
        mode_parts.append("hadamard")
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
    if use_awq:
        metadata["quantization_awq"] = "true"
    elif smooth_alpha > 0:
        metadata["quantization_smooth_alpha"] = str(smooth_alpha)
    if use_clip_mse:
        metadata["quantization_clip"] = "mse"
    if use_gptq:
        metadata["quantization_gptq"] = "true"
    if hadamard:
        metadata["quantization_hadamard"] = "true"
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
    parser.add_argument("--clip", type=str, default="absmax", choices=["absmax", "mse"],
                        help="Clipping strategy: absmax (default) or mse (grid-search optimal clip ratio)")
    parser.add_argument("--awq", action="store_true",
                        help="AWQ: activation-aware weight quantization (searches optimal scaling alpha per layer)")
    parser.add_argument("--calib", type=str, default="", metavar="FILE",
                        help="Calibration file (safetensors) with per-channel activation absmax")
    parser.add_argument("--hadamard", action="store_true",
                        help="Apply Hadamard rotation to weight K dimensions before quantization (improves per-channel INT8)")
    args = parser.parse_args()

    if args.group_size < 1:
        print("Error: --group-size must be >= 1")
        exit(1)
    if not 0 <= args.smooth <= 1:
        print("Error: --smooth alpha must be in [0, 1]")
        exit(1)
    if args.awq and args.smooth > 0:
        print("Error: --awq and --smooth are mutually exclusive (both produce .smooth scaling)")
        exit(1)
    if args.awq and not args.calib:
        print("Error: --awq requires --calib (activation calibration data)")
        exit(1)

    quantize_weights(args.input, args.output, args.group_size, args.scale_dtype,
                     args.asymmetric, args.smooth, args.gptq, args.calib,
                     args.awq, args.clip, args.hadamard)
