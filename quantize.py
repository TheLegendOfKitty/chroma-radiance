#!/usr/bin/env python3
"""Offline INT8 weight quantization for Chroma Radiance safetensors files.

Converts BF16 weight matrices to INT8 with per-group scales.
Non-weight tensors (biases, norms, embeddings) are copied unchanged.

Usage:
    python quantize.py input.safetensors output-int8.safetensors [--group-size 128] [--scale-dtype bf16]
"""

import argparse
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from collections import OrderedDict


def quantize_weights(input_path: str, output_path: str, group_size: int = 128,
                     scale_dtype: str = "f32") -> None:
    print(f"Loading {input_path}...")
    tensors = load_file(input_path)

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
            # Convert to float32 for accurate quantization
            w = tensor.float()  # [N, K]
            N, K = w.shape
            num_groups = (K + group_size - 1) // group_size
            K_pad = num_groups * group_size

            # Pad K to multiple of group_size if needed
            if K_pad > K:
                w_padded = F.pad(w, (0, K_pad - K))
            else:
                w_padded = w

            # Reshape into groups: [N, num_groups, group_size]
            w_groups = w_padded.reshape(N, num_groups, group_size)

            # Per-group absmax
            absmax = w_groups.abs().amax(dim=2)           # [N, num_groups]
            scale = (absmax / 127.0).clamp(min=1e-10)     # [N, num_groups]

            # Quantize
            w_int8 = (w_groups / scale.unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)
            w_int8 = w_int8.reshape(N, K_pad)[:, :K].contiguous()  # Trim padding

            if scale_dtype == "f32":
                scale_out = scale
            elif scale_dtype == "fp16":
                scale_out = scale.to(torch.float16)
            elif scale_dtype == "bf16":
                scale_out = scale.to(torch.bfloat16)
            else:
                raise ValueError(f"Unsupported scale_dtype: {scale_dtype}")

            output[name] = w_int8                          # [N, K] INT8
            output[name + ".scale"] = scale_out            # [N, num_groups] F32/FP16/BF16

            quantized_bytes += w_int8.nelement() * w_int8.element_size()
            quantized_bytes += scale_out.nelement() * scale_out.element_size()
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
        print(f"\r  [{bar}] {i+1}/{total} ({pct*100:.0f}%) {label} {name[:60]:<60}", end="", flush=True)

    print()
    print(f"\nQuantized {n_quantized} weight matrices (group_size={group_size}, scale_dtype={scale_dtype}), copied {n_copied} other tensors")
    print(f"Original size:  {original_bytes / 1e9:.2f} GB")
    print(f"Quantized size: {quantized_bytes / 1e9:.2f} GB")
    print(f"Reduction:      {(1 - quantized_bytes / original_bytes) * 100:.1f}%")

    print(f"\nSaving to {output_path}...")
    metadata = {"quantization_group_size": str(group_size), "quantization_scale_dtype": scale_dtype}
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
    args = parser.parse_args()

    if args.group_size < 1:
        print("Error: --group-size must be >= 1")
        exit(1)

    quantize_weights(args.input, args.output, args.group_size, args.scale_dtype)
