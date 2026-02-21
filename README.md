# Chroma Radiance Standalone (CUDA/cuBLAS)

A high-performance, standalone text-to-image generator implementing the [Chroma Radiance](https://huggingface.co/lodestones/Chroma) model in pure CUDA/C++. No Python, no PyTorch — just a single binary that loads safetensors weights and generates images directly on the GPU.

## Features

- **Pure CUDA/C++** — zero Python dependencies, single ~5 MB binary
- **cuBLAS/cuBLASLt acceleration** — BF16 tensor core GEMMs with fused bias epilogues
- **GPU arena allocator** — pre-loads all 19 GB of model weights to VRAM at startup for consistent performance
- **Classifier-Free Guidance (CFG)** — optional negative prompt support for improved image quality
- **T5-XXL text encoder** — built-in tokenizer and encoder, no external text encoding needed

## Requirements

- NVIDIA GPU with compute capability 8.0+ (RTX 3000 series or newer)
- ~22 GB VRAM (model weights + inference buffers)
- CUDA Toolkit (cuBLAS, cuBLASLt, cudart)
- CMake 3.18+
- C++17 compiler

### Model Files

Download these safetensors files and place them in your working directory (or pass paths via CLI):

| File | Description |
|------|-------------|
| `Chroma1-Radiance-v0.4.safetensors` | Main Chroma Radiance model (~19 GB) |
| `t5xxl_fp16.safetensors` | T5-XXL encoder weights in FP16 |
| `t5_tokenizer.json` | T5 tokenizer vocabulary |

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

To target a specific GPU architecture:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3090
```

## Usage

```bash
./build/chroma-radiance -p "a photo of a cat" --steps 20 -W 512 -H 512
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `-p`, `--prompt` | `"a photo of a cat"` | Text prompt |
| `-n`, `--negative-prompt` | `""` | Negative prompt (only used when `--cfg-scale` > 1.0) |
| `--cfg-scale` | `1.0` | CFG scale (> 1.0 enables classifier-free guidance) |
| `-W`, `--width` | `512` | Image width (multiple of 16) |
| `-H`, `--height` | `512` | Image height (multiple of 16) |
| `--steps` | `20` | Number of sampling steps |
| `--seed` | `42` | Random seed |
| `-o`, `--output` | `output.ppm` | Output image path |
| `-m`, `--model` | `Chroma1-Radiance-v0.4.safetensors` | Chroma model path |
| `--t5` | `t5xxl_fp16.safetensors` | T5 encoder weights path |
| `--tokenizer` | `t5_tokenizer.json` | T5 tokenizer path |
| `--rng` | `sdcpp` | RNG mode (`pytorch` or `sdcpp`) |
| `--no-mask` | `false` | Disable attention masking on padding tokens |
| `--debug` | `false` | Dump intermediate tensors to `/tmp/` |

### Examples

```bash
# Basic generation
./build/chroma-radiance -p "a sunset over mountains" --steps 30 -o sunset.ppm

# Higher resolution
./build/chroma-radiance -p "portrait of a woman" -W 1024 -H 1024 --steps 25

# With classifier-free guidance
./build/chroma-radiance -p "a beautiful landscape" -n "ugly, blurry" --cfg-scale 7.0
```

### Output Format

Images are saved as PPM (Portable PixMap) files. Convert to PNG/JPEG with ImageMagick:

```bash
convert output.ppm output.png
```

## Architecture

The inference pipeline runs in 5 phases:

1. **T5 Text Encoding** — tokenize prompt and run T5-XXL encoder to produce text embeddings
2. **Model Loading** — pre-load all Chroma Radiance weights into a GPU arena
3. **Precompute** — positional embeddings (RoPE), DCT features, attention masks
4. **Sampling** — Euler denoising loop through 19 double-stream + 38 single-stream transformer blocks
5. **Output** — NeRF MLP decoder, unpatchify, and save to disk

### File Structure

```
main.cu            — CLI, pipeline orchestration
chroma.cuh         — Chroma Radiance model (double/single stream blocks, NeRF decoder)
t5.cuh             — T5-XXL encoder
kernels.cu         — Custom CUDA kernels (softmax, GELU, RMSNorm, RoPE, etc.)
cublas_ops.cuh     — cuBLAS/cuBLASLt linear algebra (GEMM, batched matmul)
tensor.cuh         — GPU tensor abstraction with memory pool
attention.cuh      — Multi-head attention
rope.cuh           — Rotary positional embeddings
sampler.cuh        — Euler sampler with sigma scheduling
safetensors.cuh    — Safetensors file parser
t5_tokenizer.cuh   — T5 tokenizer (SentencePiece-compatible)
image_io.cuh       — PPM image output
CMakeLists.txt     — Build configuration
```

## Performance

On an RTX 3090 at 512x512 with 2 sampling steps:

- **~583 ms/step** (73% cuBLAS GEMMs, 27% custom kernels)
- T5 encoding: ~165 ms
- Model loading: ~5 s (one-time, from SSD)

## License

This project is provided as-is. The Chroma Radiance model weights are subject to their own licensing terms.
