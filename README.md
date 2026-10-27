# ComfyUI-mtlattn

Faster attention for ComfyUI on Apple Silicon. One node, **Apply mtlattn
Attention**, routes each of the model's attention calls to the fastest
correct backend on MPS:

- **Native PyTorch SDPA** for head_dim 64/96/128/256 (SDXL, Flux, SD3,
  Hunyuan3D, TRELLIS, Wan, HunyuanVideo, …). Since torch 2.14 this has a fast
  fused Metal 4 kernel, but ComfyUI on Mac defaults to its `sub_quad` backend
  instead, which is 2–12× slower at these shapes.
- **[mtlattn](https://github.com/lastowl/mtlattn)** for head_dim 80/88/160
  (SD1.5 at higher resolutions, Hunyuan-DiT), which torch's fast kernel
  doesn't cover, and for training-mode calls (mtlattn has a fused backward).
- **ComfyUI's default** for everything else (masked calls, other head dims).

## Measured (M5 Pro, macOS 26.5, torch 2.14.1, ComfyUI attention API)

| shape | dtype | sub_quad (ComfyUI default) | pytorch SDPA | this node |
|---|---|---|---|---|
| SD1.5 1024² self (B2 H8 **D80**, 4096) | fp16 | 25 ms | 13 ms | **10 ms** |
| SD1.5 self (B2 H8 **D160**, 1024) | fp16 | 3 ms | 4 ms | **~1 ms** (2.3×) |
| Hunyuan-DiT self (B2 H16 **D88**, 4096) | fp16 | 52 ms | 93 ms | **22 ms** |
| SDXL 1024² self (B2 H10 D64, 4096) | fp16 | 30 ms | 3 ms | 3 ms |
| Flux 1024² joint (H24 D128, 4608) | fp16 | 55 ms | 11 ms | 11 ms |
| TRELLIS 3D (H16 D64, 20000) | fp16 | 736 ms | 60 ms | 60 ms |
| Wan2.1 1.3B 480p·81f self (H12 D128, 32760) | bf16 | 595 ms | 288 ms | 282 ms |
| Wan2.1 14B 480p·81f self (H40 D128, 32760) | bf16 | 2060 ms | 1049 ms | 1068 ms |

All outputs agree with a CPU fp32 reference. Where native SDPA is used the
node simply matches it; the mtlattn rows are 1.3–2.4× faster than the best
native backend. Against ComfyUI's Mac default, every row is faster.

You can get the native-SDPA part without this node by launching ComfyUI with
`--use-pytorch-cross-attention`. The node adds mtlattn for the head dims
torch's fast kernel skips, and applies per model rather than globally.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lastowl/ComfyUI-mtlattn
ComfyUI/.venv/bin/pip install "mtlattn>=0.5"
```

mtlattn's published wheels are built against a specific torch version (a torch
C++ extension is ABI-tied). If ComfyUI's torch differs, build from source —
see [mtlattn's install notes](https://github.com/lastowl/mtlattn#install). The
pack self-tests at startup: on a mismatch it disables itself with a warning
and ComfyUI runs exactly as before.

## Use

Add **Apply mtlattn Attention** (category `model/patch`) between your model
loader and sampler. head_dim 80/88/160 calls with query length ≥
`min_seqlen` (default 1024) run on mtlattn; the rest go to native SDPA or
ComfyUI's default as described above.

The backend is also registered as `"mtlattn"` via
`comfy.ldm.modules.attention.register_attention_function` for other tooling.

Requires an Apple Silicon Mac on **macOS 26.2+** with **torch ≥ 2.14** and
**mtlattn ≥ 0.5**. mtlattn only wins on its Metal 4 path (fp16/bf16); on
older macOS the node falls back to native SDPA and ComfyUI's default.

## Benchmark

```bash
python bench/bench_attention.py          # add --big for HunyuanVideo 544p
```

Times ComfyUI's `sub_quad` and pytorch backends against this node's routing at
real model shapes, and checks each against a CPU fp32 reference on the last
query rows.
