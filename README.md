# ComfyUI-mtlattn

Fused Metal flash attention for ComfyUI on Apple Silicon, backed by
[mtlattn](https://github.com/lastowl/mtlattn). Aimed at the workloads where
attention dominates on Mac — video DiTs (Wan, HunyuanVideo, Mochi, LTX),
3D generation (Hunyuan3D, TRELLIS-family), Flux/SDXL at high resolution.

What you get over ComfyUI's stock attention on MPS:

- **Speed** on macOS 26.2+: attention runs on the Metal 4 `matmul2d` path —
  on M5 the Neural Accelerator. Measured against ComfyUI's backends below.
- **Correctness at video sizes on torch ≤ 2.12.** When `B·H·Nq·Nkv` exceeds
  2³² elements (a Wan 480p×81f call is ~3× over), MPS SDPA on torch ≤ 2.12
  silently returns corrupted values
  ([pytorch#179352](https://github.com/pytorch/pytorch/issues/179352));
  mtlattn is unaffected. Fixed upstream in torch 2.13.
- **No materialized score matrix** — long sequences don't blow up unified
  memory.

## Measured (M5 Pro, macOS 26.5, torch 2.13, ComfyUI attention API)

| shape | dtype | sub_quad (ComfyUI default) | pytorch SDPA | mtlattn |
|---|---|---|---|---|
| SDXL 1024² self (B2 H10 D64, 4096) | fp16 | 34 ms | 13 ms | **10 ms** |
| Flux 1024² joint (H24 D128, 4608) | fp16 | 60 ms | 39 ms | **35 ms** |
| Hunyuan3D-2 (B2 H16 D64, 3072) | fp16 | 31 ms | 12 ms | **9 ms** |
| TRELLIS 3D (H16 D64, 20000) | fp16 | 757 ms | 242 ms | **169 ms** |
| Wan2.1 480p·81f self (H12 D128, 32760) | fp16 | 1688 ms | 974 ms | **628 ms** |
| Wan2.1 480p·81f self (H12 D128, 32760) | bf16 | 624 ms | 978 ms | **627 ms** |

All backends produced correct output on torch 2.13 (verified against a CPU
fp32 reference). Versus the sub_quad default that ComfyUI picks on Mac,
mtlattn is ~1.7–4.5× faster in fp16; versus the best native backend per
shape it's 1.1–1.55×. After mtlattn's head-major grid fix the kernel
sustains ~10.5 TF/s flat to 64k tokens, so even **bf16** long self-attention
— where MPS's bf16 matmul hits a fast path on M5 that fp16 does not — is now
a tie at 32k (627 vs 624 ms); the one remaining loss is a narrow ~16k bf16
window (155 vs 139 ms). On torch ≤ 2.12 the picture shifts further toward
mtlattn: native SDPA there is ~2.3× slower than 2.13 and corrupts past 2³²
score elements.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lastowl/ComfyUI-mtlattn
ComfyUI/.venv/bin/pip install mtlattn
```

mtlattn's published wheels are built against a specific torch version (a torch
C++ extension is ABI-tied). If ComfyUI's torch differs, build from source —
see [mtlattn's install notes](https://github.com/lastowl/mtlattn#install). The
pack self-tests at startup: on a mismatch it disables itself with a warning
and ComfyUI runs exactly as before.

## Use

Add **Apply mtlattn Attention** (category `model/patch`) between your model
loader and sampler. That's it — attention calls with query length ≥
`min_seqlen` (default 1024) run on mtlattn; shorter calls, autograd, and
unsupported shapes keep using ComfyUI's pytorch attention automatically.

The backend is also registered as `"mtlattn"` via
`comfy.ldm.modules.attention.register_attention_function` for other tooling.

Requires an Apple Silicon Mac on macOS 26.2+ (any M-series GPU): ComfyUI
workloads are dense equal-length batches, and mtlattn only wins those on its
MPP fast path (fp16/bf16, head_dim 64/80/88/96/128/160/256) — on older macOS
the pack loads but routes nothing. Inference only (training falls back).

## Benchmark

```bash
python bench/bench_attention.py          # add --big for HunyuanVideo 544p
```

Times ComfyUI's pytorch attention vs mtlattn at real model shapes and
spot-checks both against a CPU fp32 reference on the last query rows (where
the >2³² corruption lands first).
