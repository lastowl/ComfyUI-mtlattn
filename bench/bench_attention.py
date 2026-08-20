"""Benchmark ComfyUI's pytorch attention vs the mtlattn backend at real
model shapes (image / video / 3D generation on Mac).

Run from anywhere with the ComfyUI venv python:

    python bench/bench_attention.py [--big] [--comfyui /path/to/ComfyUI]

Times both backends through the ComfyUI attention API ([B, L, H*D] layout,
reshape overheads included) and spot-checks correctness of the LAST 64 query
rows against a CPU fp32 reference — the MPS SDPA >2^32-element corruption
(pytorch/pytorch#179352) hits later rows first, so this column catches it.
"""

import argparse
import os
import statistics
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--big", action="store_true",
                    help="include the slow HunyuanVideo 544p shape")
parser.add_argument("--comfyui", default=os.path.expanduser("~/ComfyUI"))
cli = parser.parse_args()
sys.argv = sys.argv[:1]          # keep comfy.cli_args from seeing our flags

sys.path.insert(0, cli.comfyui)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from comfy.ldm.modules.attention import (attention_pytorch,  # noqa: E402
                                         attention_sub_quad)
from mtlattn_attention import attention_mtlattn, mpp_available  # noqa: E402

# name, B, heads, head_dim, Nq, Nkv, dtype
SHAPES = [
    ("SD1.5 512p self d40",       2,  8,  40,  4096,  4096, torch.float16),
    ("SDXL 1024p self",           2, 10,  64,  4096,  4096, torch.float16),
    ("Flux 1024p joint",          1, 24, 128,  4608,  4608, torch.float16),
    ("Hunyuan3D-2 dit",           2, 16,  64,  3072,  3072, torch.float16),
    ("TRELLIS 3D sparse",         1, 16,  64, 20000, 20000, torch.float16),
    ("Wan2.1 1.3B 480p81f self",  1, 12, 128, 32760, 32760, torch.bfloat16),
    ("Wan2.1 1.3B 480p81f cross", 1, 12, 128, 32760,   512, torch.bfloat16),
    ("Wan2.1 14B 480p81f self",   1, 40, 128, 32760, 32760, torch.bfloat16),
]
if cli.big:
    SHAPES.append(
        ("HunyuanVideo 544p97f self", 1, 24, 128, 51000, 51000, torch.bfloat16))

CHECK_ROWS = 64


def to_cpu32(t):
    # Two steps on purpose: the fused .to("cpu", torch.float32) of an offset
    # MPS view silently returns wrong values on torch <= 2.13 (found while
    # writing this bench; see mtlattn's tests/test_mps_fused_to_bug.py).
    return t.to("cpu").to(torch.float32)


def ref_last_rows(q, k, v, b, heads, dim, nq, nkv):
    """CPU fp32 attention for the last CHECK_ROWS query rows of the last batch
    element. Inputs are ComfyUI-layout [B, N, H*D]. Exact for the sampled rows
    because softmax is per-row."""
    qr = to_cpu32(q)[-1, -CHECK_ROWS:].view(CHECK_ROWS, heads, dim)
    kr = to_cpu32(k)[-1].view(nkv, heads, dim)
    vr = to_cpu32(v)[-1].view(nkv, heads, dim)
    scores = torch.einsum("qhd,khd->hqk", qr, kr) * dim ** -0.5
    probs = scores.softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, vr).reshape(CHECK_ROWS, heads * dim)


def time_backend(fn, q, k, v, heads):
    for _ in range(2):                                   # warmup
        fn(q, k, v, heads)
    torch.mps.synchronize()
    times = []
    while len(times) < 10 and sum(times) < 1.5 and (
            not times or times[0] < 5 or len(times) < 2):
        t0 = time.perf_counter()
        out = fn(q, k, v, heads)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


BACKENDS = [
    ("sub_quad", attention_sub_quad),        # ComfyUI's default on MPS
    ("pytorch", attention_pytorch),          # native MPS SDPA
    ("mtlattn", lambda q, k, v, h: attention_mtlattn(q, k, v, h, min_seqlen=0)),
]


def fmt_t(x):
    return f"{x * 1e3:9.0f}" if isinstance(x, float) else f"{x:>9}"


def fmt_e(x):
    if not isinstance(x, float):
        return f"{'-':>8} "
    return f"{x:8.4f}" + ("!" if x > 0.1 else " ")


def run():
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, MPP path: {mpp_available()}")
    print("times in ms; err vs CPU fp32 on the last "
          f"{CHECK_ROWS} query rows, '!' = corrupt\n")
    header = (f"{'shape':<27}{'dtype':<5}{'elems':>7}  "
              + "".join(f"{n:>9}" for n, _ in BACKENDS)
              + f"{'speedup':>9}{'TF/s':>6}  "
              + "".join(f"e:{n:<7}" for n, _ in BACKENDS))
    print(header)
    print("-" * len(header))

    for name, b, heads, dim, nq, nkv, dtype in SHAPES:
        q = torch.randn(b, nq, heads * dim, dtype=dtype, device="mps")
        k = torch.randn(b, nkv, heads * dim, dtype=dtype, device="mps")
        v = torch.randn(b, nkv, heads * dim, dtype=dtype, device="mps")
        ref = ref_last_rows(q, k, v, b, heads, dim, nq, nkv)
        score_elems = b * heads * nq * nkv               # >2^32 -> MPS bug zone
        flops = 4 * b * heads * nq * nkv * dim

        row = {}
        for label, fn in BACKENDS:
            try:
                t, out = time_backend(fn, q, k, v, heads)
                err = (to_cpu32(out)[-1, -CHECK_ROWS:] - ref).abs().max().item()
                row[label] = (t, err)
            except Exception as e:
                row[label] = (None, type(e).__name__[:9])
            torch.mps.empty_cache()

        tm = row["mtlattn"][0]
        # speedup vs the fastest native backend that produced a CORRECT result
        native_ok = [t for lbl in ("sub_quad", "pytorch")
                     for t, e in [row[lbl]]
                     if isinstance(t, float) and isinstance(e, float) and e <= 0.1]
        if isinstance(tm, float) and native_ok:
            speed = f"{min(native_ok) / tm:8.2f}x"
        elif isinstance(tm, float):
            speed = f"{'(only ok)':>9}"
        else:
            speed = f"{'-':>9}"
        tflops = f"{flops / tm / 1e12:6.1f}" if isinstance(tm, float) else f"{'-':>6}"
        over = "*" if score_elems > 2 ** 32 else " "
        print(f"{name:<27}{str(dtype).split('.')[1][:4]:<5}"
              f"{score_elems / 2**32:6.1f}x{over}"
              + "".join(fmt_t(row[lbl][0]) for lbl, _ in BACKENDS)
              + f"{speed}{tflops}  "
              + "".join(fmt_e(row[lbl][1]) for lbl, _ in BACKENDS))

    print("\nelems = score-matrix elements B*H*Nq*Nkv as a multiple of 2^32;")
    print("* = past the MPS SDPA silent-corruption threshold (pytorch#179352).")
    print("speedup = mtlattn vs fastest native backend with a correct result.")
    print("err flags '!' when > 0.1 (fp16/bf16 rounding is ~0.01; O(1) = corrupt).")


if __name__ == "__main__":
    run()
