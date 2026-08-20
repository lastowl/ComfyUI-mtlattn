"""ComfyUI attention backend backed by mtlattn (fused Metal flash attention
for Apple Silicon).

ComfyUI's dense attention layout [B, L, H*D] is byte-identical to mtlattn's
varlen layout [total_tokens, H, D] for an equal-length batch, so the hot path
adds no copies: build trivial cu_seqlens, run the kernel, view the result
back. Anything the kernel can't take (small shapes, autograd, odd head_dim,
unmappable masks) falls through to ComfyUI's pytorch attention.
"""

import functools
import logging

import torch

import mtlattn
from mtlattn import _C

import comfy.ldm.modules.attention

log = logging.getLogger("ComfyUI-mtlattn")

DEFAULT_MIN_SEQLEN = 1024

# head_dims with a fast MPP (Metal 4 matmul2d) kernel; any dim <= 128 also has
# a portable simdgroup kernel used when MPP is unavailable (M1/M2, macOS < 26.2).
_MPP_HEAD_DIMS = frozenset((64, 80, 88, 96, 128, 160, 256))

_mpp = None


def mpp_available():
    global _mpp
    if _mpp is None:
        try:
            _mpp = bool(_C.mpp_available())
        except Exception:
            _mpp = False
    return _mpp


_warned = set()
_announced = False


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def _supported(dtype, dim_head, has_mask):
    # ComfyUI attention is always a dense equal-length batch, where the
    # portable simdgroup path loses to native backends — so only route to
    # mtlattn when the MPP (Metal 4 matmul2d) fast path will run: macOS 26.2+,
    # fp16/bf16, and a head_dim with an MPP kernel.
    return (mpp_available()
            and dtype in (torch.float16, torch.bfloat16)
            and dim_head in _MPP_HEAD_DIMS)


def attention_mtlattn(q, k, v, heads, mask=None, attn_precision=None,
                      skip_reshape=False, skip_output_reshape=False,
                      min_seqlen=DEFAULT_MIN_SEQLEN, **kwargs):
    def native():
        # Whatever backend ComfyUI selected for this machine (sub_quad on MPS
        # by default). The guard flag keeps its wrapper from re-entering the
        # optimized_attention_override that got us here.
        kwargs.setdefault("_inside_attn_wrapper", True)
        return comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads, mask=mask, attn_precision=attn_precision,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
            **kwargs)

    if q.device.type != "mps":
        return native()
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad
                                    or v.requires_grad):
        return native()

    b = q.shape[0]
    if skip_reshape:                       # q [B, H, Nq, D], kv may have fewer heads
        _, hq, nq, dim_head = q.shape
        hkv, nkv = k.shape[1], k.shape[2]
    else:                                  # q [B, Nq, H*D]
        nq, nkv = q.shape[1], k.shape[1]
        hq = heads
        dim_head = q.shape[-1] // heads
        hkv = k.shape[-1] // dim_head      # < heads iff enable_gqa

    if nq < min_seqlen:
        return native()
    if not _supported(q.dtype, dim_head, mask is not None):
        return native()
    if hkv == 0 or hq % hkv != 0:          # GQA needs divisible head counts
        return native()

    try:
        if skip_reshape:
            # [B, H, N, D] -> [B*N, H, D] (reshape copies, permuted strides)
            q3 = q.permute(0, 2, 1, 3).reshape(b * nq, hq, dim_head)
            k3 = k.permute(0, 2, 1, 3).reshape(b * nkv, hkv, dim_head)
            v3 = v.permute(0, 2, 1, 3).reshape(b * nkv, hkv, dim_head)
        else:
            # [B, N, H*D] -> [B*N, H, D]: free, same bytes
            q3 = q.reshape(b * nq, hq, dim_head)
            k3 = k.reshape(b * nkv, hkv, dim_head)
            v3 = v.reshape(b * nkv, hkv, dim_head)

        dev = q.device
        cu_q = torch.arange(0, (b + 1) * nq, nq, dtype=torch.int32, device=dev)
        cu_kv = torch.arange(0, (b + 1) * nkv, nkv, dtype=torch.int32,
                             device=dev)

        bias = None
        if mask is not None:
            m = mask.to(dev)
            if m.ndim == 3:                # [B?, Nq, Nkv] -> [B?, 1, Nq, Nkv]
                m = m.unsqueeze(1)
            # -> [B*Nq, H or 1, Nkv] fp32; raises NotImplementedError on
            # shapes it can't map, which lands in the fallback below.
            bias = mtlattn._sdpa_mask_to_bias(m, b, hq, nq, nkv)

        out = mtlattn.varlen_attention(q3, k3, v3, cu_q, cu_kv, nq,
                                       kwargs.get("scale", None),
                                       attn_bias=bias)
    except NotImplementedError as e:
        _warn_once(("mask", str(e)), f"ComfyUI-mtlattn: {e} — using pytorch "
                   "attention for this shape")
        return native()
    except Exception as e:
        _warn_once(("err", type(e).__name__, str(e)[:100]),
                   f"ComfyUI-mtlattn: kernel failed ({e}) — using pytorch "
                   "attention for this shape")
        return native()

    global _announced
    if not _announced:
        _announced = True
        log.info("mtlattn attention active (%s path): B=%d heads=%d seq=%d "
                 "head_dim=%d %s", "MPP" if mpp_available() else "simdgroup",
                 b, hq, nq, dim_head, q.dtype)

    if skip_output_reshape:                # [B, H, Nq, D]
        return out.view(b, nq, hq, dim_head).permute(0, 2, 1, 3)
    return out.reshape(b, nq, hq * dim_head)


def self_test():
    """Tiny end-to-end kernel call; catches torch-ABI mismatches at load time
    instead of mid-sampling."""
    q = torch.randn(64, 2, 64, dtype=torch.float16, device="mps")
    cu = torch.tensor([0, 64], dtype=torch.int32, device="mps")
    out = mtlattn.varlen_attention(q, q, q, cu, cu, 64)
    torch.mps.synchronize()
    assert out.shape == q.shape and torch.isfinite(out).all()


class MtlattnAttentionPatch:
    """Route this model's attention through mtlattn where it wins; everything
    else keeps using ComfyUI's pytorch attention."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "min_seqlen": ("INT", {
                "default": DEFAULT_MIN_SEQLEN, "min": 0, "max": 1 << 20,
                "tooltip": "Attention calls with query length below this use "
                           "native SDPA (it is competitive on small shapes). "
                           "0 routes everything supported through mtlattn."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch"

    def patch(self, model, enabled, min_seqlen):
        m = model.clone()
        if enabled:
            m.set_model_optimized_attention(
                functools.partial(attention_mtlattn, min_seqlen=min_seqlen))
        return (m,)
