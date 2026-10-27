"""ComfyUI attention backend for Apple Silicon, backed by mtlattn (fused
Metal flash attention) and torch's own MPS SDPA.

ComfyUI picks sub_quad for MPS by default. On torch >= 2.14, native SDPA has a
fast prefill kernel (fp16/bf16, head_dim 64/96/128/256, macOS 26.2+) that
beats both sub_quad and mtlattn, so those calls go to ComfyUI's pytorch
attention. mtlattn takes the head dims native SDPA's kernel doesn't cover
(80/88/160) and training calls (its fused backward); everything else keeps
ComfyUI's selected backend.

ComfyUI's dense attention layout [B, L, H*D] is byte-identical to mtlattn's
varlen layout [total_tokens, H, D] for an equal-length batch, so the mtlattn
path adds no copies: build trivial cu_seqlens, run the kernel, view the result
back.
"""

import functools
import logging
from importlib.metadata import version

import torch

import mtlattn
from mtlattn import _C

import comfy.ldm.modules.attention

log = logging.getLogger("ComfyUI-mtlattn")

DEFAULT_MIN_SEQLEN = 1024

# head_dims torch's MPS SDPA prefill kernel covers -> ComfyUI's pytorch attention.
_NATIVE_HEAD_DIMS = frozenset((64, 96, 128, 256))
# head_dims only mtlattn's MPP (Metal 4 matmul2d) kernel covers -> mtlattn.
_MTLATTN_HEAD_DIMS = frozenset((80, 88, 160))

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
_announced = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def attention_mtlattn(q, k, v, heads, mask=None, attn_precision=None,
                      skip_reshape=False, skip_output_reshape=False,
                      min_seqlen=DEFAULT_MIN_SEQLEN, **kwargs):
    # The guard flag keeps the wrapped backends from re-entering the
    # optimized_attention_override that got us here.
    kwargs.setdefault("_inside_attn_wrapper", True)
    call = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape, **kwargs)

    def native():                          # ComfyUI's selected backend (sub_quad on MPS)
        return comfy.ldm.modules.attention.optimized_attention(q, k, v, heads, **call)

    def pytorch():
        return comfy.ldm.modules.attention.attention_pytorch(q, k, v, heads, **call)

    if q.device.type != "mps" or not mpp_available() \
            or q.dtype not in (torch.float16, torch.bfloat16):
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

    training = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad
                                            or v.requires_grad)
    kernel = mask is None and hkv > 0 and hq % hkv == 0 and (
        training or (dim_head in _MTLATTN_HEAD_DIMS and nq >= min_seqlen))
    if not kernel:
        return pytorch() if dim_head in _NATIVE_HEAD_DIMS else native()

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
        out = mtlattn.varlen_attention(q3, k3, v3, cu_q, cu_kv, nq,
                                       kwargs.get("scale", None))
    except Exception as e:
        _warn_once(("err", type(e).__name__, str(e)[:100]),
                   f"ComfyUI-mtlattn: kernel failed ({e}) — using ComfyUI "
                   "attention for this shape")
        return pytorch() if dim_head in _NATIVE_HEAD_DIMS else native()

    if training not in _announced:
        _announced.add(training)
        log.info("mtlattn %s active: B=%d heads=%d seq=%d head_dim=%d %s",
                 "training" if training else "attention", b, hq, nq, dim_head, q.dtype)

    if skip_output_reshape:                # [B, H, Nq, D]
        return out.view(b, nq, hq, dim_head).permute(0, 2, 1, 3)
    return out.reshape(b, nq, hq * dim_head)


def self_test():
    """Tiny end-to-end kernel call (bypassing mtlattn's native routing);
    catches torch-ABI mismatches at load time instead of mid-sampling."""
    major, minor = (int(p) for p in version("mtlattn").split(".")[:2])
    if (major, minor) < (0, 5):
        raise RuntimeError(f"needs mtlattn >= 0.5 (found {version('mtlattn')})")
    q = torch.randn(64, 2, 64, dtype=torch.float16, device="mps")
    cu = torch.tensor([0, 64], dtype=torch.int32, device="mps")
    out = _C.varlen_attention(q, q, q, cu, cu, 64, 0.125, False, 0, None, None)
    torch.mps.synchronize()
    assert out.shape == q.shape and torch.isfinite(out).all()


class MtlattnAttentionPatch:
    """Route this model's attention to the fastest MPS backend: native SDPA
    where torch's prefill kernel applies, mtlattn for head_dim 80/88/160 and
    training, ComfyUI's default otherwise."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "min_seqlen": ("INT", {
                "default": DEFAULT_MIN_SEQLEN, "min": 0, "max": 1 << 20,
                "tooltip": "head_dim 80/88/160 attention shorter than this "
                           "keeps ComfyUI's default backend. 0 routes every "
                           "such call through mtlattn."}),
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
