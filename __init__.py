"""ComfyUI-mtlattn: fused Metal flash attention for ComfyUI on Apple Silicon.

Registers an "mtlattn" attention backend and an "Apply mtlattn Attention"
model-patch node. If mtlattn is missing or was built against a different
torch, the pack loads with no nodes and a warning instead of breaking startup.
"""

import logging

log = logging.getLogger("ComfyUI-mtlattn")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available (Apple Silicon Mac required)")

    from .mtlattn_attention import (MtlattnAttentionPatch, attention_mtlattn,
                                    mpp_available, self_test)

    self_test()

    import comfy.ldm.modules.attention
    comfy.ldm.modules.attention.register_attention_function(
        "mtlattn", attention_mtlattn)

    NODE_CLASS_MAPPINGS["MtlattnAttentionPatch"] = MtlattnAttentionPatch
    NODE_DISPLAY_NAME_MAPPINGS["MtlattnAttentionPatch"] = \
        "Apply mtlattn Attention"

    if mpp_available():
        log.info("ComfyUI-mtlattn loaded (MPP fast path available)")
    else:
        log.warning(
            "ComfyUI-mtlattn loaded, but the MPP fast path is unavailable "
            "(needs macOS 26.2+). On dense ComfyUI workloads the portable "
            "kernel does not beat native attention, so all calls will fall "
            "back to ComfyUI's default backend.")
except Exception as e:
    log.warning(
        "ComfyUI-mtlattn disabled: %s. Install a matching build with "
        "`pip install mtlattn` (or build from source against your torch: "
        "https://github.com/lastowl/mtlattn#install); ComfyUI runs normally "
        "without it.", e)
