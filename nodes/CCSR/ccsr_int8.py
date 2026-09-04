# -*- coding: utf-8 -*-
"""CCSR INT8 (HSWQ ConvRot) support.

The fp16 CCSR nodes build the model from ccsr_stage2.yaml. The UNet/ControlNet
Conv/Linear modules create their layers through `comfy.ops.manual_cast`
(imported at module load into a module-level `ops` name). For an INT8
checkpoint (comfy_quant markers), we swap those module-level `ops` objects to
comfy.ops.mixed_precision_ops BEFORE constructing the model. Layers carrying
comfy_quant become QuantizedTensor (kept int8 on GPU => VRAM savings); layers
without markers load as plain fp16 Parameters. This mirrors SeedVR2's
_dit_comfy_quant_ops approach.
"""
from __future__ import annotations

import os
import sys
import torch

# module paths that set `ops = comfy.ops.manual_cast` at import time
_OPS_MODULE_PATHS = [
    "ldm.modules.attention",
    "ldm.modules.diffusionmodules.model",
    "ldm.modules.diffusionmodules.util",
]

_INT8_PREFIXES = ("model.diffusion_model.", "control_model.")


def checkpoint_is_hswq_int8(checkpoint_path) -> bool:
    """True if the safetensors has comfy_quant markers (int8_tensorwise)."""
    if not checkpoint_path:
        return False
    path = str(checkpoint_path)
    if not (path.endswith(".safetensors") or path.endswith(".sft")):
        return False
    if not os.path.isfile(path):
        return False
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith(".comfy_quant"):
                    return True
    except Exception:
        return False
    return False


def _resolve_ops_modules():
    """Return module objects whose `ops` attr should be swapped.

    Works whether CCSR is imported top-level (ldm.*) or as a package
    (nodes.CCSR.ldm.*)."""
    found = []
    for mp in _OPS_MODULE_PATHS:
        # direct top-level
        mod = sys.modules.get(mp)
        if mod is None:
            try:
                import importlib
                mod = importlib.import_module(mp)
            except Exception:
                mod = None
        if mod is not None and hasattr(mod, "ops"):
            found.append(mod)
            continue
        # try under nodes.CCSR
        pkg = sys.modules.get("nodes.CCSR")
        if pkg is not None:
            try:
                import importlib
                mod = importlib.import_module("nodes.CCSR." + mp)
                if mod is not None and hasattr(mod, "ops"):
                    found.append(mod)
            except Exception:
                pass
    # de-dup by file
    uniq = []
    seen = set()
    for mod in found:
        f = getattr(mod, "__file__", None)
        if f not in seen:
            seen.add(f)
            uniq.append(mod)
    return uniq


def get_mixed_ops(compute_dtype: torch.dtype = torch.float16):
    """Build comfy.ops.mixed_precision_ops extended with a quantized Conv2d.

    stock mixed_precision_ops only quantizes Linear; CCSR packs also include
    Conv2d (plain int8), so we subclass manual_cast.Conv2d and route
    _load_from_state_dict through comfy.ops._load_quantized_module.
    """
    import comfy.ops as comfy_ops

    mixed = comfy_ops.mixed_precision_ops(
        quant_config={},
        compute_dtype=compute_dtype,
        full_precision_mm=False,
        disabled=[],
    )

    class QuantConv2d(comfy_ops.manual_cast.Conv2d):
        """Conv2d that consumes comfy_quant/weight_scale on load_state_dict."""

        comfy_cast_weights = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.factory_kwargs = {"device": None, "dtype": compute_dtype}
            self._orig_shape = tuple(self.weight.shape) if self.weight is not None else None
            self.quant_format = None
            self.layout_type = None
            self._disabled_formats = []
            self._full_precision_mm = False
            self._full_precision_mm_config = False

        def _load_from_state_dict(self, *args):
            comfy_ops._load_quantized_module(
                self, super()._load_from_state_dict, *args, load_extra_params=True
            )

        def state_dict(self, *args, destination=None, prefix="", **kwargs):
            sd = destination if destination is not None else {}
            return comfy_ops._quantized_weight_state_dict(self, sd, prefix)

    mixed.Conv2d = QuantConv2d
    return mixed


class _ops_swap:
    """Context manager: swap module-level `ops` for the duration of model build."""

    def __init__(self, new_ops):
        self.new_ops = new_ops
        self.saved = {}

    def __enter__(self):
        for mod in _resolve_ops_modules():
            self.saved[id(mod)] = (mod, getattr(mod, "ops", None))
            mod.ops = self.new_ops
        return self

    def __exit__(self, *exc):
        for mod, old_ops in self.saved.values():
            if old_ops is not None:
                mod.ops = old_ops
        self.saved.clear()


def prepare_state_for_comfy_ops(state: dict) -> dict:
    """comfy.ops parses comfy_quant via .numpy() -> needs CPU tensors."""
    for key, value in list(state.items()):
        if not key.endswith("comfy_quant"):
            continue
        if torch.is_tensor(value) and value.device.type != "cpu":
            state[key] = value.cpu()
    return state
