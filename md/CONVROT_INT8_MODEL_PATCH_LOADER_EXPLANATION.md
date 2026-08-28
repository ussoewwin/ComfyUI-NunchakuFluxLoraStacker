# ConvRot INT8 Support for Model Patch Loader — Complete Explanation

> **Date:** 2026-08-28
> **Commit:** `ac4097f` — `feat: ConvRot INT8 support for ZImage ControlNet in Model Patch Loader`
> **Scope:** `nodes/misc_v2.py`, `README.md`

---

## 1. Purpose

The **Model Patch Loader** (`ModelPatchLoaderCustom`) previously could only load
float-precision model patches (BF16/FP16/FP32). When a user selected the
ConvRot INT8-quantized Z-Image ControlNet checkpoint
(`Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps_convrot_int8.safetensors`),
the node crashed immediately with:

```
RuntimeError: Only Tensors of floating point and complex dtype can require gradients
```

The goal of this change is to make the Model Patch Loader load **ConvRot INT8
ControlNet checkpoints natively**, so that:

- Weights stay **INT8 in memory (VRAM or system RAM)** — roughly half the
  footprint of the BF16 original (963 MB vs 1923 MB on disk for this model).
- Forward passes run through the **comfy-kitchen `int8_linear` kernel** with
  **online ConvRot (Hadamard) activation rotation**, the same optimized path
  ComfyUI uses for its own INT8 UNet/CLIP loading.
- The loader stays **100% backward compatible** with all existing BF16/FP8
  patches — no workflow or user action changes are required.

---

## 2. What Was Required to Achieve It

### 2.1 Root cause analysis

The old code did:

```python
dtype = comfy.utils.weight_dtype(sd)   # returns torch.int8 for INT8 checkpoints!
...
model = ZImage_Control(dtype=dtype, operations=comfy.ops.manual_cast, ...)
```

`weight_dtype()` computes the **dominant dtype by tensor element count**. In an
INT8 checkpoint the INT8 weight tensors dominate, so the loader received
`dtype=torch.int8` and tried to construct the module graph (RMSNorm weights,
Linear biases, ...) with that dtype. `torch.nn.RMSNorm` then calls
`torch.empty(..., dtype=torch.int8)` and wraps it in a `Parameter(...,
requires_grad=...)` — which is illegal for integer dtypes. Hence the crash
before any state dict was ever touched.

A second, subtler problem: the old code unconditionally ran `z_image_convert(sd)`,
a key-renaming helper for the **diffusers-style** Z-Image layout
(`to_q`/`to_k`/`to_v` -> fused `qkv`, `norm_q` -> `q_norm`, ...). The INT8
checkpoint is already distributed in the **comfy-native layout**
(`attention.qkv.*`, `attention.out.*`), so the conversion is a no-op for it —
but running it anyway risks corrupting quantized key grouping and must be
skipped.

### 2.2 The reference implementation

The sibling repository
`ComfyUI-HSWQ-Loader-and-Tools`
(`D:\USERFILES\GitHub\ComfyUI-HSWQ-Loader-and-Tools`)
had already solved both problems for its standalone ControlNet loader
(`nodes/hswq_load_convrot_int8_controlnet.py`). Its technique:

1. **Detect** INT8 checkpoints by scanning the state dict for
   `<layer>.comfy_quant` byte-encoded JSON entries with
   `"format": "int8_tensorwise"`.
2. **Force the graph dtype to BF16** so the module graph can be constructed.
3. **Swap in a quant-aware ops class** —
   `comfy.ops.mixed_precision_ops({"int8_tensorwise": QUANT_ALGOS["int8_tensorwise"]}, torch.bfloat16)`.
   Its `Linear._load_from_state_dict` consumes `<layer>.comfy_quant` +
   `<layer>.weight_scale` and attaches an INT8 `QuantizedTensor`
   (`TensorWiseINT8Layout`, with `convrot=True` and `convrot_groupsize`)
   to each Linear instead of a plain weight.

### 2.3 Feasibility verification performed before implementation

- **Checkpoint inspection** (custom safetensors header parser): 157 keys,
  38 quantized layers, every `comfy_quant` = `int8_tensorwise` +
  `convrot: true`; groupsize histogram `{256: 37, 4: 1}` (the `4` is
  `control_all_x_embedder.2-1`).
- **Kernel availability**: confirmed comfy-kitchen's `eager` backend implements
  `int8_linear` for CPU tensors (so `cpu_offload=True` works), and the CUDA
  backend covers GPU execution (RTX 5060 Ti, SM >= 7.5 required for INT8
  tensor cores).
- **Quant path mechanics**: read ComfyUI's `comfy/ops.py`
  (`_load_quantized_module`, `mixed_precision_ops`) to confirm the
  `comfy_quant`/`weight_scale` consumption contract and that
  `cast_bias_weight` dequantizes gracefully whenever a layer is moved to CPU
  or a LoRA is attached on the fly (fail-safe fallbacks, no hard crashes).

---

## 3. Files Added / Modified

| File | Change |
|---|---|
| `nodes/misc_v2.py` | **Modified** — INT8 detection helpers, quant-aware ops factory, and the new branch in `ModelPatchLoaderCustom.load_model_patch` |
| `README.md` | **Modified** — documented ConvRot INT8 support in the node list, the Features section, and Supported Model Types |

No new files were added; no ComfyUI core files were touched.

---

## 4. Full Code Changes

### 4.1 `nodes/misc_v2.py`

#### (a) New imports

```python
import json

import torch
from torch import nn
import folder_paths
import comfy.utils
import comfy.ops
import comfy.model_management
import comfy.ldm.common_dit
import comfy.latent_formats
import comfy.ldm.lumina.controlnet
from comfy.quant_ops import QUANT_ALGOS
```

*(added: `import json`, `from comfy.quant_ops import QUANT_ALGOS`)*

#### (b) New helper functions (inserted before `z_image_convert`)

```python
def _decode_comfy_quant(raw) -> dict:
    try:
        return json.loads(raw.numpy().tobytes())
    except Exception:  # noqa: BLE001
        return {}


def _has_int8_comfy_quant(sd) -> bool:
    """True if the checkpoint carries >=1 int8_tensorwise comfy_quant layer (ConvRot INT8)."""
    for key in sd.keys():
        if not key.endswith(".comfy_quant"):
            continue
        conf = _decode_comfy_quant(sd[key])
        if conf.get("format") == "int8_tensorwise":
            return True
    return False


def _int8_mixed_precision_ops():
    """MixedPrecisionOps supporting int8_tensorwise (ConvRot included).

    Same approach as the HSWQ ControlNet loader: build the module graph in a
    float dtype (BF16) and let MixedPrecisionOps.Linear._load_from_state_dict
    consume "<layer>.comfy_quant" / "<layer>.weight_scale", attaching an INT8
    QuantizedTensor (TensorWiseINT8Layout) to every quantized Linear.
    """
    quant_config = {
        "int8_tensorwise": QUANT_ALGOS["int8_tensorwise"],
    }
    return comfy.ops.mixed_precision_ops(
        quant_config,
        torch.bfloat16,
        full_precision_mm=False,
        disabled=[],
    )
```

#### (c) Modified branch in `ModelPatchLoaderCustom.load_model_patch`

```python
        elif 'control_all_x_embedder.2-1.weight' in sd: # alipai z image fun controlnet
            int8_checkpoint = _has_int8_comfy_quant(sd)
            if int8_checkpoint:
                # ConvRot INT8 checkpoint: keys are already in the comfy-native
                # layout (attention.qkv / attention.out), so z_image_convert is
                # a no-op and can break quantized key grouping. Build the module
                # graph in BF16; MixedPrecisionOps attaches INT8 QuantizedTensor
                # weights (TensorWiseINT8Layout, ConvRot online rotation) during
                # load_state_dict, so weights stay INT8 in memory.
                dtype = torch.bfloat16
                operations = _int8_mixed_precision_ops()
                logging.info("[ModelPatchLoaderCustom] INT8 ComfyQuant detected in '%s': loading with MixedPrecisionOps (weights stay INT8 / ConvRot)", name)
            else:
                operations = comfy.ops.manual_cast
            if not int8_checkpoint:
                sd = z_image_convert(sd)
            config = {}
            # ... (unchanged: n_control_layers detection, x_embedder shape
            #      inference, filtered_sd size-mismatch filtering) ...
```

#### (d) The model construction line (previously hardcoded)

```python
            model = comfy.ldm.lumina.controlnet.ZImage_Control(device=model_device, dtype=dtype, operations=operations, **config)
```

### 4.2 `README.md`

- Node list entry now reads: *"...with CPU offload support **and ConvRot INT8
  support**"*.
- Features section gained the bullet: **ConvRot INT8 Support** (automatic
  detection via `comfy_quant` metadata, `mixed_precision_ops` loading, INT8
  weights in memory, comfy-kitchen `int8_linear` kernel with online ConvRot
  activation rotation, works with GPU loading and CPU offload).
- Supported Model Types: `ZImage_Control` now says *"Z-Image format ControlNet
  **(BF16 and ConvRot INT8 variants)**"*.

---

## 5. What It Means

### 5.1 How the loading flow differs

| Stage | BF16 checkpoint (unchanged) | ConvRot INT8 checkpoint (new) |
|---|---|---|
| dtype detection | `weight_dtype(sd)` -> BF16 | **forced to `torch.bfloat16`** (graph construction dtype only) |
| ops class | `comfy.ops.manual_cast` | `mixed_precision_ops({int8_tensorwise}, bf16)` |
| key conversion | `z_image_convert(sd)` (diffusers -> comfy layout) | **skipped** — keys are already comfy-native |
| state dict load | plain weights assigned by `load_state_dict` | `_load_quantized_module` pops `comfy_quant` + `weight_scale`, builds a `QuantizedTensor` (`TensorWiseINT8Layout`, `convrot=True`, groupsize 4/256) per Linear |
| memory | BF16 weights | **INT8 weights** (~50% smaller); scales are tiny FP32 tensors |
| forward | `F.linear` with manual casting | comfy-kitchen `int8_linear` kernel; online Hadamard rotation of activations matches the rotated weights (`convrot_groupsize`) |
| CPU offload | native | works — `eager` backend implements `int8_linear` on CPU; `cast_bias_weight` dequantizes safely if a layer must compute on CPU |

### 5.2 Why this design is safe

- **Detection is metadata-driven, not name-driven.** A file qualifies as INT8
  only if it literally contains `comfy_quant` entries of format
  `int8_tensorwise`. Misnamed or partially-quantized files cannot trigger the
  path accidentally.
- **BF16 path is byte-for-byte the old path.** The only behavioral change for
  non-INT8 files is that `operations` now goes through a variable that holds
  exactly `comfy.ops.manual_cast` as before. Verified by a regression test.
- **No core modification.** Everything lives in the custom node; the approach
  uses only public ComfyUI APIs (`mixed_precision_ops`, `QUANT_ALGOS`,
  `weight_dtype`, `ZImage_Control`).
- **Graceful degradation.** If a quantized layer is LoRA-patched or moved to a
  device where the INT8 kernel can't run, ComfyUI's own cast logic dequantizes
  that layer to BF16 for the computation instead of failing.

### 5.3 Verification results (RTX 5060 Ti, 16 GB, ComfyUI portable)

1. **INT8 load test** — the target checkpoint loads with
   38/38 Linear layers attached as `QuantizedTensor` weights
   (`torch.int8` storage, `convrot=True`, groupsize 4/256); log line
   `[ModelPatchLoaderCustom] INT8 ComfyQuant detected ...` confirms the new
   branch.
2. **Forward pass (CPU placement, i.e. `cpu_offload=True`)** —
   `forward`, `forward_control_block(0)` and
   `forward_noise_refiner_block(0)` all return correctly shaped
   (`[1, 64, 3840]`) finite outputs.
3. **BF16 regression test** — the non-quantized Union checkpoint still loads
   via `manual_cast` + `z_image_convert` with **0** quantized weights and
   passes the same forward checks.
4. **Syntax check** — `py_compile` clean; deployed copy in
   `ComfyUI/custom_nodes` verified identical (hash match) to the repo copy.

### 5.4 Practical impact for users

- The **963 MB** ConvRot INT8 ControlNet now loads and runs through the Model
  Patch Loader with **roughly half the memory footprint** of the BF16 version
  (1923 MB), with the same workflow (load patch -> connect to
  `ZImageFunControlnet` apply node).
- On `cpu_offload=True`, system RAM usage is halved as well, and the compute
  stays on the optimized INT8 path whenever the layer resides on GPU.
- Users of `HSWQ ControlNet Loader` get identical INT8 semantics, but through
  the patch-based workflow (VAE-encoded control images, strength, mask,
  inpaint support) instead of the `CONTROL_NET` object workflow.
