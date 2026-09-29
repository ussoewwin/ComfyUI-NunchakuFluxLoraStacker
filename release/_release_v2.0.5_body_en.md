<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/blob/main/zhmd/v2.0.5.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# v2.0.5 — Model Patch Loader Removed (Now Maintained in ComfyUI-HSWQ-Loader-and-Tools)

This release removes the **Model Patch Loader** (`ModelPatchLoaderCustom`) from this pack. The node has been fully ported — as an identical node, `HSWQModelPatchLoaderCustom` — to the sibling repository **[ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)**, where it continues to be maintained (consolidating the node's development in a single repository). Nothing else in this pack is affected — in particular, `FastGroupsBypasserV2`, which shared the same module file, is completely unchanged.

---

## 1. Summary of Changes

| Area | v2.0.4 and earlier | v2.0.5 |
|---|---|---|
| **Model Patch Loader** | Shipped in this pack (`ModelPatchLoaderCustom`, display name "Model Patch Loader") | **Removed** — fully ported to ComfyUI-HSWQ-Loader-and-Tools as `HSWQModelPatchLoaderCustom` |
| **CPU-offload runtime patch** | Installed globally at import from `nodes/misc_v2.py` (wraps the stock `ZImageControlPatch` apply step) | Removed from this pack — the equivalent patch (`patches/model_patch_cpu_offload.py`) ships inside ComfyUI-HSWQ-Loader-and-Tools |
| **FastGroupsBypasserV2** | Shipped from `nodes/misc_v2.py` | **Unchanged** — kept byte-identical during the removal |
| **READMEs (EN / 中文)** | Documented the Model Patch Loader node | Node entry + section removed; node list and sections renumbered |
| **Version metadata** | `2.0.4` | `2.0.5` (`pyproject.toml` / `__init__.py`), published to the Comfy Registry |

---

## 2. Removal Details — `ModelPatchLoaderCustom`

The removed node was a loader for model patches in `models/model_patches`, with three automatically-detected architectures:

- **Qwen Image block-wise ControlNet** — discriminator key `controlnet_blocks.0.y_rms.weight`
- **SigLIP multi-feature projector** — discriminator key `feature_embedder.mid_layer_norm.bias`
- **Z-Image Fun ControlNet** — discriminator key `control_all_x_embedder.2-1.weight` (Union 2.0 / 2.1, with dynamic control-layer count detection)

and these features:

- **CPU offload** (`cpu_offload`, default `True`): the patch graph and its `ModelPatcher` are built in CPU main memory (`load_device` / `offload_device` = CPU), and a runtime apply patch kept the stock `ZImageControlPatch` apply step running on CPU for CPU-loaded patches. End-to-end CPU execution applies to the Z-Image Fun ControlNet path; the Qwen block-wise and SigLIP projector apply nodes were not patched.
- **ConvRot INT8 support**: checkpoints carrying `int8_tensorwise` `comfy_quant` layers are detected automatically and built with `mixed_precision_ops`, so weights stay **INT8 in memory** (`TensorWiseINT8Layout`) and forward passes run on the comfy-kitchen `int8_linear` kernel with online ConvRot rotation; non-quantized checkpoints keep the standard `manual_cast` path.
- Output: a standard **`MODEL_PATCH`** for the stock apply nodes (`QwenImageDiffsynthControlnet` / `ZImageFunControlnet` / `USOStyleReference`).

In this release, the node class, its helper modules (`BlockWiseControlBlock`, `QwenImageBlockWiseControlNet`, `SigLIPMultiFeatProjModel`), the INT8 detection/ops helpers, the CPU-offload apply patch and the node registration were all removed from `nodes/misc_v2.py`.

---

## 3. Where It Lives Now

The node continues to be developed as part of **[ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)** (available since v3.5.0):

| | |
|---|---|
| **Node ID** | `HSWQModelPatchLoaderCustom` |
| **Display name** | HSWQ Model Patch Loader (ConvRot INT8 / CPU offload) |
| **Inputs / Outputs** | `name` (model patch from `models/model_patches`), `cpu_offload` (BOOLEAN, default `True`) → `MODEL_PATCH` |
| **Apply with** | Stock apply nodes: `QwenImageDiffsynthControlnet` / `ZImageFunControlnet` / `USOStyleReference` |

The port is feature-identical: same architecture dispatch, same `cpu_offload` behaviour, and the same ConvRot INT8 `comfy_quant` handling.

---

## 4. Migration Guide

### For users of the previous node

1. Install or update **[ComfyUI-HSWQ-Loader-and-Tools](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)** in `custom_nodes` (ComfyUI-Manager, or a `git clone`).
2. In your workflows, replace the previous **Model Patch Loader** node with **HSWQ Model Patch Loader (ConvRot INT8 / CPU offload)**.
3. Re-select your patch file in the dropdown — files still come from the standard `models/model_patches` folder, and the `MODEL_PATCH` output connects to the same apply nodes as before.
4. No other changes are required — existing model patch files on disk are unaffected.

### For users who never used the node

No action is needed; nothing else in this pack changed.

---

## 5. File Modification Breakdown

| File | Type | Description |
|---|---|---|
| `nodes/misc_v2.py` | Update | `ModelPatchLoaderCustom` + helper classes + INT8 helpers + CPU-offload apply patch + registration removed; `FastGroupsBypasserV2` kept unchanged |
| `README.md` / `zhmd/README.md` | Update | Model Patch Loader removed from the node list and node sections; remaining entries renumbered (EN + 中文) |
| `md/CHANGELOG.md` / `zhmd/CHANGELOG.md` | Update | v2.0.5 release history entry with release-notes links |
| `pyproject.toml` / `__init__.py` | Update | Package version bumped to 2.0.5 |
| `release/_release_v2.0.5_body_en.md` | New | This release note (English release body) |
| `zhmd/v2.0.5.md` | New | Chinese release note (this document's counterpart) |

---

## 6. Verification & Unaffected Nodes

- The pack still loads normally; the only node removed is the Model Patch Loader. All other nodes — FLUX LoRA Loader V2, LoRA Stacker V2 / V3, SDNQ LoRA Stacker V2, Universal LoRA Analyzer, Color Filter, Florence-2, ControlAltAI, CCSR (TensorRT), Nunchaku Resolution Selector and Fast Groups Bypasser V2 — are untouched.
- `FastGroupsBypasserV2` (same module file) was verified byte-identical after the removal.
- The removal was validated by compiling and executing the modified module, confirming the node mapping contains exactly `FastGroupsBypasserV2`.
- The v2.0.5 version bump passed this repository's Comfy Registry publish workflow.
