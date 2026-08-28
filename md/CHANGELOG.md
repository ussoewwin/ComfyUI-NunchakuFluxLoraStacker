# Changelog

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../zhmd/CHANGELOG.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

## Release History

- v2.0.0 — Model Patch Loader ConvRot INT8: **Model Patch Loader** (`ModelPatchLoaderCustom`) now loads comfy-native **ConvRot INT8** Z-Image ControlNet checkpoints (e.g. `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps_convrot_int8.safetensors`). INT8 is detected automatically via `comfy_quant` metadata, the module graph is built in BF16 and quantized weights are attached by `mixed_precision_ops`, so weights stay **INT8 in memory** and inference runs on the comfy-kitchen `int8_linear` kernel with online ConvRot activation rotation. Works with both GPU loading and CPU offload; BF16 checkpoints keep the existing path. Fixes the `Only Tensors of floating point and complex dtype can require gradients` crash. Technique ported from the [HSWQ ControlNet Loader](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools) ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v2.0.0))

- v1.39 – AMD / ROCm nunchaku import: Merged [PR #6](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/pull/6) import guards so the pack still loads when `nunchaku` fails to import; follow-up gates FLUX registration on `_NUNCHAKU_AVAILABLE`, leaves `standard` / `standard_v3` unwrapped, and raises a clear error instead of setting `compose_lora = None` ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.39))

- v1.38 – CCSR dependencies: Added CCSR packages (`pytorch-lightning` and related deps) to `requirements.txt`, and added `install.py` so ComfyUI-Manager installs them automatically—prevents the whole pack from failing to load when `pytorch_lightning` is missing ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.38))

- v1.37 – Nunchaku Resolution Selector: Added **Nunchaku Resolution Selector** (`NunchakuResolutionSelector`) under `ussoewwin/resolution`—pick width/height from Flux1-style aspect presets (or custom size), emit hires dimensions, an empty **16-channel** latent, and an info string. Preset list expanded from ControlAltAI Megapixel Calculator ratios (1.0 MP / 1.5 MP High). Documented in README and zhmd README with screenshot.

- v1.36 – LoRA Stacker V3: Added **LoRA Stacker V3** (`LoraStackerV3_10`) for standard SD models—the same 10-slot stack as V2, plus a **global `toggle_all` master switch** (when off, no LoRAs are applied and outputs pass through unchanged) and **per-slot `enabled_1` … `enabled_10` toggles** for quick A/B comparisons and partial stacks without rewiring. Implemented in `nodes/lora/standard_v3.py`; README and zhmd README document usage and the toggle truth table.

- v1.35 – CCSR: Integrated CCSR upscaler nodes (CCSR_Upscale, CCSR_Model_Select, DownloadAndLoadCCSRModel) under `nodes/CCSR/` (upstream https://github.com/kijai/ComfyUI-CCSR based on csslc/CCSR Apache-2.0). Added Python 3.13 and latest ComfyUI compatibility, implemented dynamic package path resolution to support custom installation directory names, and added README documentation with node screenshots.

- v1.34 – Color Filter: Fixed an issue where custom exclusion terms failed to match due to stale precompiled bytecode caches (`__pycache__`), and implemented recursive comma-collapsing to properly handle formatting when multiple consecutive words are stripped out. ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.34))

- v1.33 – Color Filter: Improved user-defined `exclude_words` parsing to dynamically handle word boundaries (`\b`) based on whether words contain ASCII alphanumeric characters, preventing regex errors and missed matches for Japanese text and special characters. ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.33))

- v1.32 – ControlAltAI: Integrated ControlAltAI utility nodes under `nodes/controlaltai/` (upstream https://github.com/gseth/ControlAltAI-Nodes, MIT). **My Python 3.13–compatible fork** of those nodes is included in this pack (same node set), **to reduce my own separate-repo maintenance**. Pack registration in root `__init__.py`; README ControlAltAI section (after Florence-2); node details in [nodes/controlaltai/controlalttai.md](../nodes/controlaltai/controlalttai.md) only; frontend helper `js/integer_settings_advanced.js` for Integer Settings Advanced.

- v1.31 – Florence-2: Integrated Florence-2 VLM nodes under `nodes/florence2/` (Sage Attention 2/3 and Transformers 5.x loader path; upstream lineage [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) merged into this repo). README node overview (11 nodes), Florence-2 section, screenshot `png/Florence2.png`, Credits, and License note for MIT subtree `nodes/florence2/LICENSE`. ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.31))

- v1.30 – Color Filter: Added `ColorFilter` node (`nodes/color_filter/`) to remove monochrome / black-and-white wording from caption and tag strings (e.g. Florence-2, WD14 Tagger); README section and screenshot. ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.30))

- v1.29 – LoRA Loader V2 update: enabled negative LoRA strength values (min/max: -100.0 to 100.0) across FLUX LoRA Loader V2, LoRA Stacker V2, and SDNQ LoRA Stacker V2 to match standard ComfyUI behavior ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.29))

- v1.28 – Nunchaku Flux1 PulID: Released mitigations for errors affecting the upstream Nunchaku Flux1 PulID node; details are published in the release notes ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.28))

- v1.27 – Model Patch Loader: Fix for ComfyUI-update-induced bug (CPU offload / CoreModelPatcher) ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.27))

- v1.26 – Model Patch Loader: Fixed Z-Image ControlNet matmul shape error; infer control_in_dim from checkpoint and include checkpoint-only keys in load state_dict so embedder weights load under lazy init ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.26))

- [v1.25](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.25) – Universal LoRA Analyzer, Load Image node, SAM3 integration; README node docs and structure updates ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.25))

- [v1.24](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.24) – Fixed LoRA not working issue with ComfyUI-nunchaku 1.1.0: Addressed the problem where LoRAs were not being applied to the final image output after updating to ComfyUI-nunchaku 1.1.0. The fix ensures proper MODEL object cloning and state preservation.

- v1.21 – Z-Image ControlNet Union 2.1 Support: Added dynamic layer count detection for Z-Image ControlNet to support Union 2.1 models.

- v1.18 – SDNQ LoRA Stacker V2: Added dedicated SDNQ LoRA Stacker V2 node for SDNQ quantized models with dynamic 10-slot UI (for use with [comfyui-sdnq-splited](https://github.com/ussoewwin/comfyui-sdnq-splited)). Fixed Z-Image ControlNet loading to support Union 2.0 checkpoints with filtering for size mismatches

- v1.17 – Model Patch Loader: Added ModelPatchLoaderCustom node with CPU offload support for loading ControlNet and feature projector patches

- v1.16 – LoRA Stacker V2: Added universal LoRA loader for standard SD models (SDXL, Flux, WAN2.2) with dynamic 10-slot UI ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.16))

- v1.15 – FastGroupsBypasserV2 Fix: Fixed critical widget update bug where second property change required F5 refresh ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.15))

- v1.14 – Node Simplification: Removed test nodes (x1-x9), keeping only FLUX LoRA Loader V2 (x10) as the production node

- v1.13 – Clean Release: Removed all backup files from repository, updated FluxLoraMultiLoader_10 display name to "FLUX LoRA Loader V2"

- v1.12 – V2 Nodes Release: FLUX LoRA Loader V2 with dynamic combo box UI and Fast Groups Bypasser V2

- v1.11 – Corrected the README clone command to use the canonical repository URL. ([Issue #3](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/issues/3))

- [v1.10](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.10) – LoRA Loader Fix - Complete Version
