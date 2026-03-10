# Changelog

## Release History

- v1.27 – Model Patch Loader: Fix for ComfyUI-update-induced bug (CPU offload / CoreModelPatcher) ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.27))
- v1.26 – Model Patch Loader: Fixed Z-Image ControlNet matmul shape error; infer control_in_dim from checkpoint and include checkpoint-only keys in load state_dict so embedder weights load under lazy init ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.26))
- [v1.25](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.25) – Universal LoRA Analyzer, Load Image node, SAM3 integration; README node docs and structure updates ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.25))
- [v1.24](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.24) – Fixed LoRA not working issue with ComfyUI-nunchaku 1.1.0: Addressed the problem where LoRAs were not being applied to the final image output after updating to ComfyUI-nunchaku 1.1.0. The fix ensures proper MODEL object cloning and state preservation.
- v1.21 – Z-Image ControlNet Union 2.1 Support: Added dynamic layer count detection for Z-Image ControlNet to support Union 2.1 models.
- v1.18 – SDNQ LoRA Stacker V2: Added dedicated SDNQ LoRA Stacker V2 node for SDNQ quantized models with dynamic 10-slot UI (for use with [comfyui-sdnq-splited](https://github.com/ussoewwin/comfyui-sdnq-splited)). Fixed Z-Image ControlNet loading to support Union 2.0 checkpoints with filtering for size mismatches
- v1.17 – Model Patch Loader: Added ModelPatchLoaderCustom node with CPU offload support for loading ControlNet and feature projector patches
- v1.16 – LoRA Stacker V2: Added universal LoRA loader for standard SD models (SDXL, Flux, WAN2.2) with dynamic 10-slot UI ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.16))
- v1.15 – FastGroupsBypasserV2 Fix: Fixed critical widget update bug where second property change required F5 refresh ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.15))
- [v1.14](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.14) – Node Simplification: Removed test nodes (x1-x9), keeping only FLUX LoRA Loader V2 (x10) as the production node
- v1.13 – Clean Release: Removed all backup files from repository, updated FluxLoraMultiLoader_10 display name to "FLUX LoRA Loader V2"
- v1.12 – V2 Nodes Release: FLUX LoRA Loader V2 with dynamic combo box UI and Fast Groups Bypasser V2
- v1.11 – Corrected the README clone command to use the canonical repository URL. ([Issue #3](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/issues/3))
- [v1.10](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.10) – LoRA Loader Fix - Complete Version
