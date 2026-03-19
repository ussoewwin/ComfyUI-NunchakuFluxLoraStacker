# ComfyUI-NunchakuFluxLoraStacker

This repository provides **two independent custom nodes** for ComfyUI:

1. **FLUX LoRA Loader V2** (`FluxLoraMultiLoader_10`) - Dynamic multi-LoRA loading with combo box UI for Nunchaku FLUX models
2. **Fast Groups Bypasser V2** (`FastGroupsBypasserV2`) - Group-based node control utility (ported from [rgthree-comfy](https://github.com/rgthree/rgthree-comfy))

**Note:** Fast Groups Bypasser V2 is unrelated to LoRA functionality and is included as a workflow management utility.

---

## Features (V1 - Legacy Node)

- **Dynamic Slot Visibility**: LoRA widget count follows `lora_count`
- **Simple / Advanced Modes**: Toggle between single-strength and dual-strength inputs
- **Automatic Layout Sizing**: Node height expands or shrinks to match visible widgets
- **Nunchaku FLUX Ready**: Purpose-built for the Nunchaku FLUX checkpoint format

## Installation

1. Clone the repository inside your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker.git
   ```
2. Restart ComfyUI to load the node.

## Usage

### Basic Flow

1. Add the **Nunchaku FLUX LoRA Stack** node to your workflow.
2. Connect the Nunchaku FLUX base model to the `model` input.
3. Set `lora_count` to the number of LoRA slots you want active.
4. Choose `input_mode`:
   - **simple**: Use `lora_wt_X` for all-in-one strength control.
   - **advanced**: Use `model_str_X` and `clip_str_X` for separate strength control.
5. Pick the LoRA file in each active slot and configure the strengths.
6. Connect the output to the next node in your graph.

### Parameters

- **model**: Nunchaku FLUX base model.
- **input_mode**
  - `simple`: Display LoRA name and a single strength slider.
  - `advanced`: Display separate model and CLIP strength sliders.
- **lora_count**: Number of LoRA slots to use (1-10).
- **lora_name_X**: LoRA file for slot X.
- **lora_wt_X**: Overall LoRA strength in simple mode.
- **model_str_X** / **clip_str_X**: Individual strengths in advanced mode.

## Dynamic UI Behavior

- Toggle LoRA slots based on `lora_count`.
- Switch between strength widgets depending on `input_mode`.
- Resize node height to match the visible widget stack.
- Refresh the layout immediately when parameters change.

## Requirements

- ComfyUI (2024 builds or newer recommended)
- Nunchaku core package (`nunchaku`) installed separately in the environment
- LoRA files compatible with Nunchaku FLUX

---

## V2 Nodes (New in v1.12)

### Why V2?

V2 nodes were developed to support **ComfyUI Nodes 2.0 (Desktop version)**. The new architecture required significant changes to widget management and input handling that are incompatible with V1.

### Why Keep V1?

V1 nodes (`NunchakuFluxLoraStack`) remain available for:
- **Backward Compatibility**: Users on ComfyUI 1.x can continue using existing workflows
- **Feature Preservation**: V1's `input_mode` (simple/advanced) is still useful for some workflows
- **Gradual Migration**: Users can transition to V2 at their own pace without breaking existing projects

### V2 Node Overview

This repository now includes two major V2 nodes with enhanced functionality:

### 1. FLUX LoRA Loader V2 (`FluxLoraMultiLoader_10`)

#### Features
- **Single Dynamic Node**: One node with adjustable slot count (1-10)
- **Combo Box Selector**: Select visible LoRA count (1-10) dynamically via dropdown
- **Auto Height Adjustment**: Node resizes automatically to fit visible slots
- **No Validation Errors**: All LoRA inputs are optional; hidden slots don't cause errors
- **Workflow Persistence**: Settings are saved and restored correctly

#### Usage
1. Add **FLUX LoRA Loader V2** node to your workflow
2. Use the **"🔢 LoRA Count"** dropdown to select how many slots you want visible
3. Configure LoRA files and strengths for visible slots only
4. Hidden slots are physically removed from UI (no padding waste)

#### Parameters
- `model`: Nunchaku FLUX base model (required)
- `🔢 LoRA Count`: Dropdown to select slot count (1-10)
- `lora_name_X`: LoRA filename (optional)
- `lora_wt_X`: LoRA strength, default 1.0 (optional)

### 2. Fast Groups Bypasser V2 (`FastGroupsBypasserV2`)

**Note:** This node is a port from the original [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) implementation and is unrelated to LoRA loading functionality. It is included here as a utility feature for workflow management.

#### Features
- **Group Filtering**: Match by color codes or regex title patterns
- **Toggle Control**: Enable/disable entire node groups with checkboxes
- **Sorting Options**: Position, alphanumeric, or custom alphabet
- **Bypass/Mute Modes**: Choose effect mode
- **Restriction Modes**: Default, max one, or always one group active

#### Usage
1. Add **Fast Groups Bypasser V2** node
2. Configure filters via properties or right-click menu
3. Toggle groups using generated checkbox widgets

---

## Release History

- v1.15 – FastGroupsBypasserV2 Fix: Fixed critical widget update bug where second property change required F5 refresh ([Release Notes](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.15))
- [v1.14](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.14) – Node Simplification: Removed test nodes (x1-x9), keeping only FLUX LoRA Loader V2 (x10) as the production node
- v1.13 – Clean Release: Removed all backup files from repository, updated FluxLoraMultiLoader_10 display name to "FLUX LoRA Loader V2"
- v1.12 – V2 Nodes Release: FLUX LoRA Loader V2 with dynamic combo box UI and Fast Groups Bypasser V2
- v1.11 – Corrected the README clone command to use the canonical repository URL. ([Issue #3](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/issues/3))
- [v1.10](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.10) – LoRA Loader Fix - Complete Version

## License

Apache-2.0

## Credits

- Dynamic UI implementation based on [efficiency-nodes-comfyui](https://github.com/jags111/efficiency-nodes-comfyui)
- Fast Groups Bypasser V2 ported from [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)
