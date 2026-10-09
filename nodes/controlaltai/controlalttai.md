# ControlAltAI-Nodes-fixed-Python3.13

<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../../zhmd/controlalttai.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

This is a Python 3.13 compatible version of the original ControlAltAI-Nodes.

## Overview

The original ControlAltAI-Nodes stopped working with Python 3.13, so this version was created with all node names changed. All node names have been modified to avoid conflicts with nodes that have similar names.

## Why Renaming Was Required on Python 3.13

Python 3.13 tightened several internals around module loading and cache validation. ComfyUI discovers custom nodes by letting each extension import its modules and append entries to two global dictionaries: `NODE_CLASS_MAPPINGS` (class name → implementation) and `NODE_DISPLAY_NAME_MAPPINGS` (class name → UI label). As soon as a class name or display name collides with one already registered, the late-arriving module either overwrites the existing entry or the registration is skipped altogether, leaving the node missing from the UI. On earlier Python releases that collision was often benign; on 3.13 it translates into nodes silently disappearing.

The main reasons are:

- **Stricter importlib reuse.** Python 3.13 aggressively reuses module specs. When two modules share the same `module.__spec__`, the second import short-circuits and its body never executes, so the calls that should register nodes never run.

- **Bytecode cache enforcement.** Stale `.pyc` files are now rejected immediately. If two files keep the same path (for example `flux_resolution_cal_node.py`), the cached bytecode from one version is reused for the other. When timestamps or hashes disagree, the import fails and the node is absent.

- **Global key collisions in ComfyUI.** The ComfyUI catalogue builder assumes display names are unique. Multiple “Flux Resolution Calc” entries confuse the UI catalogue generation and cause the duplicates to be dropped even if the import succeeded.

To guarantee that every node survives Python 3.13’s stricter semantics, this fork renames the modules, the classes, and the display labels (e.g. `FluxResolutionNode` → `MegapixelCalculatorNode`, shown to users as `ControlAltAI: Megapixel Calculator`). File names were updated so cached bytecode and `module.__spec__` entries no longer collide. The top-level `__init__.py` was also rewritten to support relative and absolute imports, emit a diagnostic log, and avoid accidental double-registration. With those changes in place, Python 3.13 treats each node module as distinct, and ComfyUI reliably enumerates them.

## Node Name Changes

All node class names and display names have been changed from the original repository. Here is the complete list of changes:

> Note (2026-10-09): the 11 class types above that collide with `gseth/ControlAltAI-Nodes` were further namespaced with a `hswq_` prefix (e.g. `hswq_FluxSampler`) to eliminate same-name registrations. `MegapixelCalculatorNode` keeps its name.

## Node Class Name and Display Name Changes

| Original Node Class Name | Changed Node Class Name | Original Display Name | Changed Display Name |
|-------------------------|------------------------|----------------------|---------------------|
| FluxResolutionNode | MegapixelCalculatorNode | Flux Resolution Calc | ControlAltAI: Megapixel Calculator |
| FluxSampler | hswq_FluxSampler | Flux Sampler | ControlAltAI: Advanced Sampler |
| FluxUnionControlNetApply | hswq_FluxUnionControlNetApply | Flux Union ControlNet Apply | ControlAltAI: Union ControlNet Apply |
| BooleanBasic | hswq_BooleanBasic | Boolean Basic | ControlAltAI: Boolean Basic |
| BooleanReverse | hswq_BooleanReverse | Boolean Reverse | ControlAltAI: Boolean Reverse |
| GetImageSizeRatio | hswq_GetImageSizeRatio | Get Image Size Ratio | ControlAltAI: Get Image Size Ratio |
| IntegerSettings | hswq_IntegerSettings | Integer Settings | ControlAltAI: Integer Settings |
| IntegerSettingsAdvanced | hswq_IntegerSettingsAdvanced | Integer Settings Advanced | ControlAltAI: Integer Settings Advanced |
| PerturbationTexture | hswq_PerturbationTexture | Perturbation Texture | ControlAltAI: Perturbation Texture |
| TextBridge | hswq_TextBridge | Text Bridge | ControlAltAI: Text Bridge |
| TwoWaySwitch | hswq_TwoWaySwitch | Switch (Two Way) | ControlAltAI: Switch (Two Way) |
| ThreeWaySwitch | hswq_ThreeWaySwitch | Switch (Three Way) | ControlAltAI: Switch (Three Way) |

### Key Changes Summary

1. **FluxResolutionNode → MegapixelCalculatorNode**: The main node class name has been changed, and the file has been renamed from `flux_resolution_cal_node.py` to `megapixel_calculator_node.py`.

2. **All Display Names**: All node display names now have the "ControlAltAI: " prefix to distinguish them from the original nodes.

3. **Display Name Updates**: Some display names have been updated:
   - "Flux Sampler" → "ControlAltAI: Advanced Sampler"
   - "Flux Attention Control" → "ControlAltAI: Attention Control"

## Nodes

### List of Nodes:
- Flux
  - ControlAltAI: Megapixel Calculator (Updated, May 2025)
  - ControlAltAI: Advanced Sampler
  - ControlAltAI: Union ControlNet Apply
- Logic
  - ControlAltAI: Boolean Basic
  - ControlAltAI: Boolean Reverse
  - ControlAltAI: Integer Settings
  - ControlAltAI: Integer Settings Advanced (New, June 2025)
  - ControlAltAI: Switch (Two Way) (New, June 2025)
  - ControlAltAI: Switch (Three Way) (New, June 2025)
- Image
  - ControlAltAI: Get Image Size Ratio
  - ControlAltAI: Perturbation Texture (New, June 2025)
- Utility
  - ControlAltAI: Text Bridge (New, June 2025)

<a id="controlaltai-megapixel-calculator"></a>

### ControlAltAI: Megapixel Calculator

The ControlAltAI: Megapixel Calculator is designed to determine the optimal image resolution for outputs generated using the Flux model, which is notably more oriented towards megapixels. Unlike traditional methods that rely on standard SDXL resolutions, this calculator operates based on user-specified megapixel inputs. Users can select their desired megapixel count, ranging from 0.1 to 2.0 megapixels, and aspect ratio. The calculator then provides the exact image dimensions necessary for optimal performance with the Flux model. This approach ensures that the generated images meet specific quality and size requirements tailored to the user's needs. Additionally, while the official limit is 2.0 megapixels, during testing, I have successfully generated images at higher resolutions, indicating the model's flexibility in accommodating various image dimensions without compromising quality.

- **Supported Megapixels:** 0.1 MP - 2.5 MP (change stepping to 0.1 for fine-tuned selection)
- **Note:** Generations above 1 MP may appear slightly blurry, but resolutions of 3k+ have been successfully tested on the Flux1.Dev model.
- **Custom Ratio:** Custom Ratio is now supported. Enable or disable the Custom Ratio and input any ratio. (Example: 4:9).
- **Preview:** The preview node is just a visual representation of the ratio.
- **Divisible By:** You can now choose the divisibility by 8/16/32/64. By default, it is 64. To get fine-tuned results, choose divisibility by 8. Divisibility by 32/64 is recommended for Flux Dev 1.

<a id="controlaltai-advanced-sampler"></a>

### ControlAltAI: Advanced Sampler

The ControlAltAI: Advanced Sampler node combines the functionality of the CustomSamplerAdvance node and input nodes into a single, streamlined node.

- **CFG Setting:** The CFG is fixed at 1.
- **Conditioning Input:** Only positive conditioning is supported.
- **Compatibility:** Only the samplers and schedulers compatible with the Flux model are included.
- **Latent Compatibility:** Use SD3 Empty Latent Image only. The normal empty latent image node is not compatible.

<img src="png/Advanced%20Sampler.png" width="400" alt="ControlAltAI: Advanced Sampler">

<a id="controlaltai-union-controlnet-apply"></a>

### ControlAltAI: Union ControlNet Apply

The ControlAltAI: Union ControlNet Apply node is an all-in-one node compatible with InstanX Union Pro ControlNet. It has been tested extensively with the union controlnet type and works as intended. You can combine two ControlNet Union units and get good results. Not recommended to combine more than two. The ControlNet is tested only on the Flux 1.Dev Model.

<img src="png/Union%20ControlNet%20Apply.png" width="400" alt="ControlAltAI: Union ControlNet Apply">

**Recommended Settings:**<br>
strength: 0.15-0.65.<br>
end percentage: 0.200 - 0.900.

**Recommended PreProcessors:**<br>
Canny: Canny Edge (ControlNet Aux).<br>
Tile: Tile (ControlNet Aux).<br>
Depth: Depth Anything V2 Relative (ControlNet Aux).<br>
Blue: Direct Input (Blurry Image) or Tile (ControlNet Aux).<br>
Pose: DWPose Estimator (ControlNet Aux).<br>
Gray: Image Desaturate (Comfy Essentials Custom Node).<br>
Low Quality: Direct Input.

Results: (Canny and Depth Examples not included. They are straightforward.)<br><br>
**Pixel Low Resolution to High Resolution**<br><br>
**Photo Restoration**<br><br>
**Game Asset Low Resolution Upscale**<br><br>
**Blur to UnBlur**<br><br>
**Re-Color**<br><br>

**YouTube tutorial Union ControlNet Usage: <a href="https://www.youtube.com/watch?v=4_1A5pQkJkg">Video Tutorial</a>**

**Shakker Labs & InstantX Flux ControlNet Union Pro Model Download:** <a href="https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro">Hugging Face Link</a>

<a id="controlaltai-get-image-size-ratio"></a>

### ControlAltAI: Get Image Size Ratio
This node is designed to get the image resolution in width, height, and ratio. The node can be further connected to the ControlAltAI: Megapixel Calculator. To do so, follow the following steps:
- Right-click on the ControlAltAI: Megapixel Calculator -- > Convert widget to input -- > Convert custom_aspect_ratio to input.
- Connect Ratio output to custom_aspect_ratio input.

<img src="png/Get%20Image%20Size%20Ratio.png" width="400" alt="ControlAltAI: Get Image Size Ratio">

<a id="controlaltai-integer-settings"></a>

### ControlAltAI: Integer Settings
This node is designed to give output as a raw value of 1 or 2 integers. Enable = 2, Disable = 1.

Use case: This can be set up before a two-way switch, allowing workflow logical control to flow in one or the other direction. As of now, it only controls two logical flows. In the future, we will upgrade the node to support three or more logical switch flows.

<img src="png/Integer%20Settings.png" width="400" alt="ControlAltAI: Integer Settings">

<a id="controlaltai-integer-settings-advanced"></a>

### ControlAltAI: Integer Settings Advanced
This node is designed to give output as a raw value of 1, 2 or 3 integers. Only one integer output can be enabled at a time. Connect this node with the new ControlAltAI: Switch (Three Way) for logical control.

<img src="png/Integer%20Settings%20Advanced.png" width="400" alt="ControlAltAI: Integer Settings Advanced">

<a id="controlaltai-switch-two-way"></a>

### ControlAltAI: Switch (Two Way) / <a id="controlaltai-switch-three-way"></a> ControlAltAI: Switch (Three Way)
Unlike traditional switches, which accept only one type of input, these switches will accept multiple input types and pass through those inputs if connected to the correct output. Now seamlessly connect ControlAltAI: Switch (Two Way) with the ControlAltAI: Integer Settings and the ControlAltAI: Switch (Three Way) with the ControlAltAI: Integer Settings Advanced Nodes.

<img src="png/ControlAltAI%20Switch.png" width="400" alt="ControlAltAI: Switch (Two Way) / ControlAltAI: Switch (Three Way)">

<a id="controlaltai-perturbation-texture"></a>

### ControlAltAI: Perturbation Texture
This node adds realistic texture overlays to images using advanced noise generation techniques. This node is particularly useful for enhancing portraits, adding film grain effects, or creating natural surface textures. This is an advanced version of the ControlAltAI: Noise Plus Blend Node. The node generates multi-channel noise patterns that respect the original image's color distribution, creating realistic textures that enhance rather than overpower the source material. Can be used pre/post upscale (pixel-to-pixel). 

**Settings:**<br>
noise_scale: 0.25 - 0.50.<br>
texture_strength: 10-50.<br>
perturbation_factor: 0.10-0.25.

Node can be used with or without a mask.

<img src="png/Perturbation%20Texture.png" width="400" alt="ControlAltAI: Perturbation Texture">

**Texture Type:**
- Natural: Balanced, organic texture — ideal for stylized portraits or general image enhancement without overwhelming details.
- Film Grain: Adds cinematic noise — great for final renders or creative film looks.
- Skin Pore: Subtle realism — best for **close-ups** or portraits needing natural facial texture.
- Fine Details: Emphasizes high-frequency textures — perfect for mechanical, fabric, or intricate object renders.

<a id="controlaltai-text-bridge"></a>

### ControlAltAI: Text Bridge
Utility node that provides flexible text input/output management with manual editing capabilities. This node serves as a text processing hub, accepting text from other nodes while allowing for manual overrides and edits.

- **Passthrough Mode:** When connected to another node and the text input field is empty, the incoming text is passed through unchanged.
- **Manual Override:** When text is entered in the text input field, it uses that text instead of any passthrough input.
- **Standalone Mode:** Functions as a simple text input node when no passthrough connection is made.

<img src="png/Text%20Bridge.png" width="400" alt="ControlAltAI: Text Bridge">

## License

This integration is based on **[gseth/ControlAltAI-Nodes](https://github.com/gseth/ControlAltAI-Nodes)** (Python 3.13–compatible fork of the ControlAltAI node collection). The upstream project is licensed under the **[MIT License](https://github.com/gseth/ControlAltAI-Nodes)**. Copyright and terms follow the upstream repository and ControlAltAI contributors.
