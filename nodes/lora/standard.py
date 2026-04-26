"""
This module dynamically generates V2 nodes with fixed LoRA slot counts (1 to 10) for ComfyUI Beta 2.0 (Desktop).
No JavaScript required.
"""

import logging
import os
import folder_paths
import sys
import comfy.utils
import comfy.sd

custom_node_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if custom_node_dir not in sys.path:
    sys.path.insert(0, custom_node_dir)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class StandardLoraLoaderBase:
    """Base class for fixed-slot LoRA loaders."""
    
    _slot_count = 0

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model loaded by Nunchaku FLUX DiT Loader."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model."}),
            },
            "optional": {},
        }

        for i in range(1, cls._slot_count + 1):
            inputs["optional"][f"lora_name_{i}"] = (loras, {"tooltip": f"LoRA {i} filename"})
            inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora_stack"
    CATEGORY = "loaders" 

    def load_lora_stack(self, model, clip, **kwargs):
        loras_to_apply = []
        for i in range(1, self._slot_count + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            if not lora_name or lora_name == "None": continue
            
            lora_wt = kwargs.get(f"lora_wt_{i}", 1.0)
            strength = lora_wt
            
            if abs(strength) < 1e-5: continue
            loras_to_apply.append((lora_name, strength))

        # Deduplicate
        loras_formatted = []
        seen = set()
        for name, strength in loras_to_apply:
            if name not in seen:
                loras_formatted.append((name, strength))
                seen.add(name)

        # CHANGED: Use ComfyUI standard LoRA loading (copied from nodes.py)
        current_model = model
        current_clip = clip
        
        for name, strength in loras_formatted:
            if strength == 0:
                continue

            lora_path = folder_paths.get_full_path_or_raise("loras", name)
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    self.loaded_lora = None

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)

            current_model, current_clip = comfy.sd.load_lora_for_models(current_model, current_clip, lora, strength, strength)
        
        return (current_model, current_clip)

GENERATED_NODES = {}
GENERATED_DISPLAY_NAMES = {}

# Generate only LoraStackerV2_10
class_name = "LoraStackerV2_10"
title = "LoRA Stacker V2"
display_name = "LoRA Stacker V2"

node_class = type(class_name, (StandardLoraLoaderBase,), {
    "_slot_count": 10,
    "TITLE": title,
    "DESCRIPTION": "Load up to 10 LoRAs."
})

GENERATED_NODES[class_name] = node_class
GENERATED_DISPLAY_NAMES[class_name] = display_name
