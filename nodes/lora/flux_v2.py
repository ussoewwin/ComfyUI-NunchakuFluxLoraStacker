"""
This module dynamically generates V2 nodes with fixed LoRA slot counts (1 to 10) for ComfyUI Beta 2.0 (Desktop).
No JavaScript required.
"""

import copy
import logging
import os
import folder_paths
import sys

custom_node_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if custom_node_dir not in sys.path:
    sys.path.insert(0, custom_node_dir)

from wrappers.flux import ComfyFluxWrapper

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FluxLoraMultiLoaderBase:
    """Base class for fixed-slot LoRA loaders."""
    
    _slot_count = 0

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model loaded by Nunchaku FLUX DiT Loader."}),
            },
            "optional": {},
        }

        for i in range(1, cls._slot_count + 1):
            inputs["optional"][f"lora_name_{i}"] = (loras, {"tooltip": f"LoRA {i} filename"})
            # Restored step to 0.001 to allow decimals, kept min/max removed to avoid slider interference
            inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
            
            # Restored code as commented out blocks as requested
            # inputs["required"][f"model_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": f"LoRA {i} Model Strength"})
            # inputs["required"][f"clip_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": f"LoRA {i} Clip Strength"})

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora_stack"
    CATEGORY = "FLUX/MultiLoader" 

    def load_lora_stack(self, model, **kwargs):
        loras_to_apply = []
        for i in range(1, self._slot_count + 1):
            lora_name = kwargs.get(f"lora_name_{i}")
            if not lora_name or lora_name == "None": continue
            
            lora_wt = kwargs.get(f"lora_wt_{i}", 1.0)
            
            # model_str = kwargs.get(f"model_str_{i}", 1.0)
            # clip_str = kwargs.get(f"clip_str_{i}", 1.0)
            
            # If model_str/clip_str logic is needed in future, uncomment above and adjust strength logic
            # strength = model_str # Example logic
            
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

        model_wrapper = model.model.diffusion_model
        actual_wrapper = model_wrapper._orig_mod if hasattr(model_wrapper, "_orig_mod") else model_wrapper
        wrapper_class = type(actual_wrapper).__name__

        # IMPORTANT:
        # Return a NEW MODEL object (do not mutate the input model in-place).
        # ComfyUI 0.7.x model_config objects may return None for __deepcopy__ via __getattr__,
        # which makes copy.deepcopy(model) crash. Use ModelPatcher.clone() + shallow-copy of the inner model instead.
        ret_model = model
        ret_wrapper = actual_wrapper
        if hasattr(model, "clone") and wrapper_class == "ComfyFluxWrapper":
            ret_model = model.clone()
            # shallow copy to allow replacing diffusion_model without touching the original MODEL
            ret_model.model = copy.copy(model.model)

            transformer = actual_wrapper.model
            new_wrapper = ComfyFluxWrapper(
                transformer,
                config=getattr(actual_wrapper, "config", None),
                pulid_pipeline=getattr(actual_wrapper, "pulid_pipeline", None),
                customized_forward=getattr(actual_wrapper, "customized_forward", None),
                forward_kwargs=getattr(actual_wrapper, "forward_kwargs", None),
            )
            # keep outer optimized wrapper if present
            orig_dm = model.model.diffusion_model
            if hasattr(orig_dm, "_orig_mod"):
                outer = copy.copy(orig_dm)
                outer._orig_mod = new_wrapper
                ret_model.model.diffusion_model = outer
                ret_wrapper = outer._orig_mod
            else:
                ret_model.model.diffusion_model = new_wrapper
                ret_wrapper = new_wrapper

        if wrapper_class == "ComfyFluxWrapper":
            ret_wrapper.loras = []
            for name, strength in loras_formatted:
                path = folder_paths.get_full_path_or_raise("loras", name)
                ret_wrapper.loras.append((path, strength))
        elif wrapper_class == "NunchakuFluxTransformer2dModel":
            if loras_formatted:
                from nunchaku.lora.flux.compose import compose_lora
                tuples = [(folder_paths.get_full_path_or_raise("loras", n), s) for n, s in loras_formatted]
                if len(tuples) == 1:
                    ret_wrapper.update_lora_params(tuples[0][0])
                    ret_wrapper.set_lora_strength(tuples[0][1])
                else:
                    ret_wrapper.update_lora_params(compose_lora(tuples))
            else:
                ret_wrapper.update_lora_params(None)
        
        return (ret_model,)

GENERATED_NODES = {}
GENERATED_DISPLAY_NAMES = {}

# Generate only FluxLoraMultiLoader_10 (V2)
class_name = "FluxLoraMultiLoader_10"
title = "FLUX LoRA Loader V2"
display_name = "FLUX LoRA Loader V2"

node_class = type(class_name, (FluxLoraMultiLoaderBase,), {
    "_slot_count": 10,
    "TITLE": title,
    "DESCRIPTION": "Load up to 10 LoRAs."
})

GENERATED_NODES[class_name] = node_class
GENERATED_DISPLAY_NAMES[class_name] = display_name
