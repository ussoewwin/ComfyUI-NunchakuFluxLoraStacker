"""
This module dynamically generates V2 nodes with fixed LoRA slot counts (1 to 10) for ComfyUI Beta 2.0 (Desktop).
No JavaScript required.
"""

import logging
import os
import folder_paths
import sys
from typing import Tuple
from diffusers import DiffusionPipeline

custom_node_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if custom_node_dir not in sys.path:
    sys.path.insert(0, custom_node_dir)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class StandardLoraLoaderBase:
    """Base class for fixed-slot LoRA loaders (diffusers format)."""
    
    _slot_count = 0

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        inputs = {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model loaded by SDNQ Model Loader (DiffusionPipeline)."}),
            },
            "optional": {},
        }

        for i in range(1, cls._slot_count + 1):
            inputs["optional"][f"lora_name_{i}"] = (loras, {"tooltip": f"LoRA {i} filename"})
            inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora_stack"
    CATEGORY = "loaders" 

    def _resolve_lora_path(self, lora_selection: str) -> str:
        """
        Resolve actual LoRA path from selection.
        
        Args:
            lora_selection: Selected LoRA from dropdown (None or filename)
            
        Returns:
            Resolved LoRA path or None if no LoRA selected
        """
        if not lora_selection or lora_selection == "None":
            return None
        
        try:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_selection)
            return lora_path
        except Exception as e:
            print(f"[SDNQ LoRA Stacker] Warning: Could not resolve LoRA path: {e}")
            return None

    def load_lora_stack(self, model: DiffusionPipeline, **kwargs) -> Tuple[DiffusionPipeline]:
        """
        Main function called by ComfyUI.
        
        Supports up to 10 LoRA slots (lora_name_1, lora_wt_1, ... lora_name_10, lora_wt_10).
        Multiple LoRAs are loaded sequentially and combined using diffusers API.
        
        Args:
            model: DiffusionPipeline from SDNQModelLoader
            **kwargs: LoRA parameters (lora_name_1, lora_wt_1, ... lora_name_10, lora_wt_10)
            
        Returns:
            Tuple containing (MODEL,) with LoRAs applied
        """
        # Unload all existing LoRA adapters first to avoid name conflicts
        try:
            if hasattr(model, 'peft_config') and model.peft_config:
                # Get list of adapter names
                adapter_names = list(model.peft_config.keys())
                if adapter_names:
                    print(f"[SDNQ LoRA Stacker] Unloading {len(adapter_names)} existing adapter(s): {adapter_names}")
                    model.unload_lora_weights()
        except Exception as e:
            print(f"[SDNQ LoRA Stacker] Warning: Could not unload existing adapters: {e}")
        
        # Process up to 10 LoRA slots
        lora_adapters = []
        lora_weights = []
        
        for i in range(1, self._slot_count + 1):
            lora_name_key = f"lora_name_{i}"
            lora_wt_key = f"lora_wt_{i}"
            
            if lora_name_key not in kwargs or lora_wt_key not in kwargs:
                continue
            
            lora_selection = kwargs.get(lora_name_key)
            lora_strength = kwargs.get(lora_wt_key, 1.0)
            
            # Skip if no LoRA selected or strength is 0
            if not lora_selection or lora_selection == "None" or abs(lora_strength) < 1e-5:
                continue
            
            # Resolve LoRA path
            lora_path = self._resolve_lora_path(lora_selection)
            
            if lora_path and lora_path.strip():
                try:
                    # Check if it's a local file or HuggingFace repo
                    is_local_file = os.path.exists(lora_path) and os.path.isfile(lora_path)
                    
                    adapter_name = f"lora_{i}"
                    
                    # Check if adapter name already exists, unload it first
                    try:
                        if hasattr(model, 'peft_config') and model.peft_config and adapter_name in model.peft_config:
                            print(f"[SDNQ LoRA Stacker] Adapter {adapter_name} already exists, unloading first...")
                            model.unload_lora_weights(adapter_name)
                    except Exception as e:
                        print(f"[SDNQ LoRA Stacker] Warning: Could not unload existing adapter {adapter_name}: {e}")
                    
                    if is_local_file:
                        # Local .safetensors file
                        lora_dir = os.path.dirname(lora_path)
                        lora_file = os.path.basename(lora_path)
                        
                        model.load_lora_weights(
                            lora_dir,
                            weight_name=lora_file,
                            adapter_name=adapter_name
                        )
                    else:
                        # Assume it's a HuggingFace repo ID
                        model.load_lora_weights(
                            lora_path,
                            adapter_name=adapter_name
                        )
                    
                    lora_adapters.append(adapter_name)
                    lora_weights.append(lora_strength)
                    
                    print(f"[SDNQ LoRA Stacker] ✓ LoRA {i} loaded: {lora_selection} (strength: {lora_strength})")
                    
                except Exception as e:
                    import traceback
                    print(f"[SDNQ LoRA Stacker] ⚠️  Failed to load LoRA {i} ({lora_selection}): {e}")
                    traceback.print_exc()
                    continue
        
        # Set all adapters with their weights
        if lora_adapters:
            model.set_adapters(lora_adapters, adapter_weights=lora_weights)
            print(f"[SDNQ LoRA Stacker] ✓ {len(lora_adapters)} LoRA(s) applied to pipeline")
        else:
            print(f"[SDNQ LoRA Stacker] No LoRAs to load")
        
        return (model,)

GENERATED_NODES = {}
GENERATED_DISPLAY_NAMES = {}

# Generate only SDNQLoraStackerV2_10
class_name = "SDNQLoraStackerV2_10"
title = "SDNQ LoRA Stacker V2"
display_name = "SDNQ LoRA Stacker V2"

node_class = type(class_name, (StandardLoraLoaderBase,), {
    "_slot_count": 10,
    "TITLE": title,
    "DESCRIPTION": "Load up to 10 LoRAs."
})

GENERATED_NODES[class_name] = node_class
GENERATED_DISPLAY_NAMES[class_name] = display_name
