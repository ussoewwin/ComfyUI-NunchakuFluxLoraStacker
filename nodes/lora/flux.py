"""
This module provides the :class:`NunchakuFluxLoraStack` node
for applying multiple LoRA weights to Nunchaku FLUX models within ComfyUI.

This is a standalone implementation with dynamic UI control.
"""

import copy
import logging
import os

import folder_paths

import sys
import os

# Add the custom node directory to the path
custom_node_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if custom_node_dir not in sys.path:
    sys.path.insert(0, custom_node_dir)

from wrappers.flux import ComfyFluxWrapper

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NunchakuFluxLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku FLUX model with dynamic input.

    This node allows you to configure multiple LoRAs with their respective strengths
    in a single node, providing the same effect as chaining multiple LoRA nodes.
    Based on efficiency-nodes-comfyui implementation with dynamic LoRA count.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora_stack").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the LoRA stack node with dynamic LoRA count.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and optional LoRA inputs.
        """
        loras = ["None"] + folder_paths.get_filename_list("loras")
        
        # Base inputs
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRAs will be applied to. "
                        "Make sure the model is loaded by `Nunchaku FLUX DiT Loader`."
                    },
                ),
                "input_mode": (
                    ["simple", "advanced"],
                    {
                        "default": "simple",
                        "tooltip": "Input mode: 'simple' shows only LoRA name and weight, 'advanced' shows separate model and clip strengths."
                    },
                ),
                "lora_count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of LoRA slots to process. Adjust this to control how many LoRAs are applied.",
                    },
                ),
            },
            "optional": {},
        }

        # Add all LoRA inputs (up to 10 slots) - exactly like efficiency-nodes-comfyui
        for i in range(1, 11):  # Support up to 10 LoRAs
            inputs["required"][f"lora_name_{i}"] = (
                loras,
                {"tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."},
            )
            inputs["required"][f"lora_wt_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Overall strength for LoRA {i} (simple mode). This value can be negative.",
                },
            )
            inputs["required"][f"model_str_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Model strength for LoRA {i} (advanced mode). This value can be negative.",
                },
            )
            inputs["required"][f"clip_str_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"CLIP strength for LoRA {i} (advanced mode). This value can be negative.",
                },
            )

        return inputs

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-evaluation when lora_count changes
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # This method is called to validate inputs and can be used for dynamic UI
        return True

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "FLUX LoRA Multi Loader"

    CATEGORY = "FLUX"
    DESCRIPTION = (
        "Apply multiple LoRAs to a diffusion model in a single node. "
        "Equivalent to chaining multiple LoRA nodes but more convenient for managing many LoRAs. "
        "Supports up to 10 LoRAs simultaneously. Use 'lora_count' to control how many LoRAs are processed. "
        "Set unused slots to 'None' to skip them."
    )

    def load_lora_stack(self, model, input_mode="simple", lora_count=3, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku FLUX diffusion model.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        lora_count : int
            Number of LoRA slots to process (1-10).
        **kwargs
            Dynamic LoRA name and strength parameters.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        # Collect LoRA information to apply
        loras_to_apply = []

        # Only check the number of LoRA slots specified by lora_count
        for i in range(1, min(lora_count + 1, 11)):  # Check up to lora_count slots, max 10
            lora_name = kwargs.get(f"lora_name_{i}")

            # Skip unset or None LoRAs
            if lora_name is None or lora_name == "None" or lora_name == "":
                continue

            # Get strength based on input_mode
            if input_mode == "simple":
                lora_strength = kwargs.get(f"lora_wt_{i}", 1.0)
            else:  # advanced mode
                model_strength = kwargs.get(f"model_str_{i}", 1.0)
                clip_strength = kwargs.get(f"clip_str_{i}", 1.0)
                # For now, use model_strength as the main strength
                # In a full implementation, you might want to handle model and clip separately
                lora_strength = model_strength

            # Skip LoRAs with zero strength
            if abs(lora_strength) < 1e-5:
                continue

            loras_to_apply.append((lora_name, lora_strength))

        # Step 3: Format LoRA list (deduplicate, filter empty)
        loras_formatted = []
        seen = set()
        for lora_name, lora_strength in loras_to_apply:
            if lora_name and lora_name not in seen:
                loras_formatted.append((lora_name, lora_strength))
                seen.add(lora_name)

        print(f"DEBUG: Applying {len(loras_formatted)} LoRAs")

        # Step 1: Extract actual model from OptimizedModule if needed
        model_wrapper = model.model.diffusion_model
        actual_model_wrapper = model_wrapper._orig_mod if hasattr(model_wrapper, "_orig_mod") else model_wrapper

        # Step 2: Determine model type
        wrapper_class_name = type(actual_model_wrapper).__name__
        print(f"DEBUG: Detected model type: {wrapper_class_name}")

        if wrapper_class_name not in ("ComfyFluxWrapper", "NunchakuFluxTransformer2dModel"):
            raise ValueError(
                f"Model structure not recognized. Expected ComfyFluxWrapper or NunchakuFluxTransformer2dModel, "
                f"got {type(actual_model_wrapper)}"
            )

        # IMPORTANT:
        # Return a NEW MODEL object (do not mutate the input model in-place).
        # ComfyUI 0.7.x model_config objects may return None for __deepcopy__ via __getattr__,
        # which makes copy.deepcopy(model) crash. Use ModelPatcher.clone() + shallow-copy of the inner model instead.
        ret_model = model
        ret_model_wrapper = actual_model_wrapper
        if hasattr(model, "clone") and wrapper_class_name == "ComfyFluxWrapper":
            ret_model = model.clone()
            ret_model.model = copy.copy(model.model)  # allow replacing diffusion_model without touching original MODEL

            transformer = actual_model_wrapper.model
            new_wrapper = ComfyFluxWrapper(
                transformer,
                config=getattr(actual_model_wrapper, "config", None),
                pulid_pipeline=getattr(actual_model_wrapper, "pulid_pipeline", None),
                customized_forward=getattr(actual_model_wrapper, "customized_forward", None),
                forward_kwargs=getattr(actual_model_wrapper, "forward_kwargs", None),
            )
            orig_dm = model.model.diffusion_model
            if hasattr(orig_dm, "_orig_mod"):
                outer = copy.copy(orig_dm)
                outer._orig_mod = new_wrapper
                ret_model.model.diffusion_model = outer
                ret_model_wrapper = outer._orig_mod
            else:
                ret_model.model.diffusion_model = new_wrapper
                ret_model_wrapper = new_wrapper

        # Step 4: Handle ComfyFluxWrapper case
        if wrapper_class_name == "ComfyFluxWrapper":
            print("DEBUG: Using ComfyFluxWrapper LoRA application method")
            ret_model_wrapper.loras = []

            for lora_name, lora_strength in loras_formatted:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                ret_model_wrapper.loras.append((lora_path, lora_strength))
                print(f"DEBUG: Added LoRA {lora_name} with strength {lora_strength}")

        # Step 5: Handle NunchakuFluxTransformer2dModel case
        elif wrapper_class_name == "NunchakuFluxTransformer2dModel":
            print("DEBUG: Using NunchakuFluxTransformer2dModel LoRA application method")

            if loras_formatted:
                try:
                    from nunchaku.lora.flux.compose import compose_lora as nunchaku_compose_lora
                except ImportError:
                    nunchaku_compose_lora = None

                lora_tuples = []
                for lora_name, lora_strength in loras_formatted:
                    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
                    lora_tuples.append((lora_path, lora_strength))
                    print(f"DEBUG: Preparing LoRA {lora_name} at {lora_path}")

                if len(lora_tuples) == 1:
                    lora_path, lora_strength = lora_tuples[0]
                    ret_model_wrapper.update_lora_params(lora_path)
                    ret_model_wrapper.set_lora_strength(lora_strength)
                    print(f"DEBUG: Applied single LoRA with strength {lora_strength}")
                else:
                    composed_lora = nunchaku_compose_lora(lora_tuples)
                    ret_model_wrapper.update_lora_params(composed_lora)
                    print(f"DEBUG: Applied {len(lora_tuples)} composed LoRAs")
            else:
                ret_model_wrapper.update_lora_params(None)
                print("DEBUG: Cleared LoRA params")

        # Step 6: Validate returned model
        ret_wrapper_class_name = type(ret_model_wrapper).__name__
        if ret_wrapper_class_name not in ("ComfyFluxWrapper", "NunchakuFluxTransformer2dModel"):
            raise ValueError(
                f"Returned model structure not recognized. Expected ComfyFluxWrapper or NunchakuFluxTransformer2dModel, "
                f"got {type(ret_model_wrapper)}"
            )

        print(f"DEBUG: Successfully applied LoRA using {ret_wrapper_class_name}")
        return (ret_model,)
