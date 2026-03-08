"""
Misc V2 nodes for ComfyUI Beta 2.0 (Desktop).
"""

import logging
import torch
from torch import nn
import folder_paths
import comfy.utils
import comfy.ops
import comfy.model_management
import comfy.ldm.common_dit
import comfy.latent_formats
import comfy.ldm.lumina.controlnet

class FastGroupsBypasserV2:
    """
    A V2-compatible Fast Groups Bypasser.
    Reproduces original behavior: Dynamically lists all groups and provides toggles.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Base widgets, actual toggles are added by JS
            },
            "optional": {
            },
            "hidden": {
                "id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    # Accept any kwargs since JS will add dynamic toggles
    def do_nothing(self, **kwargs):
        return ()


# Model Patch Loader related classes and functions
class BlockWiseControlBlock(torch.nn.Module):
    # [linear, gelu, linear]
    def __init__(self, dim: int = 3072, device=None, dtype=None, operations=None):
        super().__init__()
        self.x_rms = operations.RMSNorm(dim, eps=1e-6)
        self.y_rms = operations.RMSNorm(dim, eps=1e-6)
        self.input_proj = operations.Linear(dim, dim)
        self.act = torch.nn.GELU()
        self.output_proj = operations.Linear(dim, dim)

    def forward(self, x, y):
        x, y = self.x_rms(x), self.y_rms(y)
        x = self.input_proj(x + y)
        x = self.act(x)
        x = self.output_proj(x)
        return x


class QwenImageBlockWiseControlNet(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        in_dim: int = 64,
        additional_in_dim: int = 0,
        dim: int = 3072,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.additional_in_dim = additional_in_dim
        self.img_in = operations.Linear(in_dim + additional_in_dim, dim, device=device, dtype=dtype)
        self.controlnet_blocks = torch.nn.ModuleList(
            [
                BlockWiseControlBlock(dim, device=device, dtype=dtype, operations=operations)
                for _ in range(num_layers)
            ]
        )

    def process_input_latent_image(self, latent_image):
        latent_image[:, :16] = comfy.latent_formats.Wan21().process_in(latent_image[:, :16])
        patch_size = 2
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(latent_image, (1, patch_size, patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        return self.img_in(hidden_states)

    def control_block(self, img, controlnet_conditioning, block_id):
        return self.controlnet_blocks[block_id](img, controlnet_conditioning)


class SigLIPMultiFeatProjModel(torch.nn.Module):
    """
    SigLIP Multi-Feature Projection Model for processing style features from different layers
    and projecting them into a unified hidden space.
    """

    def __init__(
        self,
        siglip_token_nums: int = 729,
        style_token_nums: int = 64,
        siglip_token_dims: int = 1152,
        hidden_size: int = 3072,
        context_layer_norm: bool = True,
        device=None, dtype=None, operations=None
    ):
        super().__init__()

        # High-level feature processing (layer -2)
        self.high_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.high_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.high_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Mid-level feature processing (layer -11)
        self.mid_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.mid_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.mid_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

        # Low-level feature processing (layer -20)
        self.low_embedding_linear = nn.Sequential(
            operations.Linear(siglip_token_nums, style_token_nums),
            nn.SiLU()
        )
        self.low_layer_norm = (
            operations.LayerNorm(siglip_token_dims) if context_layer_norm else nn.Identity()
        )
        self.low_projection = operations.Linear(siglip_token_dims, hidden_size, bias=True)

    def forward(self, siglip_outputs):
        dtype = next(self.high_embedding_linear.parameters()).dtype

        # Process high-level features (layer -2)
        high_embedding = self._process_layer_features(
            siglip_outputs[2],
            self.high_embedding_linear,
            self.high_layer_norm,
            self.high_projection,
            dtype
        )

        # Process mid-level features (layer -11)
        mid_embedding = self._process_layer_features(
            siglip_outputs[1],
            self.mid_embedding_linear,
            self.mid_layer_norm,
            self.mid_projection,
            dtype
        )

        # Process low-level features (layer -20)
        low_embedding = self._process_layer_features(
            siglip_outputs[0],
            self.low_embedding_linear,
            self.low_layer_norm,
            self.low_projection,
            dtype
        )

        # Concatenate features from all layers
        return torch.cat((high_embedding, mid_embedding, low_embedding), dim=1)

    def _process_layer_features(
        self,
        hidden_states: torch.Tensor,
        embedding_linear: nn.Module,
        layer_norm: nn.Module,
        projection: nn.Module,
        dtype: torch.dtype
    ) -> torch.Tensor:
        embedding = embedding_linear(
            hidden_states.to(dtype).transpose(1, 2)
        ).transpose(1, 2)

        embedding = layer_norm(embedding)
        embedding = projection(embedding)

        return embedding


def z_image_convert(sd):
    replace_keys = {".attention.to_out.0.bias": ".attention.out.bias",
                    ".attention.norm_k.weight": ".attention.k_norm.weight",
                    ".attention.norm_q.weight": ".attention.q_norm.weight",
                    ".attention.to_out.0.weight": ".attention.out.weight"
                    }

    out_sd = {}
    for k in sorted(sd.keys()):
        w = sd[k]

        k_out = k
        if k_out.endswith(".attention.to_k.weight"):
            cc = [w]
            continue
        if k_out.endswith(".attention.to_q.weight"):
            cc = [w] + cc
            continue
        if k_out.endswith(".attention.to_v.weight"):
            cc = cc + [w]
            w = torch.cat(cc, dim=0)
            k_out = k_out.replace(".attention.to_v.weight", ".attention.qkv.weight")

        for r, rr in replace_keys.items():
            k_out = k_out.replace(r, rr)
        out_sd[k_out] = w

    return out_sd


class ModelPatchLoaderCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "name": (folder_paths.get_filename_list("model_patches"), ),
            "cpu_offload": ("BOOLEAN", {"default": True, "tooltip": "Load model to CPU (main memory). Does not use VRAM."}),
                              }}
    RETURN_TYPES = ("MODEL_PATCH",)
    FUNCTION = "load_model_patch"
    EXPERIMENTAL = True

    CATEGORY = "advanced/loaders"

    def load_model_patch(self, name, cpu_offload):
        model_patch_path = folder_paths.get_full_path_or_raise("model_patches", name)
        sd = comfy.utils.load_torch_file(model_patch_path, safe_load=True)
        dtype = comfy.utils.weight_dtype(sd)

        # Select device based on CPU offload setting
        if cpu_offload:
            # CPU offload: Load all models to CPU (main memory)
            load_device = torch.device("cpu")
            offload_device = torch.device("cpu")
            model_device = torch.device("cpu")
        else:
            # Normal mode: Use GPU
            load_device = comfy.model_management.get_torch_device()
            offload_device = comfy.model_management.unet_offload_device()
            model_device = comfy.model_management.unet_offload_device()

        if 'controlnet_blocks.0.y_rms.weight' in sd:
            additional_in_dim = sd["img_in.weight"].shape[1] - 64
            model = QwenImageBlockWiseControlNet(additional_in_dim=additional_in_dim, device=model_device, dtype=dtype, operations=comfy.ops.manual_cast)
        elif 'feature_embedder.mid_layer_norm.bias' in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"feature_embedder.": ""}, filter_keys=True)
            model = SigLIPMultiFeatProjModel(device=model_device, dtype=dtype, operations=comfy.ops.manual_cast)
        elif 'control_all_x_embedder.2-1.weight' in sd: # alipai z image fun controlnet
            sd = z_image_convert(sd)
            config = {}
            # Check for 2.0 or 2.1 by counting control_layers
            n_control_layers = 0
            for k in sd.keys():
                if k.startswith('control_layers.') and '.adaLN_modulation.0.weight' in k:
                    layer_idx = int(k.split('.')[1])
                    n_control_layers = max(n_control_layers, layer_idx + 1)
            
            # Fallback to 2.0 detection if dynamic count fails
            if n_control_layers == 0 and 'control_layers.14.adaLN_modulation.0.weight' in sd:
                n_control_layers = 15
            
            if n_control_layers > 0:
                config['n_control_layers'] = n_control_layers
                config['additional_in_dim'] = 17
                config['refiner_control'] = True
                ref_weight = sd.get("control_noise_refiner.0.after_proj.weight", None)
                if ref_weight is not None:
                    if torch.count_nonzero(ref_weight) == 0:
                        config['broken'] = True

            # Infer control_in_dim from checkpoint so embedder input dim matches (avoids matmul shape error)
            embedder_in = sd["control_all_x_embedder.2-1.weight"].shape[1]
            expected_channels = embedder_in // 4  # f_patch_size * patch_size * patch_size = 4
            if 'additional_in_dim' in config:
                config['control_in_dim'] = expected_channels - config['additional_in_dim']
            else:
                config['control_in_dim'] = expected_channels

            model = comfy.ldm.lumina.controlnet.ZImage_Control(device=model_device, dtype=dtype, operations=comfy.ops.manual_cast, **config)
            
            # Filter only size mismatches; keep keys that are in checkpoint but not in model.state_dict()
            # (e.g. on Windows/aimdo some Linear layers have weight=None so they don't appear in state_dict
            # until load_state_dict runs and _load_from_state_dict sets them)
            model_state_dict = model.state_dict()
            filtered_sd = {}
            size_mismatch_keys = []
            keys_not_in_model = []

            for k, v in sd.items():
                if k in model_state_dict:
                    if v.shape == model_state_dict[k].shape:
                        filtered_sd[k] = v
                    else:
                        size_mismatch_keys.append(f"{k}: checkpoint shape {v.shape} vs model shape {model_state_dict[k].shape}")
                else:
                    # Include anyway so load_state_dict / _load_from_state_dict can load (e.g. lazy weight)
                    filtered_sd[k] = v
                    keys_not_in_model.append(k)

            if keys_not_in_model:
                print(f"[ModelPatchLoaderCustom] Info: {len(keys_not_in_model)} keys loaded via state_dict (e.g. lazy-init layers)")
            if size_mismatch_keys:
                print(f"[ModelPatchLoaderCustom] Warning: {len(size_mismatch_keys)} keys have size mismatches (excluded)")
                for key_info in size_mismatch_keys[:5]:
                    print(f"  - {key_info}")
                if len(size_mismatch_keys) > 5:
                    print(f"  ... and {len(size_mismatch_keys) - 5} more")

            sd = filtered_sd

        # Load with strict=False (only load matching keys)
        model.load_state_dict(sd, strict=False)
        # Set load_device and offload_device
        model = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "FastGroupsBypasserV2": FastGroupsBypasserV2,
    "ModelPatchLoaderCustom": ModelPatchLoaderCustom
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastGroupsBypasserV2": "Fast Groups Bypasser V2 (All Groups)",
    "ModelPatchLoaderCustom": "Model Patch Loader"
}
