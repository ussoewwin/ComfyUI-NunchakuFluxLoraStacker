"""
ComfyUI-NunchakuFluxLoraStacker

A standalone ComfyUI custom node for Nunchaku FLUX LoRA Stacking.
"""

import logging

# Version information
__version__ = "1.13.0"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import nodes
from .nodes.lora.flux import NunchakuFluxLoraStack
from .nodes.lora.flux_v2 import GENERATED_NODES as FLUX_NODES, GENERATED_DISPLAY_NAMES as FLUX_NAMES
from .nodes.lora.standard import GENERATED_NODES as STANDARD_LORA_NODES, GENERATED_DISPLAY_NAMES as STANDARD_LORA_NAMES
from .nodes.lora.sdnq import GENERATED_NODES as SDNQ_LORA_NODES, GENERATED_DISPLAY_NAMES as SDNQ_LORA_NAMES
from .nodes.misc_v2 import NODE_CLASS_MAPPINGS as MISC_NODES, NODE_DISPLAY_NAME_MAPPINGS as MISC_NAMES
from .nodes.load_image_ussoewwin import NODE_CLASS_MAPPINGS as LOAD_IMAGE_NODES, NODE_DISPLAY_NAME_MAPPINGS as LOAD_IMAGE_NAMES
from .nodes.lora_analyzer_node import NODE_CLASS_MAPPINGS as ANALYZER_NODES, NODE_DISPLAY_NAME_MAPPINGS as ANALYZER_NAMES

# Add version to classes
NunchakuFluxLoraStack.__version__ = __version__
for node_class in FLUX_NODES.values():
    node_class.__version__ = __version__
for node_class in STANDARD_LORA_NODES.values():
    node_class.__version__ = __version__
for node_class in SDNQ_LORA_NODES.values():
    node_class.__version__ = __version__

# Node mappings
NODE_CLASS_MAPPINGS = {
    "FluxLoraMultiLoader": NunchakuFluxLoraStack,
    **FLUX_NODES,
    **STANDARD_LORA_NODES,
    **SDNQ_LORA_NODES,
    **MISC_NODES,
    **LOAD_IMAGE_NODES,
    **ANALYZER_NODES,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraMultiLoader": "FLUX LoRA Multi Loader (Legacy - Do Not Use in V2)",
    **FLUX_NAMES,
    **STANDARD_LORA_NAMES,
    **SDNQ_LORA_NAMES,
    **MISC_NAMES,
    **LOAD_IMAGE_NAMES,
    **ANALYZER_NAMES,
}

# Register JavaScript extensions
# Serve JS from ./js (used by this extension's frontend widgets)
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

logger.info(f"ComfyUI-NunchakuFluxLoraStacker: Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
