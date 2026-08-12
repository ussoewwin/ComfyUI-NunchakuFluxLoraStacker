"""
ComfyUI-NunchakuFluxLoraStacker

A standalone ComfyUI custom node for Nunchaku FLUX LoRA Stacking.
"""

import logging

# Version information
__version__ = "1.38"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import nodes - nunchaku-dependent imports are guarded for AMD/non-NVIDIA systems
try:
    from .nodes.lora.flux import NunchakuFluxLoraStack
    from .nodes.lora.flux_v2 import GENERATED_NODES as FLUX_NODES, GENERATED_DISPLAY_NAMES as FLUX_NAMES
    _FLUX_AVAILABLE = True
except Exception as e:
    logger.warning(f"[ROCm] Skipping nunchaku FLUX nodes: {e}")
    NunchakuFluxLoraStack = None
    FLUX_NODES = {}
    FLUX_NAMES = {}
    _FLUX_AVAILABLE = False

try:
    from .nodes.lora.standard import GENERATED_NODES as STANDARD_LORA_NODES, GENERATED_DISPLAY_NAMES as STANDARD_LORA_NAMES
except Exception as e:
    logger.warning(f"[ROCm] Skipping standard LoRA nodes: {e}")
    STANDARD_LORA_NODES = {}
    STANDARD_LORA_NAMES = {}

try:
    from .nodes.lora.standard_v3 import GENERATED_NODES as STANDARD_LORA_V3_NODES, GENERATED_DISPLAY_NAMES as STANDARD_LORA_V3_NAMES
except Exception as e:
    logger.warning(f"[ROCm] Skipping standard LoRA v3 nodes: {e}")
    STANDARD_LORA_V3_NODES = {}
    STANDARD_LORA_V3_NAMES = {}


from .nodes.lora.sdnq import GENERATED_NODES as SDNQ_LORA_NODES, GENERATED_DISPLAY_NAMES as SDNQ_LORA_NAMES
from .nodes.misc_v2 import NODE_CLASS_MAPPINGS as MISC_NODES, NODE_DISPLAY_NAME_MAPPINGS as MISC_NAMES
from .nodes.load_image_ussoewwin import NODE_CLASS_MAPPINGS as LOAD_IMAGE_NODES, NODE_DISPLAY_NAME_MAPPINGS as LOAD_IMAGE_NAMES
from .nodes.lora_analyzer_node import NODE_CLASS_MAPPINGS as ANALYZER_NODES, NODE_DISPLAY_NAME_MAPPINGS as ANALYZER_NAMES
from .nodes.color_filter import NODE_CLASS_MAPPINGS as COLOR_FILTER_NODES, NODE_DISPLAY_NAME_MAPPINGS as COLOR_FILTER_NAMES
from .nodes.florence2 import NODE_CLASS_MAPPINGS as FLORENCE2_NODES, NODE_DISPLAY_NAME_MAPPINGS as FLORENCE2_NAMES
from .nodes.controlaltai import NODE_CLASS_MAPPINGS as CONTROLALTAI_NODES, NODE_DISPLAY_NAME_MAPPINGS as CONTROLALTAI_NAMES
from .nodes.CCSR import NODE_CLASS_MAPPINGS as CCSR_NODES, NODE_DISPLAY_NAME_MAPPINGS as CCSR_NAMES
from .nodes.resolution_selector import NODE_CLASS_MAPPINGS as RESOLUTION_SELECTOR_NODES, NODE_DISPLAY_NAME_MAPPINGS as RESOLUTION_SELECTOR_NAMES

# Add version to classes
if NunchakuFluxLoraStack is not None:
    NunchakuFluxLoraStack.__version__ = __version__
for node_class in FLUX_NODES.values():
    node_class.__version__ = __version__
for node_class in STANDARD_LORA_NODES.values():
    node_class.__version__ = __version__
for node_class in STANDARD_LORA_V3_NODES.values():
    node_class.__version__ = __version__
for node_class in SDNQ_LORA_NODES.values():
    node_class.__version__ = __version__
for node_class in FLORENCE2_NODES.values():
    node_class.__version__ = __version__
for node_class in CCSR_NODES.values():
    node_class.__version__ = __version__

# Node mappings
NODE_CLASS_MAPPINGS = {
    **({} if NunchakuFluxLoraStack is None else {"FluxLoraMultiLoader": NunchakuFluxLoraStack}),
    **FLUX_NODES,
    **STANDARD_LORA_NODES,
    **STANDARD_LORA_V3_NODES,
    **SDNQ_LORA_NODES,
    **MISC_NODES,
    **LOAD_IMAGE_NODES,
    **ANALYZER_NODES,
    **COLOR_FILTER_NODES,
    **FLORENCE2_NODES,
    **CONTROLALTAI_NODES,
    **CCSR_NODES,
    **RESOLUTION_SELECTOR_NODES,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    **({} if NunchakuFluxLoraStack is None else {"FluxLoraMultiLoader": "FLUX LoRA Multi Loader (Legacy - Do Not Use in V2)"}),
    **FLUX_NAMES,
    **STANDARD_LORA_NAMES,
    **STANDARD_LORA_V3_NAMES,
    **SDNQ_LORA_NAMES,
    **MISC_NAMES,
    **LOAD_IMAGE_NAMES,
    **ANALYZER_NAMES,
    **COLOR_FILTER_NAMES,
    **FLORENCE2_NAMES,
    **CONTROLALTAI_NAMES,
    **CCSR_NAMES,
    **RESOLUTION_SELECTOR_NAMES,
}

# Register JavaScript extensions
# Serve JS from ./js (used by this extension's frontend widgets)
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

logger.info(f"ComfyUI-NunchakuFluxLoraStacker: Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
