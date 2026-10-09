# Python 3.13 Compatible
print("\n\033[32mInitializing ControlAltAI Nodes\033[0m")  # Fixed green reset

try:
    from .megapixel_calculator_node import MegapixelCalculatorNode
    from .flux_sampler_node import FluxSampler
    from .flux_union_controlnet_node import FluxUnionControlNetApply
    from .boolean_basic_node import BooleanBasic
    from .boolean_reverse_node import BooleanReverse
    from .get_image_size_ratio_node import GetImageSizeRatio
    from .integer_settings_node import IntegerSettings
    from .integer_settings_advanced_node import IntegerSettingsAdvanced
    from .perturbation_texture_node import PerturbationTexture
    from .text_bridge_node import TextBridge
    from .two_way_switch_node import TwoWaySwitch
    from .three_way_switch_node import ThreeWaySwitch
except ImportError:
    # Fallback to absolute imports for Python 3.13 compatibility
    from megapixel_calculator_node import MegapixelCalculatorNode
    from flux_sampler_node import FluxSampler
    from flux_union_controlnet_node import FluxUnionControlNetApply
    from boolean_basic_node import BooleanBasic
    from boolean_reverse_node import BooleanReverse
    from get_image_size_ratio_node import GetImageSizeRatio
    from integer_settings_node import IntegerSettings
    from integer_settings_advanced_node import IntegerSettingsAdvanced
    from perturbation_texture_node import PerturbationTexture
    from text_bridge_node import TextBridge
    from two_way_switch_node import TwoWaySwitch
    from three_way_switch_node import ThreeWaySwitch

NODE_CLASS_MAPPINGS = {
    "MegapixelCalculatorNode": MegapixelCalculatorNode,
    "CAI_FluxSampler": FluxSampler,
    "CAI_FluxUnionControlNetApply": FluxUnionControlNetApply,
    "CAI_BooleanBasic": BooleanBasic,
    "CAI_BooleanReverse": BooleanReverse,
    "CAI_GetImageSizeRatio": GetImageSizeRatio,
    "CAI_IntegerSettings": IntegerSettings,
    "CAI_IntegerSettingsAdvanced": IntegerSettingsAdvanced,
    "CAI_PerturbationTexture": PerturbationTexture,
    "CAI_TextBridge": TextBridge,
    "CAI_TwoWaySwitch": TwoWaySwitch,
    "CAI_ThreeWaySwitch": ThreeWaySwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegapixelCalculatorNode": "ControlAltAI: Megapixel Calculator",
    "CAI_FluxSampler": "ControlAltAI: Advanced Sampler",
    "CAI_FluxUnionControlNetApply": "ControlAltAI: Union ControlNet Apply",
    "CAI_BooleanBasic": "ControlAltAI: Boolean Basic",
    "CAI_BooleanReverse": "ControlAltAI: Boolean Reverse",
    "CAI_GetImageSizeRatio": "ControlAltAI: Get Image Size Ratio",
    "CAI_IntegerSettings": "ControlAltAI: Integer Settings",
    "CAI_IntegerSettingsAdvanced": "ControlAltAI: Integer Settings Advanced",
    "CAI_PerturbationTexture": "ControlAltAI: Perturbation Texture",
    "CAI_TextBridge": "ControlAltAI: Text Bridge",
    "CAI_TwoWaySwitch": "ControlAltAI: Switch (Two Way)",
    "CAI_ThreeWaySwitch": "ControlAltAI: Switch (Three Way)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]