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
    "hswq_FluxSampler": FluxSampler,
    "hswq_FluxUnionControlNetApply": FluxUnionControlNetApply,
    "hswq_BooleanBasic": BooleanBasic,
    "hswq_BooleanReverse": BooleanReverse,
    "hswq_GetImageSizeRatio": GetImageSizeRatio,
    "hswq_IntegerSettings": IntegerSettings,
    "hswq_IntegerSettingsAdvanced": IntegerSettingsAdvanced,
    "hswq_PerturbationTexture": PerturbationTexture,
    "hswq_TextBridge": TextBridge,
    "hswq_TwoWaySwitch": TwoWaySwitch,
    "hswq_ThreeWaySwitch": ThreeWaySwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegapixelCalculatorNode": "ControlAltAI: Megapixel Calculator",
    "hswq_FluxSampler": "ControlAltAI: Advanced Sampler",
    "hswq_FluxUnionControlNetApply": "ControlAltAI: Union ControlNet Apply",
    "hswq_BooleanBasic": "ControlAltAI: Boolean Basic",
    "hswq_BooleanReverse": "ControlAltAI: Boolean Reverse",
    "hswq_GetImageSizeRatio": "ControlAltAI: Get Image Size Ratio",
    "hswq_IntegerSettings": "ControlAltAI: Integer Settings",
    "hswq_IntegerSettingsAdvanced": "ControlAltAI: Integer Settings Advanced",
    "hswq_PerturbationTexture": "ControlAltAI: Perturbation Texture",
    "hswq_TextBridge": "ControlAltAI: Text Bridge",
    "hswq_TwoWaySwitch": "ControlAltAI: Switch (Two Way)",
    "hswq_ThreeWaySwitch": "ControlAltAI: Switch (Three Way)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]