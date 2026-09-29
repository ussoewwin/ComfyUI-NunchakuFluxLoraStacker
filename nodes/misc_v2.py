"""
Misc V2 nodes for ComfyUI Beta 2.0 (Desktop).
"""


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


NODE_CLASS_MAPPINGS = {
    "FastGroupsBypasserV2": FastGroupsBypasserV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastGroupsBypasserV2": "Fast Groups Bypasser V2 (All Groups)",
}
