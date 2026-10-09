# Python 3.13 Compatible
class BooleanBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "process_boolean"
    CATEGORY = "ControlAltAI Nodes/Logic"

    def process_boolean(self, boolean):
        return (boolean,)

NODE_CLASS_MAPPINGS = {
    "CAI_BooleanBasic": BooleanBasic,
}

