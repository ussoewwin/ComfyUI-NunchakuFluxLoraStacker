# Python 3.13 Compatible
class BooleanReverse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "reverse_boolean"
    CATEGORY = "ControlAltAI Nodes/Logic"

    def reverse_boolean(self, boolean):
        return (not boolean,)

NODE_CLASS_MAPPINGS = {
    "hswq_BooleanReverse": BooleanReverse,
}

