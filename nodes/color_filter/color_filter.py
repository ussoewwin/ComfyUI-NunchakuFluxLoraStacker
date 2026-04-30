"""Remove monochrome / B&W related phrases from caption or prompt text."""

import re


class ColorFilter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filtered_text",)
    FUNCTION = "filter_text"
    CATEGORY = "Text/Filter"

    def filter_text(self, text):
        keywords_to_remove = [
            r"\bwhite and black\b",
            r"\bblack and white\b",
            r"\bmonochrome\b",
            r"\bgrayscale\b",
            r"\bgrey scale\b",
            r"\bgreyscale\b",
            r"\bB&W\b",
            r"\bdesaturated\b",
            r"\bblonde\b",
            r"\bblonde hair\b",
            r"\bachromatic\b",
            r"白黒",
            r"モノクロ",
            r"グレースケール",
            r"無彩色",
            r"セピア",
        ]

        filtered_text = text
        for keyword in keywords_to_remove:
            filtered_text = re.sub(keyword, "", filtered_text, flags=re.IGNORECASE)

        filtered_text = re.sub(r"\s+", " ", filtered_text).strip()

        return (filtered_text,)
