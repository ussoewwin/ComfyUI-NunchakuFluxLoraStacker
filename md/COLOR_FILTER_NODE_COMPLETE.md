# Color Filter Node — Complete Technical Reference (English)

This document describes the **Color Filter** feature added in **v1.30**: what it does, which files were touched, the **full source code** as shipped in this repository, and how each part behaves.

---

## 1. Meaning of the added feature

Vision-language captioning (e.g. **Florence-2**) and image taggers (e.g. **WD14 Tagger**) often output tokens that describe the source photograph literally—such as “black and white”, “monochrome”, or non-English equivalents that appear in training data. When that text is concatenated into a **positive prompt** for diffusion models, those tokens can **steer generation toward grayscale or desaturated results** even when the user wants full color.

**Color Filter** is a small **ComfyUI string utility node** that:

1. Accepts a multiline `STRING` (`text`).
2. Removes a **fixed list of regular-expression patterns** associated with monochrome / grayscale / sepia style descriptions (English word-boundary patterns plus additional literals used in Japanese tagging).
3. Returns a single `STRING` (`filtered_text`) with **all matched spans deleted** and **all runs of whitespace collapsed** to a single ASCII space, then **stripped** at the ends.

It does **not** call Florence-2, WD14, or any model. It only post-processes text you connect from upstream nodes.

---

## 2. Files added or modified for this feature

| Role | Path |
|------|------|
| Node implementation | `nodes/color_filter/color_filter.py` |
| ComfyUI registration bundle | `nodes/color_filter/__init__.py` |
| Extension entrypoint (merge mappings) | `__init__.py` (repository root) |
| User-facing overview | `README.md` (section “Color Filter”, item 7 in the node list, screenshot) |
| Screenshot asset | `png/colorfilter.png` |
| Release history line | `md/CHANGELOG.md` (v1.30 entry with link to GitHub Releases) |

There is **no JavaScript** for this node: inputs are standard ComfyUI string widgets.

---

## 3. Full source code

### 3.1 `nodes/color_filter/color_filter.py` (complete file)

```python
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
```

### 3.2 `nodes/color_filter/__init__.py` (complete file)

```python
from .color_filter import ColorFilter

NODE_CLASS_MAPPINGS = {
    "ColorFilter": ColorFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorFilter": "Color Filter",
}
```

### 3.3 Repository root `__init__.py` (complete file)

The Color Filter integration adds **one import line** and **two dict merge entries** (`**COLOR_FILTER_NODES`, `**COLOR_FILTER_NAMES`). The file below is the **entire** root `__init__.py` as in the repository (other nodes unchanged except for this wiring).

```python
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
from .nodes.color_filter import NODE_CLASS_MAPPINGS as COLOR_FILTER_NODES, NODE_DISPLAY_NAME_MAPPINGS as COLOR_FILTER_NAMES

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
    **COLOR_FILTER_NODES,
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
    **COLOR_FILTER_NAMES,
}

# Register JavaScript extensions
# Serve JS from ./js (used by this extension's frontend widgets)
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

logger.info(f"ComfyUI-NunchakuFluxLoraStacker: Loaded {len(NODE_CLASS_MAPPINGS)} nodes")
```

---

## 4. Code semantics (what each part means)

### 4.1 ComfyUI node contract (`ColorFilter` class)

| Attribute | Value | Meaning |
|-----------|--------|---------|
| `INPUT_TYPES` | One required `STRING`, `multiline: True` | Users paste or edit long captions/tags in the node UI. |
| `RETURN_TYPES` | `("STRING",)` | Single string output. |
| `RETURN_NAMES` | `("filtered_text",)` | Output socket label in the graph. |
| `FUNCTION` | `"filter_text"` | ComfyUI invokes `ColorFilter.filter_text(...)`. |
| `CATEGORY` | `"Text/Filter"` | Menu path in the “Add Node” UI. |

`__init__` is empty but kept for symmetry with other nodes in the codebase.

### 4.2 `keywords_to_remove` — pattern list

Each entry is a **regular expression** passed to `re.sub`.

- **English entries** use `\b` (word boundary) where appropriate so short tokens like `B&W` do not match inside unrelated longer tokens (within the limits of the `\b` rules in Python’s regex engine).
- **Japanese (and related) entries** are plain substrings without `\b`, because word boundaries do not apply the same way; any occurrence of those character sequences is removed.

**Semantic groups:**

| Patterns (conceptual) | Intent |
|------------------------|--------|
| `white and black`, `black and white`, `monochrome`, `grayscale` / `grey scale` / `greyscale`, `B&W`, `achromatic` | Common English descriptions of non-color imagery. |
| `desaturated` | Often used for “flat” or low-chroma look; included for parity with the original standalone filter this node was ported from. |
| `blonde`, `blonde hair` | **Not** a monochrome descriptor; they were kept when aligning with the original external node’s pattern list. Removing them would change backward-compatible behavior for anyone who relied on that list. Be aware they **will** strip those English phrases if present. |
| CJK literals in the source (`白黒`, `モノクロ`, `グレースケール`, `無彩色`, `セピア`) | Frequent outputs from Japanese taggers / captions for monochrome or sepia. |

If you need a different policy (e.g. drop `blonde`), edit `keywords_to_remove` in `color_filter.py`.

### 4.3 Removal loop

```python
for keyword in keywords_to_remove:
    filtered_text = re.sub(keyword, "", filtered_text, flags=re.IGNORECASE)
```

- Each pattern is applied **in sequence** on the string produced by the previous step.
- `re.IGNORECASE` applies to **all** patterns in the loop. For patterns that are only CJK characters, case is irrelevant but the flag is harmless.
- Replacing with `""` means **delete** every non-overlapping match for that pattern.

**Order matters** only if patterns could overlap; in practice these patterns are distinct phrases.

### 4.4 Whitespace normalization

```python
filtered_text = re.sub(r"\s+", " ", filtered_text).strip()
```

- `\s` matches spaces, tabs, newlines, and other Unicode whitespace categories Python assigns to `\s`.
- **Every maximal run** of whitespace becomes **one** ASCII space `0x20`.
- Therefore **multiline input becomes a single line** of words separated by single spaces (unless you later split elsewhere).
- `.strip()` removes leading/trailing space after normalization.

This behavior matches the original standalone Color Filter node: it avoids doubled gaps after deletions but **does not preserve original line breaks**.

### 4.5 Return value

```python
return (filtered_text,)
```

ComfyUI expects a **tuple** of outputs matching `RETURN_TYPES`. One string in → one string out.

---

## 5. Registration path (how ComfyUI discovers the node)

1. ComfyUI loads the custom node package (this repository) and executes root **`__init__.py`**.
2. Root `__init__.py` imports `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` from **`nodes.color_filter`**.
3. Those dicts are merged with `**COLOR_FILTER_NODES` and `**COLOR_FILTER_NAMES` into the extension’s global mappings.
4. ComfyUI registers internal node type **`ColorFilter`** with display name **“Color Filter”**.

The **Python class** name is `ColorFilter`. The **graph node type string** is also `ColorFilter` (the dict key). Renaming either requires updating `nodes/color_filter/__init__.py` and any saved workflows that reference the old type name.

---

## 6. Operational limits (honest scope)

- **Not a semantic parser:** It does not use an LLM; it only applies the fixed regex list.
- **False positives:** e.g. English `blonde` is stripped whenever it appears as a whole word (see §4.2).
- **False negatives:** any monochrome wording **not** in the list remains in the string.
- **Whitespace:** Newlines are not preserved after filtering (see §4.4).

---

## 7. Related documentation elsewhere

- **README:** high-level user story, Florence-2 / WD14 positioning, input/output table, screenshot `png/colorfilter.png`.
- **CHANGELOG:** one-line v1.30 summary and link to GitHub Releases for the tag.

This file (`md/COLOR_FILTER_NODE_COMPLETE.md`) is the **authoritative low-level** reference for behavior and code layout of the Color Filter integration.
