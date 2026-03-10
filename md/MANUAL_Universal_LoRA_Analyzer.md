# Universal LoRA Analyzer — Manual

This document describes the **Universal LoRA Analyzer** node added to **ComfyUI-NunchakuFluxLoraStacker**, including implementation details and technical background.

## 1. Overview

**Universal LoRA Analyzer** is a custom node for analyzing LoRA files in detail on ComfyUI. It obtains model type (SDXL, Pony, Flux, etc.), trigger words, base model, and source URLs (Civitai / HuggingFace), and supports workflow automation and information checking.

## 2. Added / Modified Files

| File path | Change | Description |
| :--- | :--- | :--- |
| `nodes/lora_analyzer_node.py` | **New** | Implements all node logic (class `UniversalLoRAAnalyzer`). |
| `__init__.py` | **Modified** | Registers the new node with ComfyUI (mappings). |

---

## 3. Code Details

### 3.1. Class design (`nodes/lora_analyzer_node.py`)

```python
class UniversalLoRAAnalyzer:
    # ...
NODE_CLASS_MAPPINGS = {
    "UniversalLoRAAnalyzer": UniversalLoRAAnalyzer
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalLoRAAnalyzer": "Universal LoRA Analyzer"
}
```

- **Class name**: `UniversalLoRAAnalyzer` — Indicates that the node is not tied to a single architecture and can analyze a wide range of LoRAs (SDXL, Flux, SD1.5, etc.).
- **Node ID**: `"UniversalLoRAAnalyzer"` — Unique identifier to avoid clashes with other nodes.

### 3.2. Model type detection (`_determine_model_type`)

The node inspects tensor keys (layer names) and metadata inside the LoRA to determine a detailed model type.

- **SDXL subcategory detection**:
  - Identifies not only plain "SDXL" but also variants such as **Pony V6** and **Illustrious / Animagine**.
  - **Criteria**:
    - Base model name (metadata `ss_sd_model_name`)
    - Specific hashes (e.g. Civitai hash for Pony V6)
    - Keywords in metadata ("pony", "illustrious", etc.)

- **Flux / Qwen / SD1.5 detection**:
  - Based on characteristic tensor key patterns (e.g. `lora_unet_double_blocks` → Flux).

### 3.3. URL retrieval (`_get_metadata_url` / `_get_civitai_url_by_hash`)

Retrieves Civitai and HuggingFace URLs and exposes them as node outputs.

1. **From metadata** (`_get_metadata_url`): Uses embedded URL in the file when present.
2. **API fallback** (`_get_civitai_url_by_hash`):
   - If no Civitai URL is found in metadata, the file’s SHA256 hash is computed.
   - The hash is sent to the **Civitai API** to obtain the model URL.
   - **No extra dependencies**: Implemented with Python’s standard `urllib` (no `requests`), so no additional install is required.

### 3.4. Safe file loading

```python
with safe_open(file_path, framework="np", device="cpu") as f:
```

- **NumPy framework**: Safetensors are opened with `framework="np"` so metadata can be read reliably regardless of PyTorch load state.

---

## 4. Usage

Add the **Universal LoRA Analyzer** node and select the LoRA to analyze.

**Inputs**:
- `lora_name`: Select LoRA file from dropdown (required)
- `manual_path`: Optional. Manually specify LoRA file path (.safetensors)
- `show_training_info`: Whether to show training info (default: on)
- `auto_discover`: When enabled, auto-discovers available LoRAs and uses them

**Outputs**:
- `model_type`: Main category (e.g. `SDXL`, `Flux1`)
- `sub_type`: Subcategory (e.g. `Pony`, `Illustration`, `-`)
- `trigger_words`: List of trigger words
- `base_model`: Base model name
- `analysis_result`: Detailed analysis text
- `lora_info`: Success message etc.
- `civitai_url`: Civitai model URL
- `hf_url`: HuggingFace model URL
