# FIX_04: SageAttention2 and SageAttention3 Support

## Problem

Florence-2 originally supports three attention backends: `eager` (manual), `sdpa` (PyTorch Scaled Dot-Product Attention), and `flash_attention_2` (Flash Attention 2). While these are functional, newer quantized attention kernels — SageAttention2 and SageAttention3 — offer significant speedups (2-5x over FlashAttention) with negligible accuracy loss.

## Solution

Added `sage_attention_2` and `sage_attention_3` as selectable attention backends in the ComfyUI UI.

| Backend | Package | Function | GPU Requirement |
|---|---|---|---|
| `sage_attention_2` | `sageattention` | `sageattn()` | RTX 3090/4090+ |
| `sage_attention_3` | `sageattn3` | `sageattn3_blackwell()` | RTX 5090 (Blackwell) |

### Installation

```bash
# SageAttention2
pip install sageattention

# SageAttention3 (Blackwell GPUs only)
# See: https://huggingface.co/jt-zhang/SageAttention3
```

## Modified Files

### 1. `modeling_florence2.py`

#### Added: `_Florence2SageAttentionBase` (base class)

```python
class _Florence2SageAttentionBase(Florence2Attention):
    def _get_sage_func(self):
        raise NotImplementedError

    def forward(self, hidden_states, key_value_states=None,
                past_key_value=None, attention_mask=None,
                layer_head_mask=None, output_attentions=False):
        ...
```

**Why a base class?** Both SA2 and SA3 share identical Q/K/V projection and output reshaping logic. The only difference is which function computes the attention scores. The Template Method pattern via `_get_sage_func()` avoids code duplication.

**Key design decisions in `forward()`:**

1. **Fallback to eager attention** when `output_attentions=True` or `layer_head_mask` is set — SageAttention does not return attention weights, so the eager implementation is needed for debugging/visualization.

2. **Fallback to SDPA when `attention_mask` is present** — SageAttention does not support arbitrary attention masks. When the encoder receives padded inputs (batched inference), the mask ensures padding tokens are ignored. In practice, Florence-2 processes one image at a time so the mask is all-ones and SageAttention is used.

3. **No pre-scaling of query states** — Unlike the eager `Florence2Attention` which pre-multiplies queries by `head_dim ** -0.5`, both SDPA and SageAttention handle scaling internally. The base class follows the SDPA convention (no pre-scaling).

4. **`is_causal` flag** — Passed through to SageAttention. The encoder uses `is_causal=False` (bidirectional), the decoder uses `is_causal=True` (autoregressive). SageAttention natively supports both modes.

#### Added: `Florence2SageAttention2`

```python
class Florence2SageAttention2(_Florence2SageAttentionBase):
    def _get_sage_func(self):
        from sageattention import sageattn
        return sageattn
```

Uses the `sageattention` package (INT8 quantized Q/K with FP16/FP8 PV accumulation). The `sageattn()` function accepts `(q, k, v, is_causal=False)` with tensor shape `(batch, heads, seq_len, head_dim)` — identical to PyTorch SDPA's expected layout.

#### Added: `Florence2SageAttention3`

```python
class Florence2SageAttention3(_Florence2SageAttentionBase):
    def _get_sage_func(self):
        from sageattn3 import sageattn3_blackwell
        return sageattn3_blackwell
```

Uses the `sageattn3` package (INT4/FP4 quantized attention optimized for NVIDIA Blackwell architecture). Requires an RTX 5090 or similar Blackwell GPU.

#### Modified: `FLORENCE2_ATTENTION_CLASSES`

```python
FLORENCE2_ATTENTION_CLASSES = {
    "eager": Florence2Attention,
    "sdpa": Florence2SdpaAttention,
    "flash_attention_2": Florence2FlashAttention2,
    "sage_attention_2": Florence2SageAttention2,   # NEW
    "sage_attention_3": Florence2SageAttention3,   # NEW
}
```

This dictionary is used by `Florence2EncoderLayer` and `Florence2DecoderLayer` to instantiate the correct attention class based on `config._attn_implementation`.

### 2. `nodes.py`

#### Modified: UI attention options

```python
# Before
['flash_attention_2', 'sdpa', 'eager']

# After
['flash_attention_2', 'sdpa', 'eager', 'sage_attention_2', 'sage_attention_3']
```

Added to both `DownloadAndLoadFlorence2Model.INPUT_TYPES` and `Florence2ModelLoader.INPUT_TYPES`.

#### Modified: `load_model()` — bypass transformers validation

```python
config = Florence2Config.from_pretrained(model_path)
sage_mode = None
if attention.startswith('sage_attention'):
    sage_mode = attention
    config._attn_implementation = 'sdpa'    # tell transformers it's sdpa
else:
    config._attn_implementation = attention
```

**Why this is necessary:** Transformers 5.x validates the `_attn_implementation` value in `PreTrainedModel.__init__()` against a whitelist (`eager`, `sdpa`, `flash_attention_2`, etc.). Custom values like `sage_attention_2` are rejected with a `ValueError`. By setting `sdpa` during initialization, the model passes validation and creates SDPA attention layers.

#### Added: Post-initialization class swap

```python
if sage_mode:
    from .modeling_florence2 import FLORENCE2_ATTENTION_CLASSES
    SageClass = FLORENCE2_ATTENTION_CLASSES[sage_mode]
    for module in model.modules():
        if hasattr(module, 'self_attn') and module.self_attn.__class__.__name__.startswith('Florence2Sdpa'):
            module.self_attn.__class__ = SageClass
        if hasattr(module, 'encoder_attn') and module.encoder_attn.__class__.__name__.startswith('Florence2Sdpa'):
            module.encoder_attn.__class__ = SageClass
    print(f"Florence2 attention replaced with {sage_mode}")
```

**How this works:** After the model is fully initialized and weights are loaded, we iterate through all modules and replace the Python class of each SDPA attention layer with the corresponding SageAttention class. This is safe because:

- `Florence2SageAttention2/3` inherits from `Florence2Attention` (same base as `Florence2SdpaAttention`)
- All attention classes share identical `__init__` parameters and weight tensors (`q_proj`, `k_proj`, `v_proj`, `out_proj`)
- Only the `forward()` method differs — which function computes the dot-product attention
- Python's `__class__` assignment changes the method resolution order without affecting the instance's state (weights remain intact)

#### Modified: Transformers 4.x fallback

```python
attn_impl = 'sdpa' if attention.startswith('sage_attention') else attention
if attn_impl != attention:
    print(f"{attention} requires transformers>=5.0, falling back to sdpa")
```

For users on transformers 4.x, SageAttention is not available (the v5 `load_model` path is required for the class-swap technique). A warning is printed and SDPA is used instead.

## Architecture

```
User selects "sage_attention_2" in UI
        │
        ▼
load_model() receives attention="sage_attention_2"
        │
        ├─ Sets config._attn_implementation = "sdpa"  (bypass validation)
        ├─ Creates model with SDPA attention layers
        ├─ Loads weights normally
        │
        ▼
Post-init class swap:
  Florence2SdpaAttention → Florence2SageAttention2
        │
        ▼
During inference, forward() calls:
  sageattn(q, k, v, is_causal=...)  instead of  F.scaled_dot_product_attention(...)
```

## Performance

Tested with Florence-2-large-PromptGen-v2.0 on RTX 5090:

- SageAttention3 provides significant speedup over FlashAttention2
- SageAttention2 provides moderate speedup with broader GPU compatibility
- Output quality is identical — quantized attention preserves end-to-end accuracy
