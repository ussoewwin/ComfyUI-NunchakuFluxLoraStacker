# Fix #1: transformers 5.x API Compatibility & Flash Attention 2 Support

## Root Cause

When the `transformers` library was upgraded from v4.x to v5.x, several breaking changes were introduced to internal APIs.
The Florence-2 custom model code (`modeling_florence2.py`) relied on transformers v4.x internal implementations,
causing **crashes or invalid state during model initialization/loading** in v5.x environments.

This issue breaks down into four sub-problems:

| # | Issue | Location | Symptom |
|---|-------|----------|---------|
| A | `_tie_or_clone_weights()` removed | `_tie_weights()` | `AttributeError` / broken weight sharing |
| B | `post_init()` behavior change | `__init__()` | Hang/crash during initialization |
| C | `is_flash_attn_greater_or_equal_2_10` function removed | import statement | `ImportError` |
| D | `_supports_flash_attn` attribute newly required | Model init validation | `ValueError: does not support Flash Attention 2` |

---

## A. Removal of `_tie_or_clone_weights()`

### Background

In BART-based models, the encoder, decoder, and lm_head share the same embedding weights via "weight tying."
In transformers v4.x, this was achieved through a utility method `PreTrainedModel._tie_or_clone_weights()`.
**In v5.x, this method was removed.**

### Modified File

`modeling_florence2.py`

### Changes

**`Florence2LanguageModel._tie_weights()`** (L1964-1971):

```python
def _tie_weights(self):
    if self.config.tie_word_embeddings:
        if hasattr(self, '_tie_or_clone_weights'):
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
        else:
            self.encoder.embed_tokens.weight = self.shared.weight
            self.decoder.embed_tokens.weight = self.shared.weight
```

**`Florence2LanguageForConditionalGeneration._tie_weights()`** (L2091-2100):

```python
def _tie_weights(self):
    if self.config.tie_word_embeddings:
        if hasattr(self, '_tie_or_clone_weights'):
            self._tie_or_clone_weights(self.model.encoder.embed_tokens, self.model.shared)
            self._tie_or_clone_weights(self.model.decoder.embed_tokens, self.model.shared)
            self._tie_or_clone_weights(self.lm_head, self.model.shared)
        else:
            self.model.encoder.embed_tokens.weight = self.model.shared.weight
            self.model.decoder.embed_tokens.weight = self.model.shared.weight
            self.lm_head.weight = self.model.shared.weight
```

### Rationale

Uses `hasattr` to check for method existence: v4.x uses `_tie_or_clone_weights()` as before,
while v5.x falls back to direct `.weight` attribute assignment.
This maintains **backward compatibility with both versions**.

When weight sharing is broken, the encoder/decoder/lm_head become independent parameters,
fundamentally corrupting inference results (producing garbage or empty output).

---

## B. `post_init()` Behavior Change

### Background

`PreTrainedModel.post_init()` is a method that performs weight initialization and weight tying after construction.
In transformers v5.x, its internal implementation changed, with stricter validation and different timing for `_tie_weights()` calls.

Florence-2 creates models inside `accelerate`'s `init_empty_weights` context,
then manually assigns weights afterward.
The v5.x `post_init()` is incompatible with this pattern, running invalid validation on empty tensors.

### Modified File

`modeling_florence2.py`

### Changes

**`Florence2LanguageModel.__init__()`** (L1961-1962):

```python
# Initialize weights and apply final processing
if not version.parse(transformers.__version__) >= version.parse('5.0.0'):
    self.post_init()
```

**`Florence2LanguageForConditionalGeneration.__init__()`** (L2087-2089):

```python
# Initialize weights and apply final processing
if not version.parse(transformers.__version__) >= version.parse('5.0.0'):
    self.post_init()
```

The `_tied_weights_keys` class attribute is also conditionally defined (L1948-1949, L2077-2078):

```python
class Florence2LanguageModel(Florence2LanguagePreTrainedModel):
    if not version.parse(transformers.__version__) >= version.parse('5.0.0'):
        _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
```

### Rationale

On v5.x, `post_init()` is skipped; weight tying is instead handled manually
by calling `model.language_model.tie_weights()` in the `load_model()` function.
`_tied_weights_keys` is unnecessary for the v5.x load path (which doesn't use `from_pretrained()`),
and its presence can cause inconsistencies, so it is conditionally excluded.

---

## C. Removal of `is_flash_attn_greater_or_equal_2_10`

### Background

In transformers v4.x, a dedicated function `is_flash_attn_greater_or_equal_2_10` existed in `transformers.utils`
to check whether Flash Attention 2.10 or higher was available.
In v5.x, this was replaced by a generic `is_flash_attn_greater_or_equal(version_string)`,
and the old function was removed.

### Error Message

```
ImportError: cannot import name 'is_flash_attn_greater_or_equal_2_10' from 'transformers.utils'
```

### Modified File

`modeling_florence2.py`

### Changes (L44-53)

```python
try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10
except ImportError:
    try:
        from transformers.utils import is_flash_attn_greater_or_equal
        def is_flash_attn_greater_or_equal_2_10():
            return is_flash_attn_greater_or_equal("2.10")
    except ImportError:
        def is_flash_attn_greater_or_equal_2_10():
            return True
```

### Rationale

Three-level fallback:
1. v4.x: Use the original function as-is
2. v5.x: Use the new generic function with `"2.10"` for equivalent behavior
3. Neither available: Return `True` (assume Flash Attention is installed)

---

## D. New `_supports_flash_attn` Class Attribute Requirement

### Background

In transformers v5.x, attention implementation validation was tightened during model initialization.
When `attn_implementation="flash_attention_2"` is specified, the model class must have
`_supports_flash_attn = True` (newly introduced in v5.x) set, or a `ValueError` is raised.

In v4.x, only `_supports_flash_attn_2 = True` was sufficient.
In v5.x, `_supports_flash_attn = True` is additionally required.

### Error Message

```
ValueError: Florence2ForConditionalGeneration does not support Flash Attention 2 yet.
```

### Modified File

`modeling_florence2.py`

### Changes

**`Florence2LanguagePreTrainedModel`** (L1435-1437):

```python
_supports_flash_attn_2 = True   # v4.x compatibility
_supports_flash_attn = True     # v5.x new requirement
_supports_sdpa = True
```

**`Florence2PreTrainedModel`** (L2366-2368):

```python
_supports_flash_attn_2 = True   # v4.x compatibility
_supports_flash_attn = True     # v5.x new requirement
_supports_sdpa = True
```

### Rationale

Florence-2 internally supports both Flash Attention 2 and SDPA.
Adding these class attributes to **both PreTrainedModel base classes** allows
the model to pass transformers v5.x validation.

---

## Additional Fix: `configuration_florence2.py` - `forced_bos_token_id` Ordering

### Background

`Florence2LanguageConfig.__init__()` set `forced_bos_token_id`, but
`super().__init__(**kwargs)` was overwriting this value.
In transformers v5.x, the internal handling in `PretrainedConfig.__init__()` changed,
making this overwrite problem manifest.

Even when config.json specifies `forced_bos_token_id: 0`,
the value could become `None` or a different value after `super().__init__()`.

### Modified File

`configuration_florence2.py`

### Changes (L253-266)

```python
forced_bos = kwargs.pop('forced_bos_token_id', bos_token_id)

super().__init__(
    num_labels=num_labels,
    pad_token_id=pad_token_id,
    bos_token_id=bos_token_id,
    eos_token_id=eos_token_id,
    is_encoder_decoder=is_encoder_decoder,
    decoder_start_token_id=decoder_start_token_id,
    forced_eos_token_id=forced_eos_token_id,
    **kwargs,
)

self.forced_bos_token_id = forced_bos
```

### Rationale

Extract `forced_bos_token_id` from `kwargs` first (via `pop`),
then explicitly set it **after** `super().__init__()`.
This ensures the value from config.json is reliably preserved.

`forced_bos_token_id` forces the first token in the decoder output.
If this value is incorrect, generation results are completely broken.

---

## Additional Fix: `nodes.py` - `generation_config` Consistency

### Background

In transformers v5.x, the default value handling when `GenerationMixin.generate()` references `generation_config` was changed.
Models created with `init_empty_weights` may not have `generation_config` properly initialized.

### Modified File

`nodes.py`

### Changes - `load_model()` function (L78-91)

```python
if len(tokenizer) != model.language_model.model.shared.num_embeddings:
    model.resize_token_embeddings(len(tokenizer))
    model.language_model.tie_weights()

# Ensure generation_config has correct values from the language model config
lang_config = config.text_config
if hasattr(model.language_model, 'generation_config'):
    gen_cfg = model.language_model.generation_config
    gen_cfg.decoder_start_token_id = getattr(lang_config, 'decoder_start_token_id', 2)
    gen_cfg.eos_token_id = getattr(lang_config, 'eos_token_id', 2)
    gen_cfg.pad_token_id = getattr(lang_config, 'pad_token_id', 1)
    gen_cfg.forced_bos_token_id = getattr(lang_config, 'forced_bos_token_id', 0)
    gen_cfg.forced_eos_token_id = getattr(lang_config, 'forced_eos_token_id', 2)
```

### Rationale

Two operations are added:

1. **Tokenizer size check**: After adding special tokens (`<loc_0>` through `<loc_999>`, etc.),
   if the tokenizer vocabulary size doesn't match the model's embedding size,
   `resize_token_embeddings()` expands it and `tie_weights()` is re-executed.

2. **Explicit `generation_config` setup**: `decoder_start_token_id`, `eos_token_id`, `pad_token_id`,
   `forced_bos_token_id`, and `forced_eos_token_id` are manually set from the config.
   Incorrect values here cause issues like decoding failing to start or failing to stop.

---

## Scope of Impact

| File | Changes | Impact |
|------|---------|--------|
| `modeling_florence2.py` | 6 locations | Model init, weight tying, FA2 support |
| `configuration_florence2.py` | 1 location | Config parameter accuracy |
| `nodes.py` | 1 location | Post-load consistency |

## Test Environment

- transformers 5.4.0
- PyTorch 2.x (CUDA)
- Tested with Florence-2-large / Florence-2-large-ft

## Backward Compatibility

All fixes are implemented using `hasattr` checks or `version.parse()` branching,
and **remain fully functional in transformers v4.x environments**.
