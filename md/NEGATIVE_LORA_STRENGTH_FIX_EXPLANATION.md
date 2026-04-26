# Negative LoRA Strength Support Fix (V2 Loaders)

## Purpose of This Fix

This fix enables all V2 LoRA loader nodes in this repository to accept **negative LoRA strength values** (e.g. `-1.0`), matching the behavior of the standard ComfyUI LoRA loader.

The user-reported issue was:

- V2 loaders in this repository did not allow negative values in the UI.
- Standard ComfyUI LoRA loader does allow negative values.

So the target was to make this repository behave like standard ComfyUI.

---

## Standard ComfyUI Reference (Behavior Baseline)

In standard ComfyUI (`D:/USERFILES/ComfyUI/ComfyUI/nodes.py`), `LoraLoader` defines:

- `strength_model`: `min=-100.0`, `max=100.0`
- `strength_clip`: `min=-100.0`, `max=100.0`

That explicitly permits negative strengths.

---

## Files Modified

Exactly 3 files were modified:

1. `nodes/lora/flux_v2.py`
2. `nodes/lora/standard.py`
3. `nodes/lora/sdnq.py`

No other files were changed for this fix.

---

## Complete Code Changes and Meaning

Below are the exact changed lines (all of them), with before/after and explanation.

### 1) `nodes/lora/flux_v2.py`

#### Before

```python
inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
```

#### After

```python
inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
```

#### Meaning

- Adds lower/upper bounds to the UI parameter.
- Most importantly, `min: -100.0` allows entering negative strengths.
- Keeps existing precision (`step: 0.001`), so fine-grained control is preserved.

---

### 2) `nodes/lora/standard.py`

#### Before

```python
inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
```

#### After

```python
inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
```

#### Meaning

- Same fix applied to the standard-model V2 LoRA stacker.
- Negative LoRA weights are now accepted by the node input definition.
- Runtime logic already passed strength through as-is; the missing part was the input range declaration.

---

### 3) `nodes/lora/sdnq.py`

#### Before

```python
inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
```

#### After

```python
inputs["optional"][f"lora_wt_{i}"] = ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.001, "tooltip": f"LoRA {i} Strength"})
```

#### Meaning

- Same range fix for SDNQ (Diffusers-based) V2 loader.
- Negative adapter weights are now possible through the node UI definition.

---

## Why This Fix Was Needed

The main issue was in **input schema**, not in downstream loading logic:

- Each loader read `lora_wt_X` and used it directly (`strength = lora_wt` or equivalent).
- But the input field lacked explicit negative range metadata.
- In ComfyUI node UI behavior, float input definitions strongly influence valid user entry range.

By adding `min: -100.0`, negative values are now explicitly valid and accepted.

---

## What Was Not Changed

To keep the fix minimal and safe, the following were intentionally unchanged:

- LoRA application algorithm (no model math changes).
- Deduplication logic.
- Zero-strength skip behavior (`abs(strength) < 1e-5`).
- Node generation strategy and class names.
- Step size (`0.001`) in this repository.

Note:

- Standard ComfyUI uses `step: 0.01`.
- This repository keeps `step: 0.001` for finer control.
- This difference does not affect negative support.

---

## Compatibility Notes

- Legacy node `nodes/lora/flux.py` already supported negative values (`min: -100.0` / `max: 100.0`), so it required no change.
- This fix aligns all V2 loaders with standard ComfyUI negative-strength capability.

---

## Validation Checklist

After restarting ComfyUI, each of the following should work:

1. `FLUX LoRA Loader V2`:
   - Set any `lora_wt_X` to a negative value (example: `-1.0`)
2. `LoRA Stacker V2`:
   - Set any `lora_wt_X` to a negative value
3. `SDNQ LoRA Stacker V2`:
   - Set any `lora_wt_X` to a negative value

Expected result:

- UI accepts negative values.
- Node execution proceeds without input-range rejection.

---

## Summary

This was a focused schema-level compatibility fix:

- Added `min/max` bounds (`-100.0` to `100.0`) to `lora_wt_X` in all V2 LoRA loaders.
- Made V2 loaders support negative strengths, consistent with standard ComfyUI LoRA loader behavior.
- Kept runtime logic and behavior otherwise unchanged.
