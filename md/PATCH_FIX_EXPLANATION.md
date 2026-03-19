# ComfyUI-NunchakuFluxLoraStacker Patch Fix — Full Explanation (English)

## Release-notes summary (v1.26 / v1.27)

**Scope**: `Model Patch Loader (ModelPatchLoaderCustom)`. These are fixes for issues that appeared after ComfyUI updates.

| Version | Problem | Fix |
|---------|---------|-----|
| **v1.26** | Z-Image patch loading bug: embedder weights were not loaded, leading to matmul shape errors at inference. | Infer `control_in_dim` from the checkpoint and include checkpoint-only keys in the `load_state_dict` path to support lazy-init layers. |
| **v1.27** | CPU offload execution drift/inconsistency for Z-Image patches. | Ensure CPU offload is respected during execution by patching `ZImageControlPatch` behavior and wrapping patch execution with correct `load_device/offload_device`. |

**What to do**: Restart ComfyUI, then run `Model Patch Loader` once to reload the patch, and re-run sampling.

---

## 1. The error (Z-Image matmul shape mismatch)

### 1.1 Message

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (6110x132 and 3840x132)
```

### 1.2 Where it happens

- File: `comfy/ldm/lumina/controlnet.py` (`ZImage_Control.forward`)
- Code path: Control image is processed and passed through `control_all_x_embedder` (Linear layer)
- Call site: `comfy_extras/nodes_model_patch.py` (`ZImageControlPatch.__call__`) during the sampling noise-refiner flow

### 1.3 Log warning (example)

```text
[ModelPatchLoaderCustom] Warning: 51 keys not found in model (excluded to match latest model structure)
  - control_all_x_embedder.2-1.bias
  - control_all_x_embedder.2-1.weight
  - control_layers.0.adaLN_modulation.0.bias
  ...
```

When this happens, embedder weights (and related parameters) were excluded from the loaded state dict, so the model kept an incompatible in-feature layout.

---

## 2. Cause (why the embedder ends up incompatible)

### 2.1 Direct cause (matrix shapes)

- Input control tensor is flattened into 2x2 patches with shape `(batch, num_patches, 4*C)`.
- In the failing case, `C = 33`, so the Linear layer receives **132**-dim features per patch (`4*33=132`).
- The checkpoint Linear weight is compatible with `in_features=132` (shape `(3840, 132)`).
- However, the model ended up using weights/configuration compatible with a different `in_features` (e.g. 64-dim version or uninitialized/lazy weights), causing the matmul mismatch.

### 2.2 Why the correct 132-dim embedder weights were not loaded

Two main factors:

1. **`control_in_dim` mismatch** between the config used to construct `ZImage_Control` and the checkpoint’s expected channel layout.
2. **Lazy-init / Windows/aimdo behavior**: some Linear parameters may not appear in `model.state_dict()` until load-time dispatch runs. If the loader filters “keys not in model.state_dict()”, those checkpoint-only keys may be dropped and never assigned.

---

## 3. Fix overview (what changed)

### Fix 1 (v1.26): infer `control_in_dim` from the checkpoint

In `nodes/misc_v2.py`, inside the Z-Image branch (`'control_all_x_embedder.2-1.weight' in sd`), the loader computes:

- `embedder_in = sd["control_all_x_embedder.2-1.weight"].shape[1]`
- `expected_channels = embedder_in // 4`
- if `additional_in_dim` is present: `control_in_dim = expected_channels - additional_in_dim`
- else: `control_in_dim = expected_channels`

This ensures the constructed `ZImage_Control` embedder matches the checkpoint layout (avoids the matmul shape error).

### Fix 2 (v1.26): keep checkpoint-only keys for lazy-init load

Also in `nodes/misc_v2.py`, the state-dict filtering logic was changed to:

- Exclude only **size mismatches**.
- Do **not** exclude keys that are missing from `model.state_dict()` (needed for lazy-init parameters).

This allows `model.load_state_dict(..., strict=False)` to route those parameters to the correct submodules during `_load_from_state_dict`.

---

## 4. v1.27: CPU offload + `ZImageControlPatch` execution fix

Even when a patch is constructed for CPU, ComfyUI may still call into patch execution paths that attempt to move the model to GPU.

In `nodes/misc_v2.py`, `_model_patch_cpu_offload_apply()` patches:

- `comfy_extras.nodes_model_patch.ZImageControlPatch.to`
- `comfy_extras.nodes_model_patch.ZImageControlPatch.__call__`

so that when the patcher’s `load_device` is CPU:

- `.to(cuda)` requests are ignored (for the patch instance)
- inputs are moved to CPU for the patch computation
- outputs are applied back to the original device as needed

---

## 5. User checklist

1. Restart ComfyUI.
2. Re-run your workflow so `Model Patch Loader` reloads the patch.
3. If the error persists, ensure the same ComfyUI version/build is used and re-check logs for:
   - `control_in_dim`/embedder mismatch behavior
   - `keys loaded via state_dict (e.g. lazy-init layers)` messages

