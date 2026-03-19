# ComfyUI `torch.AcceleratorError: CUDA error: invalid argument` (surfaces near VAE decode) — mitigation (custom node-side fix)

## Conclusion (what was done)

- **No changes were made to ComfyUI core**; the mitigation is implemented only in the custom node `ComfyUI-NunchakuFluxLoraStacker`.
- Added logic to **force PuLID `id_embeddings`** (carried by `customized_forward`, a `functools.partial` around `pulid_forward`) **to match the executing Nunchaku block’s device/dtype**.
- Added logic to **ensure PuLID’s `pulid_ca` module** (attention/cross-attn module used by PuLID) is moved to the same execution device/dtype before use.
- Changed PuLID weight restoration after LoRA updates to a **safe per-parameter `copy_` restore** (with device/dtype/contiguous alignment), avoiding mixed-device `load_state_dict` that can trigger async CUDA failures.
- This logic is **strictly gated to run only when PuLID is in use**, so other nodes/forwards are not affected.

---

## 1. Error context (reading the user-provided log)

In the provided log excerpt, the flow is:

- `decode` in `ComfyUI\\nodes.py` → enters `VAE.decode()` in `comfy\\sd.py`
- `model_management.load_models_gpu()` unloads models to free VRAM
- During that unload, `self.model.to(device_to)` in `model_patcher.py` is called and the exception surfaces

The exception is:

- `torch.AcceleratorError: CUDA error: invalid argument`
- With the note “CUDA kernel errors might be asynchronously reported …”

### Key point (typical async CUDA failure)

With this class of CUDA errors, it is common that **the kernel that actually failed** and **the API call where the error surfaces** do not match.

In the log, the error surfaces around `VAE.decode()` during “reload/move models to GPU / another device”. However, if an invalid-argument CUDA call happened earlier (custom-node-injected forward, LoRA composition, PuLID residual callback), the failure can surface later at a different CUDA API such as `.to(device)`.

---

## 2. Suspected cause (why it crashes at “VAE decode” even if the cause is earlier)

The primary suspected pattern was: **PuLID `id_embeddings` being passed into the Nunchaku C backend without matching the Nunchaku execution device/dtype.**

Reasons:

- PuLID injects `pulid_forward` into the forward via `partial(..., id_embeddings=...)` (custom node `NunchakuFluxPuLIDApplyV2`).
- On the Nunchaku side, transformer blocks move inputs to `self.device` / `self.dtype` before calling the C backend, but **`id_embeddings` is not explicitly moved to `self.device` / `self.dtype` there**.
- `id_embeddings` is typically created assuming `pipeline_flux_pulid.py` runs on `self.device` / `self.weight_dtype` (PuLID pipeline often assumes GPU execution).
- In ComfyUI, the actual execution device/dtype can vary with VRAM pressure, offload settings, optimization wrappers, etc.

This “**only some tensors remain on a different device/dtype**” situation can trigger `cudaErrorInvalidValue` in C++/CUDA. If that failure is asynchronous, it may surface later at another CUDA API call such as `.to(device)`.

---

## 3. Modified file

This mitigation is implemented **only in the custom node**.

- `wrappers/flux.py`

(In line with the policy of not modifying ComfyUI core `comfy\\...` files.)

---

## 4. Code changes (what was added)

### 4-1. Call the PuLID embedding alignment function right before `customized_forward`

Inside `ComfyFluxWrapper.forward()`, the following call was added right before `self.customized_forward(...)`.

```python
                else:
                    self._ensure_pulid_embedding_device_dtype()
                    out = self.customized_forward(
                        model,
                        hidden_states=img,
                        encoder_hidden_states=context,
                        pooled_projections=y,
                        timestep=timestep,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance if self.config["guidance_embed"] else None,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        **self.forward_kwargs,
                    ).sample
```

This call is added in every branch where `customized_forward` is invoked (both cache/no-cache paths).

### 4-2. `_ensure_pulid_embedding_device_dtype()` — align `id_embeddings` device/dtype

The added function looks like this (key parts only).

```python
    def _ensure_pulid_embedding_device_dtype(self):
        # Strictly gate: only apply this fix for PuLID integration.
        if self.pulid_pipeline is None:
            return
        cf = self.customized_forward
        if not isinstance(cf, functools.partial):
            return
        # Only for PuLID's forward wrapper.
        try:
            if getattr(cf.func, "__name__", None) != "pulid_forward":
                return
        except Exception:
            return
        kw = cf.keywords or {}
        emb = kw.get("id_embeddings", None)
        if not isinstance(emb, torch.Tensor):
            return
        try:
            block0 = self.model.transformer_blocks[0]
            target_device = getattr(block0, "device", emb.device)
            target_dtype = getattr(block0, "dtype", emb.dtype)
        except Exception:
            return

        if emb.device != target_device or emb.dtype != target_dtype or not emb.is_contiguous():
            kw["id_embeddings"] = emb.to(device=target_device, dtype=target_dtype, non_blocking=True).contiguous()
```

### 4-3. `_ensure_pulid_ca_device_dtype()` — align PuLID `pulid_ca` module device/dtype

If PuLID is enabled, this wrapper ensures the PuLID module is on the same device/dtype as the executing Nunchaku block before it is attached:

- Call site: right before `self.model.transformer_blocks[0].pulid_ca = self.pulid_pipeline.pulid_ca`

```python
if self.pulid_pipeline is not None:
    self._ensure_pulid_ca_device_dtype()
    self.model.transformer_blocks[0].pulid_ca = self.pulid_pipeline.pulid_ca
```

The helper is PuLID-only and intentionally conservative: if it cannot reliably infer target device/dtype, it does nothing.

### 4-4. Safer PuLID weight restore after LoRA update

When LoRA composition runs, this wrapper may back up PuLID weights and restore them after `update_lora_params()`.
To avoid mixed-device/mixed-dtype state_dict loads, restoration is done by copying tensors into the existing parameters:

- Move the saved tensor to the **parameter’s device/dtype**
- Ensure it is **contiguous**
- Restore via `param.data.copy_(...)`

---

## 5. Why this helps (and how the gating keeps it safe)

### 5-1. Why it helps (goal)

There is only one goal:

- Prevent **PuLID `id_embeddings`** from reaching the backend while **not matching Nunchaku’s execution device/dtype**.

This is expected to:

- Suppress an earlier “invalid argument” CUDA call that likely happened upstream
- Break the chain where the async error surfaces later during VAE decode load/offload

…and therefore mitigate the crash.

### 5-2. Narrow scope via strict gating

As requested, this logic triggers **only for PuLID**.

The gating conditions are:

- `self.pulid_pipeline is not None` (do nothing if PuLID is not used)
- `self.customized_forward` is a `functools.partial` (do nothing if not)
- `partial.func.__name__ == "pulid_forward"` (do not touch partials unrelated to PuLID)
- `partial.keywords["id_embeddings"]` is a `torch.Tensor` (do nothing if missing)

So it should not affect **normal LoRA-only usage** or other custom nodes’ forward overrides.

---

## 6. Other possible causes and next steps (if it still crashes)

If this does not stop the crash, the next candidates are other operations within the same custom node (not ComfyUI core).

- In `wrappers/flux.py`, the part that **backs up `pulid_ca` weights → `update_lora_params()` → restores**
  - It performs many tensor operations: `clone()` / `load_state_dict()` / manual `param.data.copy_()`
  - If device/dtype/contiguous mismatches or unexpected references occur, this can also become a source of async CUDA errors

If the crash persists even after the above mitigations, the next step is to run with `CUDA_LAUNCH_BLOCKING=1` to make the stack trace point closer to the real failing CUDA call, then inspect which tensor/module is on the wrong device/dtype (PuLID-only logging is recommended).

In that case, the next step is to add minimal logging (PuLID only) to confirm where each tensor lives (device/dtype/contiguity).

---

## 7. How to verify (user side)

1. Restart ComfyUI (fully restart the Python process)
2. Re-run with the same workflow/model/resolution
3. If it still crashes, paste the **full error log** (especially the lines right before `custom_nodes` and the stack around `VAE.decode`)

