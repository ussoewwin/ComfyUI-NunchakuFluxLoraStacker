# CCSR (TensorRT) upscaler nodes

TensorRT-accelerated image upscaling nodes leveraging the CCSR (Creative
Content Super-Resolution) architecture, bundled under `nodes/CCSR/` of
**ComfyUI-NunchakuFluxLoraStacker**. They appear under the ComfyUI category
**CCSR**.

Only the **TensorRT** path is shipped: the ControlNet+UNet denoise runs on a
TensorRT-RTX engine, while the VAE / cond_encoder run on PyTorch fp16 from aux
weights. (The fp16 / ConvRot-INT8 PyTorch nodes were removed - TensorRT is
both faster and lighter.)

## Model / engine download

Prebuilt engine, aux weights and the ConvRot INT8 checkpoint (Hugging Face):

<https://huggingface.co/ussoewwin/CCSR-ConvRot-INT8-and-TensorRT-Engine>

| File | Put it in |
|------|-----------|
| `ccsr_apply_f16io.rtxplan` | `nodes/CCSR/trt_engines/` |
| `ccsr_trt_aux.safetensors` | `nodes/CCSR/trt_engines/` |

The engine dropdown in **Load CCSR Model (TensorRT)** auto-lists every
`*.rtxplan` under `nodes/CCSR/trt_engines/`; the aux file is loaded
automatically from the same folder.

> Requires an RTX GPU and the TensorRT-RTX runtime (`tensorrt-rtx`).
> `install.py` installs the runtime stack; the engine itself is not rebuilt
> during install (prebuilt engines are published on Hugging Face).

## Nodes

### Load CCSR Model (TensorRT) — `LoadCCSRModelTensorRT`

Engine-only loader. Select an engine from `nodes/CCSR/trt_engines/*.rtxplan`;
the ControlNet+UNet apply-model runs on the TensorRT engine (~1.4x vs fp16 PyTorch).
Aux weights (`ccsr_trt_aux.safetensors`, VAE + cond_encoder) are loaded
automatically, so **no full checkpoint is required**. Returns **`ccsr_model`**
(`CCSRMODEL`).

### CCSR Upscale (TRT) — `CCSR_Upscale_TRT`

TRT-accelerated upscale with tiled VAE encode/decode and color correction.
Fixed **tile 512** (latent 64x64) to match the static engine shape. Returns
**`upscaled_image`** (`IMAGE`).

`steps` is the effective diffusion step count: the t_max/t_min band design is
kept, but the schedule is densified so the truncated range still contains
exactly `steps` timesteps.

## Usage

1. Install the pack (ComfyUI-Manager runs `install.py`, which ensures the
   TensorRT-RTX runtime stack).
2. Download `ccsr_apply_f16io.rtxplan` + `ccsr_trt_aux.safetensors` from
   Hugging Face and put them in `nodes/CCSR/trt_engines/`.
3. Restart ComfyUI.
4. Build the workflow: **Load CCSR Model (TensorRT)** → **CCSR Upscale (TRT)**,
   feed the image in, run.
5. The log confirms the TRT path (`[CCSR-TRT] ...` lines).
