# ComfyUI- CCSR upscaler node
## Update:

Models now available here in safetensors format here, by default fp16 is used:

https://huggingface.co/Kijai/ccsr-safetensors/tree/main

There's also a new node that autodownloads them, in which case they go to `ComfyUI/models/CCSR`

![image](https://github.com/kijai/ComfyUI-CCSR/assets/40791699/f7301285-1753-49f7-9828-c8273ee06bb9)

Model loading is also twice as fast as before, and memory use should be bit lower.


The old node simply selects from checkpoints -folder, for backwards compatibility I won't change that.

https://github.com/kijai/ComfyUI-CCSR/assets/40791699/a22306f0-90a4-4a3e-97de-1f795fa8decd

![image](https://github.com/kijai/ComfyUI-CCSR/assets/40791699/5ea77221-441d-41b2-8ede-50c4fd1cfa4f)

This is a simple wrapper node for https://github.com/csslc/CCSR

As such, it's NOT a proper native ComfyUI implementation, so not very efficient and there might be memory issues, tested on 4090 and 4x upscale tiled worked well.



Original model:
The model (https://drive.google.com/drive/folders/1jM1mxDryPk9CTuFTvYcraP2XIVzbPiw_?usp=drive_link) goes to `ComfyUI/models/checkpoints`

I suggest installing with the comfyui-manager:
![image](https://github.com/kijai/ComfyUI-CCSR/assets/40791699/b7214913-4789-4da2-b05a-4ff18e6619b2)


## ConvRot INT8 support (VRAM reduction)

The fp16 nodes (`CCSR_Model_Select` / `DownloadAndLoadCCSRModel`) also accept a
**ConvRot INT8** checkpoint. It quantizes the UNet (ConvRot Linear) and the
ControlNet (plain INT8 Conv2d — ComfyUI has no ConvRot backend for 2D convs),
which cuts checkpoint size from 3.2 GB to ~2.0 GB and reduces VRAM by roughly
1.1 GiB while keeping the surrounding modules (VAE, cond_encoder) in fp16.

Convert the fp16 checkpoint with the hswq script (Conv2d is packed plain INT8):

```
python convert_convrot_int8_ccsr.py real-world_ccsr-fp16.safetensors \
    --out real-world_ccsr_convrot_int8.safetensors
```

Put it under `ComfyUI/models/unet` (or `models/CCSR` / `checkpoints`) and select
it in `CCSR_Model_Select`. The loader auto-detects INT8 (`comfy_quant` markers)
and builds the model with quantized-loading ops.

## TensorRT engine nodes

Prebuilt engine + aux weights + ConvRot INT8 model (Hugging Face): <https://huggingface.co/ussoewwin/CCSR-ConvRot-INT8-and-TensorRT-Engine>


- **Load CCSR Model (TensorRT)** — engine-only loader. Select the engine file
  from `nodes/CCSR/trt_engines/*.rtxplan`; the ControlNet+UNet are executed by
  the TRT engine (built from the fp16 checkpoint, ~24 ms/step vs ~113 ms fp16).
  Aux weights (`ccsr_trt_aux.safetensors`, VAE + cond_encoder) are loaded
  automatically from the same folder, so no full checkpoint is required.
- **CCSR Upscale (TRT)** — same upscale flow as the fp16 node, fixed tile 512
  (latent 64x64) to match the static engine shape; non-matching sizes fall back
  to fp16.

`steps` is the effective diffusion step count: the t_max/t_min band design is
kept, but the schedule is densified so the truncated range still contains
exactly `steps` timesteps.
