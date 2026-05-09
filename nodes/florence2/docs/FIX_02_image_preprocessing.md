# Fix #2: Image Preprocessing Pipeline Compatibility

## Root Cause

When `Florence2Processor.__call__()` passes image processing parameters to `CLIPImageProcessor`,
it was forwarding `None` values directly.
In transformers v4.x, `CLIPImageProcessor` interpreted `None` as "not specified, use defaults."
**In v5.x, `None` is interpreted as "skip this processing step."**

This behavior change triggers the following cascade of failures:

```
Florence2Processor.__call__(do_resize=None, do_rescale=None)
    |
CLIPImageProcessor: do_resize=None -> no resize
CLIPImageProcessor: do_rescale=None -> no float conversion
    |
Input image: original size, still uint8
    |
RuntimeError: Input type (unsigned char) and bias type (struct c10::Half) should be the same
AssertionError: only support square feature maps for now
```

---

## Error Details

### Error 1: Type Mismatch

```
RuntimeError: Input type (unsigned char) and bias type (struct c10::Half) should be the same
```

**Cause**: When `do_rescale=None` is passed, `CLIPImageProcessor` returns pixel values as `[0, 255]` `uint8`.
However, model weights are `float16 (Half)`, causing a type mismatch in PyTorch convolution operations.

Under normal operation, `CLIPImageProcessor` applies `rescale_factor=1/255.0`,
converting `[0, 255]` (uint8) to `[0.0, 1.0]` (float32).

### Error 2: Non-Square Feature Maps

```
AssertionError: only support square feature maps for now
```

**Cause**: When `do_resize=None` is passed, `CLIPImageProcessor` does not resize the image to 768x768.
The original image size (e.g., 1920x1080) is fed directly to the vision tower,
producing non-square feature maps that violate this assertion:

```python
# modeling_florence2.py L2629-2630
h, w = int(num_tokens ** 0.5), int(num_tokens ** 0.5)
assert h * w == num_tokens, 'only support square feature maps for now'
```

---

## Modified File

`processing_florence2.py`

---

## Code Details

### Before (Original)

```python
pixel_values = self.image_processor(
    images,
    do_resize=do_resize,           # <- None is passed
    do_normalize=do_normalize,     # <- None is passed
    image_mean=image_mean,
    image_std=image_std,
    return_tensors=return_tensors,
    input_data_format=input_data_format,
    data_format=data_format,
    resample=resample,
    do_convert_rgb=do_convert_rgb,
)["pixel_values"]
```

With this approach, since all default arguments in `Florence2Processor.__call__()` are `None`,
`do_resize=None`, `do_normalize=None`, etc. are passed to `CLIPImageProcessor`
unless the caller explicitly specifies values.

### After (L253-270)

```python
image_processor_kwargs = {"return_tensors": return_tensors}
if do_resize is not None:
    image_processor_kwargs["do_resize"] = do_resize
if do_normalize is not None:
    image_processor_kwargs["do_normalize"] = do_normalize
if image_mean is not None:
    image_processor_kwargs["image_mean"] = image_mean
if image_std is not None:
    image_processor_kwargs["image_std"] = image_std
if input_data_format is not None:
    image_processor_kwargs["input_data_format"] = input_data_format
if data_format is not None:
    image_processor_kwargs["data_format"] = data_format
if resample is not None:
    image_processor_kwargs["resample"] = resample
if do_convert_rgb is not None:
    image_processor_kwargs["do_convert_rgb"] = do_convert_rgb
pixel_values = self.image_processor(images, **image_processor_kwargs)["pixel_values"]
```

---

## Rationale

By **excluding `None` parameters from the keyword arguments dict**,
`CLIPImageProcessor.__call__()` treats them as unspecified and uses
**default values from its `__init__`**.

Values set during `CLIPImageProcessor` initialization in `load_model()`:

```python
# nodes.py L61-71
image_processor = CLIPImageProcessor(
    do_resize=True,               # <- this is used
    size={"height": 768, "width": 768},
    resample=3,                   # BICUBIC
    do_center_crop=False,
    do_rescale=True,              # <- this is used
    rescale_factor=1/255.0,
    do_normalize=True,            # <- this is used
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
)
```

Unless the caller intentionally overrides:

| Parameter | `__init__` Default | Processing |
|-----------|-------------------|------------|
| `do_resize` | `True` | Resize to 768x768 |
| `do_rescale` | `True` | Convert `[0,255]` to `[0.0, 1.0]` |
| `do_normalize` | `True` | Apply ImageNet normalization |

---

## Call Pattern in `nodes.py`

In `Florence2Run.encode()` (L454), `do_rescale=False` is explicitly specified:

```python
inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)
```

This is because in the ComfyUI pipeline, `F.to_pil_image()` (L453) has already converted the tensor to a PIL Image.
The image is already a `[0, 255]` uint8 PIL Image. With the fixed code,
`do_rescale=False` is explicitly passed, so `CLIPImageProcessor`'s default `True` is properly overridden,
and `False` is correctly used.

In this case, even though `do_rescale=False` means no scaling is applied, there is no type mismatch issue
because the `do_resize=True` processing step converts the PIL Image to a float tensor.

---

## Why This Wasn't a Problem in v4.x

The transformers v4.x `CLIPImageProcessor.__call__()` implementation:

```python
# v4.x behavior (pseudocode)
def __call__(self, images, do_resize=None, ...):
    do_resize = do_resize if do_resize is not None else self.do_resize
    # -> If None, falls back to self.do_resize (=True)
```

In transformers v5.x, this fallback behavior was changed, and `None` is now processed
as "disabled" rather than "unspecified."

---

## Scope of Impact

| File | Changes | Impact |
|------|---------|--------|
| `processing_florence2.py` | 1 location in `__call__` method | Entire image preprocessing flow |

## Test Environment

- transformers 5.4.0
- Tested with various image sizes and aspect ratios
- Florence-2-large / Florence-2-large-ft

## Backward Compatibility

This change only avoids passing `None`, so **it behaves identically in transformers v4.x**.
In v4.x, "not passing the parameter" and "`__init__` default is used" produce the same behavior, with no side effects.
