<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/blob/main/zhmd/v2.0.4.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# v2.0.4 — CCSR TensorRT Automated Installer & Stack Modernization

This release modernizes the CCSR TensorRT acceleration pipeline by introducing an automated one-click installation framework modeled after SeedVR2, consolidating all dependencies into the root `requirements.txt`, upgrading to the latest PyPI stable runtime stack, and integrating automatic Hugging Face engine streaming downloads alongside comprehensive environment verification.

---

## 1. Summary of Changes

| Area | v2.0.3 and earlier | v2.0.4 |
|---|---|---|
| **One-Click Installation** | Required manual `python install.py` execution or manager install | Added root one-click `Install TensorRT CCSR.bat` and PowerShell automation `scripts/install.ps1` |
| **Dependency Management** | Split requirements without automated upgrade probes | Consolidated all dependencies into root `requirements.txt` with `--no-deps` protection and auto-upgrade detection in `install.py` |
| **Triton Runtime** | Older / unpinned Triton packages | Upgraded to the latest PyPI build: `triton-windows==3.8.0.post28` |
| **ONNX & TensorRT Stack** | Unpinned auxiliary libraries | Pinned `tensorrt-rtx==1.6.1.120`, `onnx==1.22.0`, `onnxscript==0.7.1`, `polygraphy==0.53.4` |
| **Engine Distribution** | Threw warning on missing engine requiring manual download | Automatic streaming download for `ccsr_apply_f16io.rtxplan` and `ccsr_trt_aux.safetensors` with integrity checks |
| **Verification & Safeguards** | Import errors caused unhandled node crashes | Added comprehensive `scripts/verify_install.py` and graceful runtime missing guards in `LoadCCSRModelTensorRT` |
| **Logging** | Standard console stream only | Full execution logging recorded to `outputs/install.log` |

---

## 2. Automated Installer Architecture (SeedVR2 Pattern)

To provide a zero-intervention setup experience for Windows users and prevent CUDA/PyTorch package conflicts:

### 2-1. `Install TensorRT CCSR.bat` & `scripts/install.ps1`
- **Environment Auto-Detection**: Automatically searches for the embedded ComfyUI Python environment (`python_embeded\python.exe`), active Virtualenvs (`$env:VIRTUAL_ENV`), or PATH-registered Python instances.
- **Pre-Installation Probing**: Verifies PyTorch and CUDA runtime status before executing pip commands, displaying PyTorch version, CUDA availability, and device name.
- **Safe Dependency Installation**: Uses `--no-deps` for TensorRT runtime packages, ensuring the underlying PyTorch/CUDA wheels in ComfyUI remain completely untouched.
- **Engine Streaming Download**: Automatically fetches missing prebuilt engines and auxiliary weights directly into `nodes/CCSR/trt_engines/`.
- **Readiness Verification**: Executes `scripts/verify_install.py` to test actual TensorRT context creation before creating the `.installed_ccsr_trt` completion marker.

---

## 3. Dependency Modernization & Consolidation

All dependencies are unified directly into the repository root `requirements.txt`:

```txt
# CCSR TensorRT upscaler stack (nodes/CCSR)
triton-windows==3.8.0.post28
tensorrt-rtx==1.6.1.120
onnx==1.22.0
onnxscript==0.7.1
polygraphy==0.53.4
```

- **`triton-windows==3.8.0.post28`**: Latest PyPI release of the Windows Triton wheel, fully compatible with TensorRT-RTX 1.6.1 JIT compilation.
- **Auto-Upgrade Logic**: `install.py` inspects installed module versions; if an outdated Triton build (such as 3.5.x or 3.6.x) is detected, it is upgraded to `3.8.0.post28`.
- **Validated Stack Versions**: Ensures compatibility with `onnx==1.22.0`, `onnxscript==0.7.1`, and `polygraphy==0.53.4`.

---

## 4. Automatic Hugging Face Engine Streaming Download

CCSR TensorRT requires two model files (~2.2 GB total):
1. **`ccsr_apply_f16io.rtxplan`**: Fused ControlNet + UNet static-shape (512x512 tile / 64x64 latent) TensorRT-RTX engine.
2. **`ccsr_trt_aux.safetensors`**: Companion auxiliary weights (VAE encoder/decoder and conditioning text encoder).

`install.py` automatically downloads any missing files from the Hugging Face repository (`ussoewwin/CCSR-TensorRT-Engine`) directly into `nodes/CCSR/trt_engines/` with streaming chunk verification.

---

## 5. File Modification Breakdown

| File | Type | Description |
|---|---|---|
| `Install TensorRT CCSR.bat` | New | Root one-click Windows batch entry point |
| `scripts/install.ps1` | New | PowerShell automation pipeline for Python discovery and dependency deployment |
| `scripts/verify_install.py` | New | Diagnostic script verifying PyTorch, CUDA, TRT-RTX, Triton, and engine artifacts |
| `install.py` | Overhaul | Automated installer implementing SeedVR2 architecture with `--no-deps` and engine streaming |
| `requirements.txt` | Update | Consolidated base dependencies and the full TensorRT runtime stack |
| `nodes/CCSR/nodes.py` | Update | Added graceful import guards to `LoadCCSRModelTensorRT` with actionable instructions |
| `README.md` / `zhmd/README.md` | Update | Documented automated installation, one-click batch setup, and updated requirements |
| `md/CHANGELOG.md` / `zhmd/CHANGELOG.md` | Update | Added v2.0.4 release history and documentation links |
| `pyproject.toml` / `__init__.py` | Update | Bumped package version to 2.0.4 |

---

## 6. How to Install & Verify

### Method 1: One-Click Batch (Recommended for Windows)
Double-click **`Install TensorRT CCSR.bat`** in the repository root. The script handles environment discovery, package installation, engine downloading, and verification automatically.

### Method 2: ComfyUI-Manager
Installing or updating through ComfyUI-Manager automatically triggers `install.py` in the background.

### Verification Command
Run the diagnostic script anytime in your ComfyUI environment:
```bash
python scripts/verify_install.py
```
All green `[OK]` checks confirm that the CCSR TensorRT pipeline is ready to run in ComfyUI (~1.4x faster upscale compared to PyTorch fp16).
