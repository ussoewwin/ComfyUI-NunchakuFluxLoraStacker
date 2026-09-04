"""
ComfyUI-Manager install / update hook.

Installs this pack's requirements.txt into the active ComfyUI Python
environment, then ensures the TensorRT-RTX runtime stack required by the CCSR
TensorRT nodes (`LoadCCSRModelTensorRT` / `CCSR_Upscale_TRT`).

Runtime requirements (mirrors the SeedVR2 TensorRT installer pattern):
  - tensorrt-rtx  : engine runtime (nodes/CCSR/trt_engine.py imports tensorrt_rtx)
  - triton-windows: tensorrt-rtx dependency on Windows
  - onnx / onnxscript / polygraphy: needed to *build* an engine from the
    fp16 checkpoint; harmless if present, skipped here when already installed.

Everything is installed with --no-deps so this hook never upgrades the
CUDA/torch ecosystem underneath ComfyUI.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (import_name, install_spec) - checked with an import probe, installed only
# when missing. Pinned to the versions validated on the dev machine.
TRT_STACK = [
    ("tensorrt_rtx", "tensorrt-rtx==1.6.1.120"),
    ("triton", "triton-windows==3.5.1.post24"),
    ("onnx", "onnx==1.22.0"),
    ("onnxscript", "onnxscript==0.7.1"),
    ("polygraphy", "polygraphy==0.53.4"),
]

ENGINE_DIR = ROOT / "nodes" / "CCSR" / "trt_engines"

# Prebuilt engine / aux weights / ConvRot INT8 model (Hugging Face)
HF_REPO = "https://huggingface.co/ussoewwin/CCSR-ConvRot-INT8-and-TensorRT-Engine"
HF_FILES = {
    "ccsr_apply_f16io.rtxplan": "nodes/CCSR/trt_engines/",
    "ccsr_trt_aux.safetensors": "nodes/CCSR/trt_engines/",
    "real-world_ccsr_convrot_int8.safetensors": "ComfyUI/models/unet/ (fp16 path)",
}


def log_step(message: str) -> None:
    print(f"\n[ComfyUI-NunchakuFluxLoraStacker] == {message} ==", flush=True)


def pip_install(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(f"> {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def ensure_package(import_name: str, install_spec: str) -> None:
    try:
        __import__(import_name)
        print(f"[ComfyUI-NunchakuFluxLoraStacker] {install_spec} is already installed.", flush=True)
        return
    except ImportError:
        pass
    log_step(f"Installing {install_spec}")
    pip_install([install_spec, "--no-deps"])


def engine_status() -> tuple[bool, list[str]]:
    """Return (all_present, list of missing engine files)."""
    expected = ["ccsr_apply_f16io.rtxplan", "ccsr_trt_aux.safetensors"]
    missing = [name for name in expected if not (ENGINE_DIR / name).exists()]
    return (len(missing) == 0, missing)


def main() -> None:
    # 1. Base requirements (includes CCSR deps)
    req = ROOT / "requirements.txt"
    log_step(f"installing -r {req.name}")
    pip_install(["-r", str(req)])

    # 2. TensorRT-RTX runtime stack (CCSR TRT nodes)
    log_step("ensuring TensorRT-RTX runtime (CCSR TensorRT nodes)")
    for import_name, install_spec in TRT_STACK:
        ensure_package(import_name, install_spec)

    # 3. Engine presence hint
    all_present, missing = engine_status()
    if all_present:
        log_step("CCSR TRT engine found in nodes/CCSR/trt_engines/")
    else:
        log_step("CCSR TRT engine NOT found")
        lines = [
            "[ComfyUI-NunchakuFluxLoraStacker] The TensorRT nodes need an engine.",
            f"  missing: {', '.join(missing)}",
            f"  expected location: {ENGINE_DIR}",
            "  Download the prebuilt engine + aux weights (and the ConvRot INT8 model) from:",
            f"    {HF_REPO}",
        ]
        for fname, dest in HF_FILES.items():
            lines.append(f"    - {fname} -> {dest}")
        lines.append("  (fp16 / INT8 PyTorch CCSR nodes work without the engine.)")
        print("\n".join(lines), flush=True)

    print("[ComfyUI-NunchakuFluxLoraStacker] install.py done.", flush=True)


if __name__ == "__main__":
    main()
