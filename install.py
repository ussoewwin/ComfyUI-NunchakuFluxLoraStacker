"""
ComfyUI-Manager install / update hook.

Installs this pack's requirements.txt into the active ComfyUI Python
environment, then ensures the TensorRT-RTX runtime stack required by the CCSR
TensorRT nodes (`LoadCCSRModelTensorRT` / `CCSR_Upscale_TRT`).

Runtime requirements (mirrors the SeedVR2 TensorRT installer pattern):
  - tensorrt-rtx  : engine runtime (nodes/CCSR/trt_engine.py imports tensorrt_rtx)
  - triton-windows: tensorrt-rtx dependency on Windows
  - onnx / onnxscript / polygraphy: needed to *build* an engine from the
    fp16 checkpoint (scripts/build_ccsr_trt_engine.py); harmless if present,
    and skipped here when already installed.

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
ENGINE_HINT = (
    "ccsr_apply_f16io.rtxplan + ccsr_trt_aux.safetensors in "
    "nodes/CCSR/trt_engines/ (see README 'TensorRT engine nodes')"
)


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
        print(
            "[ComfyUI-NunchakuFluxLoraStacker] The TensorRT nodes need an engine.\n"
            f"  missing: {', '.join(missing)}\n"
            f"  expected location: {ENGINE_DIR}\n"
            f"  Build it from an fp16 CCSR checkpoint with:\n"
            f"    python scripts/build_ccsr_trt_engine.py --checkpoint <real-world_ccsr-fp16.safetensors>\n"
            "  (fp16 / INT8 PyTorch CCSR nodes work without the engine.)",
            flush=True,
        )

    print("[ComfyUI-NunchakuFluxLoraStacker] install.py done.", flush=True)


if __name__ == "__main__":
    main()
