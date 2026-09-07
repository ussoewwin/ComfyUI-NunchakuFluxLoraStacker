"""Fast, non-rendering readiness check for ComfyUI CCSR TensorRT (ComfyUI-NunchakuFluxLoraStacker)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED_FILES = (
    "ccsr_apply_f16io.rtxplan",
    "ccsr_trt_aux.safetensors",
)

SEARCH_DIRS = [
    ROOT / "nodes" / "CCSR" / "trt_engines",
    ROOT.parents[1] / "models" / "ccsr" / "trt",
]


def main() -> int:
    print("=" * 80)
    print("ComfyUI CCSR TensorRT Readiness Check")
    print(f"Python: {sys.executable}")
    print("=" * 80)

    failures: list[str] = []

    # 1. Module imports
    for module in ("torch", "triton", "tensorrt_rtx", "onnx", "polygraphy"):
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "available")
            print(f"[{module}] {ver}")
        except Exception as exc:
            failures.append(f"cannot import {module}: {exc}")

    # 2. CUDA check
    try:
        import torch
        if not torch.cuda.is_available():
            failures.append("PyTorch cannot access an NVIDIA CUDA GPU")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda
            print(f"CUDA GPU: {gpu_name} (CUDA {cuda_ver})")
    except Exception as exc:
        failures.append(f"PyTorch CUDA verification error: {exc}")

    # 3. Engine & aux files check
    missing_files = []
    found_files = []
    for name in REQUIRED_FILES:
        found = False
        for s_dir in SEARCH_DIRS:
            p = s_dir / name
            if p.exists() and p.stat().st_size > 1_000_000:
                found = True
                found_files.append((name, str(p), p.stat().st_size))
                break
        if not found:
            missing_files.append(name)

    if found_files:
        print("\nTensorRT engine and aux files detected:")
        for name, p, sz in found_files:
            mb = sz / (1024 * 1024)
            print(f" - {name} ({mb:.1f} MB) -> {p}")

    if missing_files:
        print("\nNote: Some CCSR TensorRT engine files are not yet present:")
        for name in missing_files:
            print(f" - {name}")
        print("They can be downloaded automatically via install.py or from:")
        print("  https://huggingface.co/ussoewwin/CCSR-TensorRT-Engine")
        failures.append(f"Missing required TRT files: {', '.join(missing_files)}")

    if failures:
        print("\nInstallation is incomplete or needs attention:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("\nComfyUI CCSR TensorRT installation is fully ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
