"""
ComfyUI-Manager install / update hook and automated TensorRT installer.
Mirrors the SeedVR2 TensorRT installer pattern (D:\USERFILES\GitHub\ComfyUI-SeedVR2-VideoUpscaler-with-TensorRT-Decoder).

Installs this pack's requirements.txt into the active ComfyUI Python
environment, then ensures the TensorRT-RTX runtime stack and engine artifacts
required by the CCSR TensorRT nodes (`LoadCCSRModelTensorRT` / `CCSR_Upscale_TRT`).

Runtime requirements:
  - tensorrt-rtx  : engine runtime (nodes/CCSR/trt_engine.py imports tensorrt_rtx)
  - triton-windows: tensorrt-rtx dependency on Windows
  - onnx / onnxscript / polygraphy: needed to build/parse TRT/ONNX models.

Everything is installed with --no-deps so this hook never upgrades or disturbs
the CUDA/PyTorch ecosystem underneath ComfyUI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "nodes" / "CCSR" / "trt_engines"

# (import_name, install_spec, min_version) - checked with an import probe,
# installed or upgraded to latest when missing or outdated.
TRT_STACK = [
    ("tensorrt_rtx", "tensorrt-rtx==1.6.1.120", "1.6.1"),
    ("triton", "triton-windows==3.8.0.post28", "3.8.0"),
    ("onnx", "onnx==1.22.0", "1.22.0"),
    ("onnxscript", "onnxscript==0.7.1", "0.7.1"),
    ("polygraphy", "polygraphy==0.53.4", "0.53.4"),
]

# Required CCSR TensorRT artifacts
REQUIRED_FILES = [
    "ccsr_apply_f16io.rtxplan",
    "ccsr_trt_aux.safetensors",
]

# Hugging Face repositories hosting prebuilt engine and weights
HF_REPOS = [
    "https://huggingface.co/ussoewwin/CCSR-TensorRT-Engine/resolve/main",
    "https://huggingface.co/ussoewwin/CCSR-ConvRot-INT8-and-TensorRT-Engine/resolve/main",
]


def log_step(message: str) -> None:
    print(f"\n[CCSR TensorRT Installer] == {message} ==", flush=True)


def pip_install(args: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(f"> {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def install_requirements() -> None:
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        return
    log_step(f"Installing base requirements from {req_file.name}")
    try:
        pip_install(["-r", str(req_file)])
    except Exception as exc:
        print(f"[CCSR] Warning: Failed to install some requirements.txt packages: {exc}", flush=True)


def ensure_package(import_name: str, install_name: str | None = None, no_deps: bool = True, min_version: str | None = None) -> None:
    pkg = install_name or import_name
    try:
        mod = __import__(import_name)
        curr_ver = getattr(mod, "__version__", "")
        if min_version and curr_ver:
            # Parse versions without external dependencies (fallback simple split or packaging)
            needs_upgrade = False
            try:
                from packaging import version
                needs_upgrade = (version.parse(curr_ver) < version.parse(min_version))
            except Exception:
                # Basic string/tuple comparison
                clean_curr = [int(x) for x in curr_ver.split("+")[0].split("-")[0].split(".") if x.isdigit()]
                clean_min = [int(x) for x in min_version.split("+")[0].split("-")[0].split(".") if x.isdigit()]
                needs_upgrade = (clean_curr < clean_min)

            if needs_upgrade:
                log_step(f"Upgrading {import_name} from {curr_ver} to latest ({pkg})")
                args = [pkg, "--upgrade"]
                if no_deps:
                    args.append("--no-deps")
                try:
                    pip_install(args)
                    return
                except Exception as exc:
                    print(f"[CCSR] Warning: Could not upgrade {pkg}: {exc}", flush=True)
                    return

        print(f"[CCSR] {import_name} is already available ({curr_ver or 'ok'}).", flush=True)
        return
    except ImportError:
        pass
    log_step(f"Installing {pkg}")
    args = [pkg]
    if no_deps:
        args.append("--no-deps")
    try:
        pip_install(args)
    except Exception as exc:
        print(f"[CCSR] Warning: Could not install {pkg}: {exc}", flush=True)


def download_file_with_progress(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".tmp")
    try:
        print(f"[CCSR] Downloading {dest_path.name} from {url} ...", flush=True)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out_f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            last_print = 0.0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print > 0.5 or downloaded == total:
                    last_print = now
                    if total > 0:
                        pct = downloaded / total * 100.0
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(f"\r  -> {mb_down:.1f} / {mb_total:.1f} MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        mb_down = downloaded / (1024 * 1024)
                        print(f"\r  -> {mb_down:.1f} MB", end="", flush=True)
            print("", flush=True)
        if tmp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            tmp_path.rename(dest_path)
        return True
    except Exception as exc:
        print(f"\n[CCSR] Download error for {dest_path.name}: {exc}", flush=True)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def sync_or_download_engines() -> None:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    for fname in REQUIRED_FILES:
        target = ENGINE_DIR / fname
        if target.exists() and target.stat().st_size > 1_000_000:
            mb = target.stat().st_size / (1024 * 1024)
            print(f"[CCSR] Engine artifact ready: {fname} ({mb:.1f} MB)", flush=True)
            continue
        log_step(f"Downloading missing engine artifact: {fname}")
        downloaded = False
        for base_url in HF_REPOS:
            url = f"{base_url}/{fname}?download=true"
            if download_file_with_progress(url, target):
                downloaded = True
                break
        if not downloaded or not target.exists() or target.stat().st_size < 1_000_000:
            print(f"[CCSR] Warning: Could not automatically download {fname}.", flush=True)
            print(f"[CCSR] Please manually download from {HF_REPOS[0]} and place in {ENGINE_DIR}", flush=True)


def verify_installation() -> None:
    verify_script = ROOT / "scripts" / "verify_install.py"
    if verify_script.exists():
        log_step("Running readiness verification")
        try:
            subprocess.run([sys.executable, str(verify_script)], check=False)
        except Exception as exc:
            print(f"[CCSR] Verification check warning: {exc}", flush=True)


def main() -> int:
    print("=" * 80, flush=True)
    print("ComfyUI-NunchakuFluxLoraStacker — CCSR TensorRT Auto-Installer", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print("=" * 80, flush=True)

    # 0. PyTorch & CUDA check
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
        cuda_ver = torch.version.cuda
        print(f"PyTorch {torch.__version__} | CUDA: {cuda_avail} ({cuda_ver}) | GPU: {gpu_name}", flush=True)
    except Exception as exc:
        print(f"PyTorch / CUDA status probe: {exc}", flush=True)

    # 1. Base requirements from requirements.txt
    install_requirements()

    # 2. TensorRT-RTX & ONNX stack
    log_step("Ensuring TensorRT-RTX runtime stack")
    for import_name, install_spec, min_ver in TRT_STACK:
        ensure_package(import_name, install_spec, no_deps=True, min_version=min_ver)

    # 3. CCSR TensorRT Engine & Aux weights download
    sync_or_download_engines()

    # 4. Final verification
    verify_installation()

    print("\n" + "=" * 80, flush=True)
    print("ComfyUI-NunchakuFluxLoraStacker CCSR TensorRT installation complete.", flush=True)
    print("=" * 80 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
