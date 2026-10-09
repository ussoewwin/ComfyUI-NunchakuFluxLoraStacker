"""
ComfyUI-Manager install / update hook and automated TensorRT installer.
Mirrors the SeedVR2 TensorRT installer pattern.

Installs this pack's requirements.txt into the active ComfyUI Python
environment, then ensures the TensorRT-RTX runtime stack and engine artifacts
required by the CCSR TensorRT nodes (`LoadCCSRModelTensorRT` / `CCSR_Upscale_TRT`).

Runtime requirements:
  - tensorrt-rtx  : engine runtime (nodes/CCSR/trt_engine.py imports tensorrt_rtx)
  - triton-windows: tensorrt-rtx dependency on Windows
  - onnx / onnxscript / polygraphy: needed to build/parse TRT/ONNX models.

The TensorRT stack packages are ensured with --no-deps so this hook never
upgrades or disturbs the CUDA/PyTorch ecosystem underneath ComfyUI; the base
requirements.txt installs normally (it never contains torch/CUDA wheels).

Usage:
  python install.py                 full install: runtime stack + engine artifacts + verify
  python install.py --skip-engine   runtime stack only (no engine download / validation)
  python install.py --repair        force-reinstall the runtime stack and deep-verify (SHA256)
                                    the engine artifacts, re-downloading any that fail

Engine artifacts are downloaded from Hugging Face with resume support, retries
and SHA256 verification against the published files (see ENGINE_SPECS below).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE_DIR = ROOT / "nodes" / "CCSR" / "trt_engines"

# (import_name, install_spec, min_version) - checked with an import probe,
# installed or upgraded to latest when missing or outdated.
TRT_STACK = [
    ("tensorrt_rtx", "tensorrt-rtx==1.6.1.120", "1.6.1"),
    ("triton", "triton-windows" if sys.platform.startswith("win") else "triton", None),
    ("onnx", "onnx==1.22.0", "1.22.0"),
    ("onnxscript", "onnxscript==0.7.1", "0.7.1"),
    ("polygraphy", "polygraphy==0.53.4", "0.53.4"),
]

# Required CCSR TensorRT artifacts with the expected size and SHA256 of the
# published files (ussoewwin/CCSR-TensorRT-Engine on Hugging Face).
ENGINE_SPECS = {
    "ccsr_apply_f16io.rtxplan": {
        "size": 2_485_444_548,
        "sha256": "c8bbe29106e42676e03648bddb9af0fb3e096e88106617ce0936737ac86fca43",
    },
    "ccsr_trt_aux.safetensors": {
        "size": 235_677_486,
        "sha256": "29bc16c259750ffa39c8a51b83a853048386382238a2ea9f142a67e084ac65ba",
    },
}
REQUIRED_FILES = list(ENGINE_SPECS.keys())

# Hugging Face repositories hosting the prebuilt engine and weights
HF_REPOS = [
    "https://huggingface.co/ussoewwin/CCSR-TensorRT-Engine/resolve/main",
    "https://huggingface.co/ussoewwin/CCSR-ConvRot-INT8-and-TensorRT-Engine/resolve/main",
]

# Download tuning (same defaults as the SeedVR2 installer)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024   # streaming chunk for the download loop
HASH_CHUNK_SIZE = 8 * 1024 * 1024   # chunk used when computing SHA256
DOWNLOAD_MAX_RETRIES = 3            # attempts per repository
DOWNLOAD_RETRY_DELAY = 2            # seconds (x attempt number)
DOWNLOAD_TIMEOUT = 60               # socket timeout in seconds


def log_step(message: str) -> None:
    print(f"\n[CCSR TensorRT Installer] == {message} ==", flush=True)


def fmt_size(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GB"
    return f"{num_bytes / (1024 ** 2):.1f} MB"


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


def ensure_package(import_name: str, install_name: str | None = None, no_deps: bool = True, min_version: str | None = None, force: bool = False) -> None:
    pkg = install_name or import_name

    if force:
        log_step(f"Reinstalling {pkg} (repair mode)")
        args = [pkg, "--upgrade", "--force-reinstall"]
        if no_deps:
            args.append("--no-deps")
        try:
            pip_install(args)
        except Exception as exc:
            print(f"[CCSR] Warning: Could not reinstall {pkg}: {exc}", flush=True)
        return

    try:
        mod = __import__(import_name)
        curr_ver = getattr(mod, "__version__", "")
        if min_version and curr_ver:
            # Version check; `packaging` is used when available, otherwise a
            # plain numeric comparison of the leading version components.
            needs_upgrade = False
            try:
                from packaging import version
                needs_upgrade = (version.parse(curr_ver) < version.parse(min_version))
            except Exception:
                clean_curr = [int(x) for x in curr_ver.split("+")[0].split("-")[0].split(".") if x.isdigit()]
                clean_min = [int(x) for x in min_version.split("+")[0].split("-")[0].split(".") if x.isdigit()]
                needs_upgrade = (clean_curr < clean_min)

            if needs_upgrade:
                log_step(f"Upgrading {import_name} from {curr_ver} to {pkg}")
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


def sha256_of_file(path: Path, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def part_path_for(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def validate_engine_file(path: Path, filename: str, deep: bool = False) -> bool:
    """Return True when the artifact matches the published size (and SHA256 when deep)."""
    spec = ENGINE_SPECS[filename]
    try:
        if not path.exists():
            return False
        if path.stat().st_size != spec["size"]:
            return False
        if deep and sha256_of_file(path) != spec["sha256"]:
            return False
        return True
    except OSError:
        return False


def download_file_with_progress(url: str, dest_path: Path) -> bool:
    """Stream `url` into `dest_path`; keeps `<dest>.part` on failure so a retry
    can resume with a Range request."""
    part = part_path_for(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    existing = part.stat().st_size if part.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
            status = resp.getcode()
            if existing > 0 and status != 206:
                # Server did not honor the resume request - start over.
                print(f"[CCSR] Server ignored resume for {dest_path.name}; restarting download", flush=True)
                existing = 0

            content_length = int(resp.headers.get("Content-Length", 0))
            total = existing + content_length if content_length else 0
            downloaded = existing
            last_print = 0.0

            mode = "ab" if existing > 0 else "wb"
            with open(part, mode) as out_f:
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print > 0.5:
                        last_print = now
                        if total > 0:
                            pct = downloaded / total * 100.0
                            print(f"\r  -> {fmt_size(downloaded)} / {fmt_size(total)} ({pct:.1f}%)", end="", flush=True)
                        else:
                            print(f"\r  -> {fmt_size(downloaded)}", end="", flush=True)
            print("", flush=True)

        if not part.exists() or part.stat().st_size < 1_000_000:
            print(f"[CCSR] Downloaded file looks truncated: {dest_path.name}", flush=True)
            return False
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            print(f"[CCSR] Resume range rejected for {dest_path.name}; discarding partial file", flush=True)
            try:
                part.unlink()
            except OSError:
                pass
        else:
            print(f"\n[CCSR] Download error for {dest_path.name}: {exc}", flush=True)
        return False
    except Exception as exc:
        print(f"\n[CCSR] Download error for {dest_path.name}: {exc}", flush=True)
        # Keep the .part file so a retry can resume.
        return False


def download_engine_artifact(filename: str, target: Path) -> bool:
    """Download one artifact with retries across mirror repositories, then verify SHA256."""
    spec = ENGINE_SPECS[filename]
    part = part_path_for(target)

    for base_url in HF_REPOS:
        url = f"{base_url}/{filename}?download=true"
        for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
            if attempt > 1:
                delay = DOWNLOAD_RETRY_DELAY * (attempt - 1)
                print(f"[CCSR] Retry {attempt}/{DOWNLOAD_MAX_RETRIES} for {filename} in {delay}s ...", flush=True)
                time.sleep(delay)

            if not download_file_with_progress(url, target):
                continue

            got = part.stat().st_size if part.exists() else 0
            if got != spec["size"]:
                print(f"[CCSR] Incomplete transfer for {filename} ({got}/{spec['size']} bytes) - will resume", flush=True)
                continue

            print(f"[CCSR] Verifying SHA256 of {filename} ({fmt_size(spec['size'])}) ...", flush=True)
            if sha256_of_file(part) != spec["sha256"]:
                print(f"[CCSR] SHA256 mismatch for {filename} - discarding file and retrying from scratch", flush=True)
                try:
                    part.unlink()
                except OSError:
                    pass
                continue

            part.replace(target)
            print(f"[CCSR] {filename} downloaded and verified ({fmt_size(spec['size'])})", flush=True)
            return True

    return False


def sync_or_download_engines(skip_engine: bool = False, repair: bool = False) -> bool:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for filename in REQUIRED_FILES:
        target = ENGINE_DIR / filename

        if skip_engine:
            state = "present" if target.exists() else "missing"
            print(f"[CCSR] Engine artifact {filename}: {state} (engine sync skipped via --skip-engine)", flush=True)
            continue

        if target.exists():
            if validate_engine_file(target, filename, deep=repair):
                print(f"[CCSR] Engine artifact ready: {filename} ({fmt_size(target.stat().st_size)})", flush=True)
                continue
            print(f"[CCSR] Engine artifact failed validation, re-downloading: {filename}", flush=True)
            try:
                target.unlink()
            except OSError:
                pass

        log_step(f"Downloading missing engine artifact: {filename}")
        if not download_engine_artifact(filename, target):
            all_ok = False
            print(f"[CCSR] ERROR: Could not download {filename}.", flush=True)
            print(f"[CCSR] Manual download options (save to {ENGINE_DIR}):", flush=True)
            for base_url in HF_REPOS:
                print(f"[CCSR]   {base_url}/{filename}", flush=True)

    return all_ok


def verify_installation(allow_missing_engines: bool = False) -> None:
    verify_script = ROOT / "scripts" / "verify_install.py"
    if verify_script.exists():
        log_step("Running readiness verification")
        args = [sys.executable, str(verify_script)]
        if allow_missing_engines:
            args.append("--allow-missing-engines")
        try:
            subprocess.run(args, check=False)
        except Exception as exc:
            print(f"[CCSR] Verification check warning: {exc}", flush=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="install.py",
        description="ComfyUI-NunchakuFluxLoraStacker: CCSR TensorRT auto-installer (runtime stack + engine artifacts).",
    )
    parser.add_argument(
        "--skip-engine",
        action="store_true",
        help="skip engine artifact download and validation (runtime stack only)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="force-reinstall the TensorRT-RTX runtime stack and deep-verify (SHA256) the engine artifacts, re-downloading any that fail",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 80, flush=True)
    print("ComfyUI-NunchakuFluxLoraStacker — CCSR TensorRT Auto-Installer", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    if args.skip_engine:
        print("Mode: --skip-engine (engine artifacts will not be synced)", flush=True)
    if args.repair:
        print("Mode: --repair (force-reinstall + deep SHA256 verification)", flush=True)
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
    log_step("Ensuring TensorRT-RTX runtime stack" + (" (repair mode)" if args.repair else ""))
    for import_name, install_spec, min_ver in TRT_STACK:
        ensure_package(import_name, install_spec, no_deps=True, min_version=min_ver, force=args.repair)

    # 3. CCSR TensorRT Engine & Aux weights download
    sync_or_download_engines(skip_engine=args.skip_engine, repair=args.repair)

    # 4. Final verification
    verify_installation(allow_missing_engines=args.skip_engine)

    print("\n" + "=" * 80, flush=True)
    print("ComfyUI-NunchakuFluxLoraStacker CCSR TensorRT installation complete.", flush=True)
    if args.skip_engine:
        print("Note: engine artifacts were not synced (--skip-engine); rerun install.py without it to download them.", flush=True)
    print("=" * 80 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
