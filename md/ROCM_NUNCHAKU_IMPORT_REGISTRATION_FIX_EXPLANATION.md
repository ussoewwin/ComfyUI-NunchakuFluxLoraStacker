<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../zhmd/ROCM_NUNCHAKU_IMPORT_REGISTRATION_FIX_EXPLANATION.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# AMD / ROCm Nunchaku Import Guard and FLUX Registration Fix

## Purpose

This document explains how this pack stays loadable when the official `nunchaku` package cannot import (typical on AMD / ROCm), and how FLUX nodes are gated so broken FLUX entries are not registered without a working nunchaku install.

Related history:

- Upstream contribution merged as [PR #6](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/pull/6) (`6608421`): import guards around nunchaku.
- Follow-up on this repository: commit `2042e0c` (registration gate, `standard` / `standard_v3` import scope, no `compose_lora = None`).

---

## Problem

Without nunchaku (or when nunchaku fails to import on non-CUDA stacks), an unguarded import of nunchaku-dependent modules could take down the **entire** custom-node pack at ComfyUI startup.

PR #6 correctly introduced try/except around nunchaku imports so the pack can still load. That alone was not enough for a clean FLUX registration path:

1. If `wrappers/flux.py` swallows the import failure but FLUX node modules still import successfully, `__init__.py` may treat “module import OK” as “register FLUX”.
2. Broken FLUX nodes then appear in the UI without a working nunchaku backend.
3. Assigning `compose_lora = None` turns a missing dependency into a late `TypeError` (`NoneType` is not callable) instead of a clear failure.
4. Wrapping non-nunchaku loaders (`standard` / `standard_v3`) in the same broad try/except can hide unrelated bugs and mislabel failures as `[ROCm]`.

---

## Design

| Concern | Rule |
|---------|------|
| Pack load | Failure to import nunchaku must not abort registration of non-FLUX nodes. |
| FLUX registration | Register FLUX nodes only when `_NUNCHAKU_AVAILABLE` is true. |
| `compose_lora` | Never leave a `None` callable. Missing nunchaku must raise a clear `RuntimeError`. |
| `standard` / `standard_v3` | Import normally; do not wrap in nunchaku/ROCm try/except. |
| Logging | `[ROCm]`-style messages apply to nunchaku / FLUX absence only. |

---

## Files and behavior

### 1) `wrappers/flux.py`

- Keeps try/except around nunchaku imports.
- Sets `_NUNCHAKU_AVAILABLE` from that outcome.
- On failure, provides a **raising** `compose_lora` stub (not `None`).

### 2) `__init__.py`

- Imports `_NUNCHAKU_AVAILABLE` from `wrappers.flux`.
- Imports and registers FLUX (`flux` / `flux_v2`) **only when** `_NUNCHAKU_AVAILABLE` is true.
- Imports `standard` / `standard_v3` without a broad nunchaku try/except.

### 3) `nodes/lora/flux.py` and `nodes/lora/flux_v2.py`

- On `compose_lora` ImportError, raise `RuntimeError` with an explicit message.
- Do not assign `compose_lora = None`.

### 4) `nodes/lora/sdnq.py`

- Keeps `traceback.print_exc()` from PR #6 (diagnostics only; unrelated to FLUX registration).

---

## Expected results

1. **No nunchaku / ROCm import failure:** ComfyUI loads the pack. Non-FLUX nodes register. FLUX nodes do **not** register.
2. **NVIDIA + working nunchaku:** FLUX nodes register and behave as before the follow-up fix.
3. **Forced use of compose without nunchaku:** clear `RuntimeError`, not `TypeError` from calling `None`.
4. **`standard` / `standard_v3` failures:** not treated as ROCm/nunchaku failures.

---

## What to do after update

1. Fully restart ComfyUI so `__init__.py` and wrappers reload.
2. Confirm node list: FLUX loaders present only when nunchaku imports successfully.
3. On AMD / ROCm without nunchaku, confirm the rest of this pack still appears.

---

## Out of scope

- Changes to ComfyUI core or other custom-node repositories.
- Asking PR authors to implement the registration-gate follow-up (that work is done in this repository).
