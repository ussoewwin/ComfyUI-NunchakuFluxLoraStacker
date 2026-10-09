<table align="center">
  <tr>
    <td align="center" bgcolor="#3478ca" width="88" height="36"><font color="#ffffff"><b>EN</b></font></td>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/blob/main/zhmd/v2.1.2.md"><font color="#4b5563"><b>中文</b></font></a></td>
  </tr>
</table>

# v2.1.2 — Node Class Types Namespaced with `hswq_` Prefix (All 16 Conflicts Eliminated)

This release renames **every node class type in this pack that shared a `class_type` name with another published pack**, prefixing all of them with a single unified namespace: **`hswq_`**. This eliminates all 16 ComfyUI-Manager conflict entries attributed to this pack (4 Florence-2 + 12 ControlAltAI-family). Node behaviour, widgets, categories, inputs, outputs and data types are **unchanged**.

---

## 1. Summary of Changes

| Area | v2.0.5 and earlier | v2.1.2 |
|---|---|---|
| **Florence-2 class types (4)** | `DownloadAndLoadFlorence2Model` / `DownloadAndLoadFlorence2Lora` / `Florence2ModelLoader` / `Florence2Run` — names shared with `kijai/ComfyUI-Florence2`, `un-seen/comfyui-tensorops`, `Ltamann/ComfyUI-TBG-ETUR` | **`hswq_`-prefixed** — no longer claims any of those shared names |
| **ControlAltAI utility class types (11 + 1)** | `FluxSampler`, `FluxUnionControlNetApply`, `BooleanBasic`, `BooleanReverse`, `GetImageSizeRatio`, `IntegerSettings`, `IntegerSettingsAdvanced`, `PerturbationTexture`, `TextBridge`, `TwoWaySwitch`, `ThreeWaySwitch` (+ `FluxControlNetApply` in the module map) — names shared with `gseth/ControlAltAI-Nodes` | **`hswq_`-prefixed** — conflict-ledger entries cleared |
| **`MegapixelCalculatorNode`** | Pack-specific name | **Unchanged** — no collision exists, all existing workflows keep working |
| **Node behaviour / I/O types** | — | Identical to v2.0.5 (`FL2MODEL`, `PEFTLORA`, `IMAGE`, `MASK`, `STRING`, `JSON` unchanged; Florence-2 fork keeps Transformers 5.x compatibility and Sage Attention 3 support) |
| **Frontend helper** | `js/integer_settings_advanced.js` matched `IntegerSettingsAdvanced` | Now matches `hswq_IntegerSettingsAdvanced`; mutual-exclusion logic untouched |
| **Version metadata** | `2.0.5` | `2.1.2` (`pyproject.toml` / `__init__.py`), published to the Comfy Registry |

---

## 2. Why the Renames — Conflict Accounting in ComfyUI-Manager

ComfyUI-Manager counts a conflict when one node name is registered by more than one pack in the official registry ledger (`extension-node-map`). Until now this pack's ledger entry reused 16 names owned (in the public DB sense) by other published packs, so the Manager UI showed **16 conflicts** for this pack. After the registry picks up this version, this pack registers only names no other published pack claims, and the conflict count drops to 0. Same-name collisions were also a live risk: ComfyUI keeps the last registration, so installing another Florence-2 pack would silently shadow this pack's fork (and vice versa).

## 3. Renamed Class Types — Florence-2 (4)

| Old class_type | New class_type |
|---|---|
| `DownloadAndLoadFlorence2Model` | `hswq_DownloadAndLoadFlorence2Model` |
| `DownloadAndLoadFlorence2Lora` | `hswq_DownloadAndLoadFlorence2Lora` |
| `Florence2ModelLoader` | `hswq_Florence2ModelLoader` |
| `Florence2Run` | `hswq_Florence2Run` |

## 4. Renamed Class Types — ControlAltAI Utilities (11 registered + 1 in module map)

| Old class_type | New class_type |
|---|---|
| `FluxSampler` | `hswq_FluxSampler` |
| `FluxUnionControlNetApply` | `hswq_FluxUnionControlNetApply` |
| `BooleanBasic` | `hswq_BooleanBasic` |
| `BooleanReverse` | `hswq_BooleanReverse` |
| `GetImageSizeRatio` | `hswq_GetImageSizeRatio` |
| `IntegerSettings` | `hswq_IntegerSettings` |
| `IntegerSettingsAdvanced` | `hswq_IntegerSettingsAdvanced` |
| `PerturbationTexture` | `hswq_PerturbationTexture` |
| `TextBridge` | `hswq_TextBridge` |
| `TwoWaySwitch` | `hswq_TwoWaySwitch` |
| `ThreeWaySwitch` | `hswq_ThreeWaySwitch` |

`FluxControlNetApply` (defined in `nodes/controlaltai/flux_controlnet_node.py`, not present in the shipped registration) follows the same rule and is renamed to `hswq_FluxControlNetApply`.

## 5. Unchanged Node

`MegapixelCalculatorNode` keeps its original class type: it is not registered by any other published pack in the conflict ledger, and 34 local workflows reference it. Renaming it would have broken those workflows for no conflict-resolution benefit.

## 6. Workflow Migration

`class_type` strings are stored inside workflow JSON files. If a workflow still uses an old name listed above:

1. Re-add the node from the search menu (it now appears as `hswq_<OldName>`; display names are unchanged where they were pack-specific), or
2. Find/replace in the JSON: `"type": "<OldName>"` to `"type": "hswq_<OldName>"` (API-format exports also contain `"class_type"`).

A workflow left with an old name will resolve to the *other* pack's node implementation if that pack is installed — behaviour may differ (e.g. the other Florence-2 packs lack the Transformers 5.x / Sage Attention 3 fixes in this fork). All workflows under this install have already been migrated.

## 7. File Modification Breakdown

| File | Type | Description |
|---|---|---|
| `nodes/florence2/nodes.py` | Update | 4 class types + display mappings namespaced `hswq_` |
| `nodes/controlaltai/*.py` (13 files) | Update | 11 registered class types (+ `FluxControlNetApply` module map) namespaced `hswq_`; registration dicts updated |
| `js/integer_settings_advanced.js` | Update | Frontend `nodeData.name` check matches `hswq_IntegerSettingsAdvanced` |
| `nodes/controlaltai/controlalttai.md` | Update | Changed-name table updated to `hswq_` prefix + dated note |
| `README.md` | Update | Florence-2 / ControlAltAI node names and `hswq_` note |
| `md/CHANGELOG.md` / `zhmd/CHANGELOG.md` | Update | v2.1.2 release-history entry with release-notes links (EN + 中文) |
| `pyproject.toml` / `__init__.py` | Update | Package version bumped to 2.1.2 |
| `release/_release_v2.1.2_body_en.md` | New | This release note (English release body) |
| `zhmd/v2.1.2.md` | New | Chinese release note (this document's counterpart) |

## 8. Verification & Unaffected Nodes

- All modified `.py` files byte-compile in both the repository and the installed copy (`custom_nodes`); repo==live SHA-256 verified per file.
- Disk-wide scan of repo, installed pack, and all local workflow JSONs: zero `NFL2_` / `CAI_` intermediate-prefix residue; 16 names registered under `hswq_` only.
- Unaffected: FLUX LoRA Loader V2, LoRA Stacker V2 / V3, SDNQ LoRA Stacker V2, Universal LoRA Analyzer, Color Filter, CCSR (TensorRT), Nunchaku Resolution Selector, Fast Groups Bypasser V2, Load Image (ussoewwin).
- 15 local workflow files migrated mechanically with per-file JSON validity and exact-diff assertions.
- The 2.1.2 version string passed this repository's Comfy Registry publish workflow.
