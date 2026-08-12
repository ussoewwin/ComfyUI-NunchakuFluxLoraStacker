<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/ROCM_NUNCHAKU_IMPORT_REGISTRATION_FIX_EXPLANATION.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

# AMD / ROCm nunchaku 导入防护与 FLUX 注册修复

## 目的

本文说明：在官方 `nunchaku` 包无法导入时（典型场景为 AMD / ROCm），本包如何仍可加载；以及如何门控 FLUX 节点，避免在没有可用 nunchaku 时注册损坏的 FLUX 条目。

相关历史：

- 上游贡献已合并为 [PR #6](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/pull/6)（`6608421`）：对 nunchaku 的导入防护。
- 本仓库后续：提交 `2042e0c`（注册门控、`standard` / `standard_v3` 导入范围、禁止 `compose_lora = None`）。

---

## 问题

在没有 nunchaku（或在非 CUDA 栈上 nunchaku 导入失败）时，若对依赖 nunchaku 的模块做未防护导入，会在 ComfyUI 启动时拖垮**整个**自定义节点包。

PR #6 正确地为 nunchaku 导入加入了 try/except，使整包仍可加载。但仅此不足以形成干净的 FLUX 注册路径：

1. 若 `wrappers/flux.py` 吞掉导入失败，但 FLUX 节点模块仍能成功导入，`__init__.py` 可能把「模块导入成功」当成「应注册 FLUX」。
2. 损坏的 FLUX 节点会出现在界面中，却没有可用的 nunchaku 后端。
3. 将 `compose_lora = None` 会把缺失依赖变成晚到的 `TypeError`（`NoneType` is not callable），而不是清晰失败。
4. 把非 nunchaku 加载器（`standard` / `standard_v3`）包进同一宽 try/except，会掩盖无关错误，并把失败误标为 `[ROCm]`。

---

## 设计

| 关注点 | 规则 |
|--------|------|
| 整包加载 | nunchaku 导入失败不得中止非 FLUX 节点注册。 |
| FLUX 注册 | 仅当 `_NUNCHAKU_AVAILABLE` 为真时注册 FLUX 节点。 |
| `compose_lora` | 绝不留下可调用的 `None`。缺少 nunchaku 必须抛出明确的 `RuntimeError`。 |
| `standard` / `standard_v3` | 正常导入；不要包在 nunchaku/ROCm try/except 中。 |
| 日志 | `[ROCm]` 类消息仅用于 nunchaku / FLUX 缺失。 |

---

## 文件与行为

### 1) `wrappers/flux.py`

- 保留对 nunchaku 导入的 try/except。
- 根据结果设置 `_NUNCHAKU_AVAILABLE`。
- 失败时提供**会抛错**的 `compose_lora` 桩（不是 `None`）。

### 2) `__init__.py`

- 从 `wrappers.flux` 导入 `_NUNCHAKU_AVAILABLE`。
- **仅当** `_NUNCHAKU_AVAILABLE` 为真时导入并注册 FLUX（`flux` / `flux_v2`）。
- 导入 `standard` / `standard_v3` 时不使用宽的 nunchaku try/except。

### 3) `nodes/lora/flux.py` 与 `nodes/lora/flux_v2.py`

- 在 `compose_lora` 的 ImportError 时抛出带明确信息的 `RuntimeError`。
- 不赋值 `compose_lora = None`。

### 4) `nodes/lora/sdnq.py`

- 保留 PR #6 的 `traceback.print_exc()`（仅诊断；与 FLUX 注册无关）。

---

## 预期结果

1. **无 nunchaku / ROCm 导入失败：** ComfyUI 可加载本包。非 FLUX 节点会注册。FLUX 节点**不会**注册。
2. **NVIDIA + 可用 nunchaku：** FLUX 节点照常注册，行为与后续修复前一致。
3. **在无 nunchaku 时强行调用 compose：** 明确的 `RuntimeError`，而不是调用 `None` 的 `TypeError`。
4. **`standard` / `standard_v3` 失败：** 不按 ROCm/nunchaku 失败处理。

---

## 更新后应做事项

1. 完全重启 ComfyUI，使 `__init__.py` 与 wrappers 重新加载。
2. 确认节点列表：仅在 nunchaku 导入成功时出现 FLUX 加载器。
3. 在无 nunchaku 的 AMD / ROCm 上，确认本包其余节点仍出现。

---

## 范围之外

- 修改 ComfyUI 核心或其他自定义节点仓库。
- 要求 PR 作者实现注册门控后续（该工作已在本仓库完成）。
