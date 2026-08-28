# 更新日志

<table align="center">
  <tr>
    <td align="center" bgcolor="#e5e7eb" width="88" height="36"><a href="../md/CHANGELOG.md"><font color="#4b5563"><b>EN</b></font></a></td>
    <td align="center" bgcolor="#d4465e" width="88" height="36"><font color="#ffffff"><b>中文</b></font></td>
  </tr>
</table>

## 发布历史

- v2.0.0 — Model Patch Loader ConvRot INT8：**Model Patch Loader**（`ModelPatchLoaderCustom`）现在可加载 comfy 原生 **ConvRot INT8** 量化的 Z-Image ControlNet 检查点（如 `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps_convrot_int8.safetensors`）。通过 `comfy_quant` 元数据自动检测 INT8，以 BF16 构建模块图并由 `mixed_precision_ops` 接入量化权重，因此权重始终以 **INT8 形式保存在内存中**，推理走 comfy-kitchen 的 `int8_linear` 内核（含在线 ConvRot 激活旋转）。GPU 加载与 CPU 卸载均支持；BF16 检查点保持原有路径。修复了 `Only Tensors of floating point and complex dtype can require gradients` 崩溃。技术移植自 [HSWQ ControlNet Loader](https://github.com/ussoewwin/ComfyUI-HSWQ-Loader-and-Tools)；完整说明见 [CONVROT_INT8_MODEL_PATCH_LOADER_EXPLANATION.md](CONVROT_INT8_MODEL_PATCH_LOADER_EXPLANATION.md)（[发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v2.0.0)）

- v1.39 – AMD / ROCm nunchaku 导入：合并 [PR #6](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/pull/6) 的 import 防护，使 `nunchaku` 导入失败时整包仍可加载；后续将 FLUX 注册限制为 `_NUNCHAKU_AVAILABLE`、不对 `standard` / `standard_v3` 包宽 try/except，并以明确错误替代 `compose_lora = None` ([发行说明](v1.39.md))

- v1.38 – CCSR 依赖：在 `requirements.txt` 中补充 CCSR 所需包（`pytorch-lightning` 及相关依赖），并新增 `install.py`，由 ComfyUI-Manager 自动安装—避免缺少 `pytorch_lightning` 时整个自定义节点包加载失败 ([发行说明](v1.38.md))

- v1.37 – Nunchaku Resolution Selector：新增 **Nunchaku Resolution Selector**（`NunchakuResolutionSelector`，菜单 `ussoewwin/resolution`）— 从 Flux1 风格宽高比预设（或自定义尺寸）选择宽高，输出 hires 尺寸、空 **16 通道** latent 与 info 字符串。预设沿用 ControlAltAI Megapixel Calculator 的比例（1.0 MP / 1.5 MP High）。README 与 zhmd README 已补充说明与截图。

- v1.36 – LoRA Stacker V3：新增标准 SD 模型用 **LoRA Stacker V3**（`LoraStackerV3_10`）— 与 V2 相同的 10 槽位堆叠，另增 **全局 `toggle_all` 总开关**（关闭时不应用任何 LoRA，输出原样透传）及 **每槽位 `enabled_1` … `enabled_10` 独立开关**，便于快速 A/B 对比与部分堆叠而无需重连。实现于 `nodes/lora/standard_v3.py`；README 与 zhmd README 已补充用法与开关真值表说明。

- v1.35 – CCSR: 在 `nodes/CCSR/` 下集成了 CCSR 放大节点（CCSR_Upscale、CCSR_Model_Select、DownloadAndLoadCCSRModel；上游源自 [kijai/ComfyUI-CCSR](https://github.com/kijai/ComfyUI-CCSR)，基于原始 [csslc/CCSR](https://github.com/csslc/CCSR) 的 Apache-2.0 实现）。添加了对 Python 3.13 和最新 ComfyUI 环境的兼容支持，实现了动态包路径解析以支持自定义安装目录名称，并在 README 中添加了节点说明和截图。

- v1.34 – Color Filter: 修复了由于过期的预编译字节码缓存（`__pycache__`）导致自定义排除词失效的问题；并实现了递归逗号折叠算法，以正确处理移除多个连续单词时的格式化问题。 ([发行说明](v1.34.md))

- v1.33 – Color Filter: 改进了用户自定义 `exclude_words` 的解析，根据单词是否包含 ASCII 字母数字字符来动态处理单词边界 (`\b`)，从而防止正则表达式错误并避免日语文本和特殊字符的匹配遗漏。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.33))

- v1.32 – ControlAltAI: 在 `nodes/controlaltai/` 下集成了 ControlAltAI 工具节点（上游 https://github.com/gseth/ControlAltAI-Nodes，MIT）。此工具包中包含了该节点的兼容 Python 3.13 的个人分支（相同节点集），以减少维护多个独立仓库的精力。在根目录 `__init__.py` 中进行了注册；在 README 中添加了 ControlAltAI 部分（位于 Florence-2 之后）；节点详细信息仅记录在 [zhmd/controlalttai.md](controlalttai.md) 中；为 Integer Settings Advanced 提供了前端辅助脚本 `js/integer_settings_advanced.js`。

- v1.31 – Florence-2: 在 `nodes/florence2/` 下集成了 Florence-2 VLM 节点（支持 Sage Attention 2/3 和 Transformers 5.x 加载路径；合并了上游源自 [kijai/ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2) 的分支）。更新了 README 中的节点概览（11个节点）、Florence-2 部分、截图 `../png/Florence2.png`、致谢以及 MIT 子目录 `nodes/florence2/LICENSE` 的许可证说明。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.31))

- v1.30 – Color Filter: 添加了 `ColorFilter` 节点 (`nodes/color_filter/`)，用于从描述和标签字符串（例如 Florence-2、WD14 Tagger）中移除单色/黑白相关词语；更新了 README 部分和截图。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.30))

- v1.29 – LoRA Loader V2 更新: 在 FLUX LoRA Loader V2、LoRA Stacker V2 和 SDNQ LoRA Stacker V2 中启用了负数 LoRA 强度值（最小值/最大值：-100.0 到 100.0），以匹配标准的 ComfyUI 行为。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.29))

- v1.28 – Nunchaku Flux1 PulID: 发布了针对影响上游 Nunchaku Flux1 PulID 节点的错误的缓解措施；详细信息已在发行说明中发布。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.28))

- v1.27 – Model Patch Loader: 修复了由 ComfyUI 更新引起的错误（CPU 卸载 / CoreModelPatcher）。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.27))

- v1.26 – Model Patch Loader: 修复了 Z-Image ControlNet 矩阵乘法形状错误；从权重文件（checkpoint）推断 control_in_dim，并在加载 state_dict 中包含仅权重文件的键，以便在延迟初始化（lazy init）下加载嵌入器（embedder）权重。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.26))

- [v1.25](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.25) – Universal LoRA Analyzer、图像加载（Load Image）节点、SAM3 集成；更新了 README 中的节点文档和结构。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.25))

- [v1.24](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.24) – 修复了 ComfyUI-nunchaku 1.1.0 下 LoRA 不工作的问题: 解决了更新到 ComfyUI-nunchaku 1.1.0 后，LoRA 未应用于最终图像输出的问题。该修复确保了正确的 MODEL 对象克隆和状态保留。

- v1.21 – Z-Image ControlNet Union 2.1 支持: 为 Z-Image ControlNet 添加了动态层数检测以支持 Union 2.1 模型。

- v1.18 – SDNQ LoRA Stacker V2: 为 SDNQ 量化模型添加了专用的 SDNQ LoRA Stacker V2 节点，带有动态 10 槽位 UI（与 [comfyui-sdnq-splited](https://github.com/ussoewwin/comfyui-sdnq-splited) 配合使用）。修复了 Z-Image ControlNet 加载以支持 Union 2.0 权重文件，并对尺寸不匹配进行了过滤。

- v1.17 – Model Patch Loader: 添加了 ModelPatchLoaderCustom 节点，支持 CPU 卸载以加载 ControlNet 和特征投影器补丁。

- v1.16 – LoRA Stacker V2: 为标准 SD 模型（SDXL、Flux、WAN2.2）添加了通用 LoRA 加载器，带有动态 10 槽位 UI。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.16))

- v1.15 – FastGroupsBypasserV2 修复: 修复了关键的控件更新 bug，即第二次属性更改需要按 F5 刷新。 ([发行说明](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.15))

- v1.14 – 节点简化: 移除了测试节点 (x1-x9)，仅保留 FLUX LoRA Loader V2 (x10) 作为正式节点。

- v1.13 – 清理发布: 从仓库中移除了所有备份文件，将 FluxLoraMultiLoader_10 的显示名称更新为 "FLUX LoRA Loader V2"。

- v1.12 – V2 节点发布: 带有动态下拉框 UI 的 FLUX LoRA Loader V2 以及 Fast Groups Bypasser V2。

- v1.11 – 修正了 README 中的克隆命令，以使用规范的仓库 URL。 ([问题 #3](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/issues/3))

- [v1.10](https://github.com/ussoewwin/ComfyUI-NunchakuFluxLoraStacker/releases/tag/v1.10) – LoRA 加载器修复 - 完整版本
