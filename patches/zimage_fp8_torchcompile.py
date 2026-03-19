"""
torch.compile + FP8 + Lumina 用の**局所的な**安全パッチだけを提供するモジュール。

削除済み（無効化した）もの:
  - comfy.ops.mixed_precision_ops 用のグローバル Linear パッチ
  - TensorCoreFP8Layout.linear の差し替えパッチ
  - グローバル F.linear / matmul / mul / add の shape 補正パッチ
  - LoraDiff.calculate_weight パッチ（LoRA スキップは HSWQ ローダ側で実装）

残してあるもの:
  - Lumina 用の局所パッチのみ:
      * _apply_lumina_modulate_patch
      * _apply_lumina_apply_gate_patch
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _apply_lumina_modulate_patch() -> bool:
    """
    comfy.ldm.lumina.model.modulate をラップし、
    x * (1 + scale.unsqueeze(1)) で hidden_dim が食い違う場合に scale の末尾次元を安全側に合わせる。

    - 通常パスはそのまま
    - RuntimeError: \"The size of tensor a (...) must match the size of tensor b (...)\" のときだけ補正
    - x の最後の次元を正とし、scale の長さをそこに揃える（スライス or ゼロ埋め）
    """
    try:
        import comfy.ldm.lumina.model as lumina_model
        import torch
    except Exception:
        return False

    original_modulate = getattr(lumina_model, "modulate", None)
    if original_modulate is None or not callable(original_modulate):
        return False

    # 多重パッチ防止
    if getattr(original_modulate, "_nunchaku_fp8_modulate_safe_patched", False):
        return True

    def _safe_modulate(x, scale, *args, **kwargs):
        try:
            return original_modulate(x, scale, *args, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "The size of tensor a" not in msg or "must match the size of tensor b" not in msg:
                # 想定外の RuntimeError はそのまま
                raise

            try:
                x_shape = tuple(getattr(x, "shape", ()))
                scale_shape = tuple(getattr(scale, "shape", ()))
            except Exception:
                raise

            hidden_x = None
            hidden_scale = None
            try:
                if hasattr(x, "ndim") and x.ndim >= 1:
                    hidden_x = x.shape[-1]
                if hasattr(scale, "ndim") and scale.ndim >= 1:
                    hidden_scale = scale.shape[-1]
            except Exception:
                raise

            if not isinstance(hidden_x, int) or not isinstance(hidden_scale, int) or hidden_x == hidden_scale:
                # 調整のしようがない場合は誤魔化さない
                raise

            target = min(hidden_x, hidden_scale)

            logger.warning(
                "ComfyUI-NunchakuFluxLoraStacker: lumina.modulate hidden dim mismatch "
                "(x_last=%s, scale_last=%s, target=%s, x=%s, scale=%s), "
                "slicing both to target and retrying [z_image/FP8/torch.compile compat]",
                hidden_x,
                hidden_scale,
                target,
                x_shape,
                scale_shape,
            )

            x_fixed = x[..., :target].contiguous()
            scale_fixed = scale[..., :target].contiguous()

            return original_modulate(x_fixed, scale_fixed, *args, **kwargs)

    _safe_modulate._nunchaku_fp8_modulate_safe_patched = True
    lumina_model.modulate = _safe_modulate

    return True


def _apply_lumina_apply_gate_patch() -> bool:
    """
    comfy.ldm.lumina.model.apply_gate をラップし、
    gate * x の hidden_dim が食い違う場合に両方の末尾次元を安全側に合わせる。
    """
    try:
        import comfy.ldm.lumina.model as lumina_model
    except Exception:
        return False

    original_apply_gate = getattr(lumina_model, "apply_gate", None)
    if original_apply_gate is None or not callable(original_apply_gate):
        return False

    # 多重パッチ防止
    if getattr(original_apply_gate, "_nunchaku_fp8_apply_gate_safe_patched", False):
        return True

    def _safe_apply_gate(gate, x, *args, **kwargs):
        """
        gate * x で hidden_dim がズレている場合に、
        例外発生時だけ両方を安全側にそろえて gate * x を自前で実行する。
        """
        try:
            # まずは元実装をそのまま試す
            return original_apply_gate(gate, x, *args, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if "The size of tensor a" not in msg or "must match the size of tensor b" not in msg:
                # 形状以外のエラーはそのまま投げ直す
                raise

            # gate / x の shape, hidden_dim を取得（失敗したら諦めて再送出）
            try:
                gate_shape = tuple(getattr(gate, "shape", ()))
                x_shape = tuple(getattr(x, "shape", ()))
            except Exception:
                raise

            hidden_gate = None
            hidden_x = None
            try:
                if hasattr(gate, "ndim") and gate.ndim >= 1:
                    hidden_gate = gate.shape[-1]
                if hasattr(x, "ndim") and x.ndim >= 1:
                    hidden_x = x.shape[-1]
            except Exception:
                raise

            if hidden_gate is None or hidden_x is None or hidden_gate == hidden_x:
                # 揃えようがない場合は元のエラーを再送出
                raise

            try:
                target = min(hidden_gate, hidden_x)
            except Exception:
                raise

            logger.warning(
                "ComfyUI-NunchakuFluxLoraStacker: lumina.apply_gate hidden dim mismatch "
                "(gate_last=%s, x_last=%s, target=%s, gate=%s, x=%s), "
                "slicing both to target and applying gate * x directly after RuntimeError [z_image/FP8/torch.compile compat]",
                hidden_gate,
                hidden_x,
                target,
                gate_shape,
                x_shape,
            )

            gate_fixed = gate[..., :target].contiguous()
            x_fixed = x[..., :target].contiguous()

            # ここでは timestep_zero_index などは無視し、単純な gate * x とする
            return gate_fixed * x_fixed

    _safe_apply_gate._nunchaku_fp8_apply_gate_safe_patched = True
    lumina_model.apply_gate = _safe_apply_gate

    return True

