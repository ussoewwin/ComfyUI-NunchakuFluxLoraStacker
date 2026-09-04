import os
import torch
import sys
import logging
from torch.nn import functional as F
from contextlib import nullcontext
from omegaconf import OmegaConf

from .model.q_sampler import SpacedSampler
from .model.ccsr_stage1 import ControlLDM

from .utils.common import instantiate_from_config, load_state_dict

import comfy.model_management as mm
import comfy.utils
import folder_paths
from nodes import ImageScaleBy
from nodes import ImageScale

try:
    from .trt_engine import get_engine, release_trt_engines
    _TRT_AVAILABLE = True
except Exception:
    _TRT_AVAILABLE = False

script_directory = os.path.dirname(os.path.abspath(__file__))

# xformersログ制御用のクラス
class XFormersKernelOnce:
    """ロガー＋stdout/errを収集モードにし、終了時に1行だけ出す"""
    def __init__(self):
        self._filter = self._XFormersKernelFilter()
        self._loggers = []
        self._saved_out = None
        self._saved_err = None
        self._proxy_out = None
        self._proxy_err = None
        self._agg = self._KernelAggregator()
    
    class _KernelAggregator:
        """カーネル選択を記録・集約"""
        def __init__(self):
            self._kernels = []
            self._fa2_seen = False
        
        def record(self, kernel):
            if "fa2" in kernel.lower():
                self._fa2_seen = True
            self._kernels.append(kernel)
        
        def selected(self):
            if self._fa2_seen:
                # FA2が含まれていればFA2を優先
                for k in self._kernels:
                    if "fa2" in k.lower():
                        return k
            # 最後に観測したカーネルを返す
            return self._kernels[-1] if self._kernels else None
    
    class _XFormersKernelFilter(logging.Filter):
        """xformersのカーネル選択ログを捕捉して抑止"""
        def filter(self, record):
            try:
                msg = record.getMessage()
            except Exception:
                return True
            if "memory_efficient_attention: selected kernel" in msg:
                return False  # ログを抑止
            return True
    
    class _StdoutProxy:
        """stdout/stderrをラップして対象行のみ収集して抑止"""
        def __init__(self, underlying, agg):
            self._u = underlying
            self._agg = agg
        
        def write(self, s):
            try:
                text = str(s)
            except Exception:
                text = s
            if "memory_efficient_attention: selected kernel" in text:
                # カーネル名を抽出
                if "=" in text:
                    kernel = text.split("=")[-1].strip()
                    self._agg.record(kernel)
                return len(s)  # 書き込み長を返す（進行バー等の整合性を維持）
            return self._u.write(s)
        
        def flush(self):
            return self._u.flush()
        
        # 以降は透過委譲
        def fileno(self): return self._u.fileno() if hasattr(self._u, "fileno") else 1
        def isatty(self): return self._u.isatty() if hasattr(self._u, "isatty") else False
        def readable(self): return self._u.readable() if hasattr(self._u, "readable") else False
        def writable(self): return self._u.writable() if hasattr(self._u, "writable") else True
        def seekable(self): return self._u.seekable() if hasattr(self._u, "seekable") else False
        @property
        def encoding(self): return getattr(self._u, "encoding", "utf-8")
        @property
        def errors(self): return getattr(self._u, "errors", None)
        @property
        def buffer(self): return getattr(self._u, "buffer", None)
        def __getattr__(self, name): return getattr(self._u, name)
    
    def __enter__(self):
        # よく使われるロガーに一括装着
        self._loggers = [
            logging.getLogger(),
            logging.getLogger("xformers"),
            logging.getLogger("xformers.ops"),
            logging.getLogger("xformers.ops.fmha"),
            logging.getLogger("xformers_attention_log"),
        ]
        for lg in self._loggers:
            try: 
                lg.addFilter(self._filter)
            except Exception: 
                pass
        
        # stdout/errをプロキシに差し替え（より強力に）
        self._saved_out, self._saved_err = sys.stdout, sys.stderr
        self._proxy_out = self._StdoutProxy(self._saved_out, self._agg)
        self._proxy_err = self._StdoutProxy(self._saved_err, self._agg)
        
        # グローバルに設定
        sys.stdout = self._proxy_out
        sys.stderr = self._proxy_err
        
        # さらに、xformersの内部ロガーも制御
        try:
            import xformers.ops.fmha
            if hasattr(xformers.ops.fmha, '_memory_efficient_attention_forward'):
                # 元の関数を保存
                if not hasattr(xformers.ops.fmha, '_original_memory_efficient_attention_forward'):
                    xformers.ops.fmha._original_memory_efficient_attention_forward = xformers.ops.fmha._memory_efficient_attention_forward
                
                # ログ出力を無効化した関数で置き換え
                def _silent_memory_efficient_attention_forward(inp, op=None):
                    inp.validate_inputs()
                    output_shape = inp.normalize_bmhk()
                    if op is None:
                        op = xformers.ops.fmha._dispatch_fw(inp, False)
                        # ログ出力を無効化
                        if not hasattr(_silent_memory_efficient_attention_forward, "_last_kernel"):
                            _silent_memory_efficient_attention_forward._last_kernel = None
                        last_kernel = _silent_memory_efficient_attention_forward._last_kernel
                        current = getattr(op, "NAME", str(op))
                        if current != last_kernel:
                            # ログ出力を無効化し、集約器に記録のみ
                            if "fa2" in str(current).lower():
                                self._agg._fa2_seen = True
                            self._agg._kernels.append(str(current))
                            _silent_memory_efficient_attention_forward._last_kernel = current
                    else:
                        xformers.ops.fmha._ensure_op_supports_or_raise(ValueError, "memory_efficient_attention", op, inp)
                    out, *_ = op.apply(inp, needs_gradient=False)
                    return out.reshape(output_shape)
                
                xformers.ops.fmha._memory_efficient_attention_forward = _silent_memory_efficient_attention_forward
                
                # requires_grad版もパッチ
                if hasattr(xformers.ops.fmha, '_memory_efficient_attention_forward_requires_grad'):
                    if not hasattr(xformers.ops.fmha, '_original_memory_efficient_attention_forward_requires_grad'):
                        xformers.ops.fmha._original_memory_efficient_attention_forward_requires_grad = xformers.ops.fmha._memory_efficient_attention_forward_requires_grad
                    
                    def _silent_memory_efficient_attention_forward_requires_grad(inp, op=None):
                        inp.validate_inputs()
                        output_shape = inp.normalize_bmhk()
                        if op is None:
                            op = xformers.ops.fmha._dispatch_fw(inp, True)
                            # ログ出力を無効化
                            if not hasattr(_silent_memory_efficient_attention_forward_requires_grad, "_last_kernel"):
                                _silent_memory_efficient_attention_forward_requires_grad._last_kernel = None
                            last_kernel = _silent_memory_efficient_attention_forward_requires_grad._last_kernel
                            current = getattr(op, "NAME", str(op))
                            if current != last_kernel:
                                # ログ出力を無効化し、集約器に記録のみ
                                if "fa2" in str(current).lower():
                                    self._agg._fa2_seen = True
                                self._agg._kernels.append(str(current))
                                _silent_memory_efficient_attention_forward_requires_grad._last_kernel = current
                        else:
                            xformers.ops.fmha._ensure_op_supports_or_raise(ValueError, "memory_efficient_attention", op, inp)
                        out = op.apply(inp, needs_gradient=True)
                        assert out[1] is not None
                        return (out[0].reshape(output_shape), out[1])
                    
                    xformers.ops.fmha._memory_efficient_attention_forward_requires_grad = _silent_memory_efficient_attention_forward_requires_grad
        except Exception as e:
            print(f"Warning: Could not patch xformers: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # まず復元（副作用を残さない）
        if self._saved_out is not None: 
            sys.stdout = self._saved_out
        if self._saved_err is not None: 
            sys.stderr = self._saved_err
        
        for lg in self._loggers:
            try: 
                lg.removeFilter(self._filter)
            except Exception: 
                pass
        
        # xformersのパッチを元に戻す
        try:
            import xformers.ops.fmha
            if hasattr(xformers.ops.fmha, '_original_memory_efficient_attention_forward'):
                xformers.ops.fmha._memory_efficient_attention_forward = xformers.ops.fmha._original_memory_efficient_attention_forward
            if hasattr(xformers.ops.fmha, '_original_memory_efficient_attention_forward_requires_grad'):
                xformers.ops.fmha._memory_efficient_attention_forward_requires_grad = xformers.ops.fmha._original_memory_efficient_attention_forward_requires_grad
        except Exception:
            pass
        
        # 観測結果から1行だけ表示（FA2優先→なければ最後に観測）
        selected = self._agg.selected()
        if selected:
            print(f"[xformers] memory_efficient_attention: selected kernel = {selected} (CCSR)")

_TRT_ENGINE_DIR = os.path.join(script_directory, "trt_engines")


def _list_engines():
    """Return engine filenames present under nodes/CCSR/trt_engines."""
    d = _TRT_ENGINE_DIR
    if not os.path.isdir(d):
        return []
    return sorted(fn for fn in os.listdir(d) if fn.endswith(".rtxplan"))


def _find_engine():
    """Locate the CCSR TRT apply engine."""
    candidates = [
        os.path.join(script_directory, "trt_engines", "ccsr_apply_f16io.rtxplan"),
        os.path.join(folder_paths.models_dir, "unet", "ccsr_apply_f16io.rtxplan"),
        os.path.join(folder_paths.models_dir, "ccsr", "trt", "ccsr_apply_f16io.rtxplan"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    return None


class CCSRTRTModelWrapper:
    """Wrap CCSR ControlLDM: apply_model dispatches to the TRT engine when the
    spatial latent matches (64x64), else falls back to the original FP16 path."""

    def __init__(self, model, engine, latent_size=64):
        self._model = model
        self._engine = engine
        self._latent_size = latent_size
        self._fallback = model.apply_model
        self.control_scales = model.control_scales

    def __getattr__(self, item):
        return getattr(self._model, item)

    def apply_model(self, x_noisy, t, cond, *a, **k):
        h, w = x_noisy.shape[-2], x_noisy.shape[-1]
        if self._engine is not None and (h, w) == (self._latent_size, self._latent_size):
            hint = cond["c_latent"][0] if cond.get("c_latent") else torch.zeros_like(x_noisy)
            context = cond["c_crossattn"][0] if cond.get("c_crossattn") else None
            tv = int(t.reshape(-1)[0].item()) if t.numel() == 1 else "?"
            print(f"[CCSR-TRT] apply_model -> TRT engine (tile {h}x{w}, t={tv})", flush=True)
            return self._engine.run(x_noisy, hint, t, context)
        print(f"[CCSR-TRT] apply_model -> FP16 fallback (tile {h}x{w})", flush=True)
        return self._fallback(x_noisy, t, cond, *a, **k)

    def to(self, *a, **k):
        self._model.to(*a, **k)
        return self

    def eval(self):
        self._model.eval()
        return self

    @property
    def device(self):
        return next(self._model.parameters()).device

    @property
    def dtype(self):
        return next(self._model.parameters()).dtype


class LoadCCSRModelTensorRT:
    """Load CCSR checkpoint and the TRT apply_model engine."""

    @classmethod
    def INPUT_TYPES(s):
        engines = _list_engines()
        if not engines:
            engines = ["(no engine in trt_engines)"]
        return {"required": {
            "engine": (engines,),
        }}

    RETURN_TYPES = ("CCSRMODEL",)
    RETURN_NAMES = ("ccsr_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "CCSR"

    def loadmodel(self, engine):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = torch.float16

        engine_dir = _TRT_ENGINE_DIR
        engine_path = os.path.join(engine_dir, engine)
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"engine not found: {engine_path}")
        eng = get_engine(engine_path, device)
        print(f"[CCSR-TRT] engine loaded: {engine}", flush=True)

        # aux weights (VAE + cond_encoder) auto-loaded next to the engine
        aux_path = os.path.join(engine_dir, "ccsr_trt_aux.safetensors")
        if not os.path.exists(aux_path):
            raise FileNotFoundError(
                f"aux weights not found: {aux_path} (need ccsr_trt_aux.safetensors in trt_engines)"
            )

        config_path = os.path.join(script_directory, "configs/model/ccsr_stage2.yaml")
        config = OmegaConf.load(config_path)
        ccsr = instantiate_from_config(config)
        sd = comfy.utils.load_torch_file(aux_path)
        ccsr.load_state_dict(sd, strict=False)
        del sd
        mm.soft_empty_cache()
        ccsr = ccsr.to(device, dtype=dtype).eval()
        for m in ccsr.modules():
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = False

        wrapped = CCSRTRTModelWrapper(ccsr, eng)
        ccsr_model = {"model": wrapped, "dtype": dtype, "trt": True}
        return (ccsr_model,)

class CCSR_Upscale_TRT:
    """CCSR upscale using the TRT apply_model engine (tile 512 fixed)."""

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ccsr_model": ("CCSRMODEL",),
            "image": ("IMAGE",),
            "resize_method": (s.upscale_methods, {"default": "lanczos"}),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
            "steps": ("INT", {"default": 15, "min": 3, "max": 4096, "step": 1}),
            "t_max": ("FLOAT", {"default": 0.64, "min": 0, "max": 1, "step": 0.01}),
            "t_min": ("FLOAT", {"default": 0.35, "min": 0, "max": 1, "step": 0.01}),
            "tile_size": ("INT", {"default": 512, "min": 512, "max": 512, "step": 1}),
            "tile_stride": ("INT", {"default": 256, "min": 8, "max": 512, "step": 8}),
            "vae_tile_size_encode": ("INT", {"default": 1024, "min": 2, "max": 4096, "step": 8}),
            "vae_tile_size_decode": ("INT", {"default": 1024, "min": 2, "max": 4096, "step": 8}),
            "color_fix_type": (["none", "adain", "wavelet"], {"default": "adain"}),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 123, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "process"
    CATEGORY = "CCSR"

    @torch.no_grad()
    def process(self, ccsr_model, image, resize_method, scale_by, steps, t_max, t_min,
                tile_size, tile_stride, vae_tile_size_encode, vae_tile_size_decode,
                color_fix_type, keep_model_loaded, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mm.unload_all_models()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = ccsr_model["dtype"]
        model = ccsr_model["model"]
        trt_active = ccsr_model.get("trt", False)

        empty_text_embed_sd = comfy.utils.load_torch_file(
            os.path.join(script_directory, "empty_text_embed.safetensors"))
        empty_text_embed = empty_text_embed_sd["empty_text_embed"].to(dtype).to(device)

        sampler = SpacedSampler(model, var_type="fixed_small")
        image, = ImageScaleBy.upscale(self, image, resize_method, scale_by)
        B, H, W, C = image.shape
        new_height = H // 64 * 64
        new_width = W // 64 * 64
        image = image.permute(0, 3, 1, 2).contiguous()
        resized_image = F.interpolate(image, size=(new_height, new_width), mode="bilinear", align_corners=False)

        strength = 1.0
        model.control_scales = [strength] * 13
        model.to(device, dtype=dtype).eval()
        print(f"[CCSR-TRT] upscale start: image={tuple(image.shape)} steps={steps} tile={tile_size} TRT={'yes' if trt_active else 'no'} engine={ccsr_model.get('trt', False)}", flush=True)

        height, width = resized_image.size(-2), resized_image.size(-1)
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

        out = []
        if B > 1:
            pbar = comfy.utils.ProgressBar(B)
        autocast_condition = dtype == torch.float16 and not mm.is_device_mps(device)
        with XFormersKernelOnce():
            with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
                for i in range(B):
                    img = resized_image[i].unsqueeze(0).to(device)
                    model._init_tiled_vae(encoder_tile_size=vae_tile_size_encode // 8,
                                          decoder_tile_size=vae_tile_size_decode // 8)
                    samples = sampler.sample_with_tile_ccsr(
                        empty_text_embed, tile_size=tile_size, tile_stride=tile_stride,
                        steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=img,
                        positive_prompt="", negative_prompt="", x_T=x_T,
                        cfg_scale=1.0, color_fix_type=color_fix_type)
                    out.append(samples.squeeze(0).cpu())
                    mm.throw_exception_if_processing_interrupted()
                    if B > 1:
                        pbar.update(1)
                        print("Sampled image ", i + 1, " out of ", B)

        original_height, original_width = H, W
        processed_height = out[0].size(1) if len(out) > 0 else samples.size(2)
        target_width = int(processed_height * (original_width / original_height))
        out_stacked = torch.stack(out, dim=0).cpu().to(torch.float32).permute(0, 2, 3, 1)
        resized_back_image, = ImageScale.upscale(self, out_stacked, "lanczos", target_width, processed_height, crop="disabled")

        if not keep_model_loaded:
            release_trt_engines()
            model.to(offload_device)
            mm.soft_empty_cache()
        return (resized_back_image,)

NODE_CLASS_MAPPINGS = {
    "LoadCCSRModelTensorRT": LoadCCSRModelTensorRT,
    "CCSR_Upscale_TRT": CCSR_Upscale_TRT
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCCSRModelTensorRT": "Load CCSR Model (TensorRT)",
    "CCSR_Upscale_TRT": "CCSR Upscale (TRT)"
}