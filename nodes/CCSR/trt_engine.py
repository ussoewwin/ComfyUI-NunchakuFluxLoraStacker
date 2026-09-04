# -*- coding: utf-8 -*-
"""CCSR TRT engine integration: replace ControlLDM.apply_model with a
TensorRT-RTX engine call. Engine inputs: x, hint, timesteps(int64), context.
Fixed at tile latent size 64x64 (=512px tile). Falls back to the original
FP16 apply_model when the tile size does not match the engine.
"""
import os
import torch

import tensorrt_rtx as trt

_ENGINES = {}


class CCSRTRTEngine:
    """Thin wrapper around a deserialized TRT engine with async execution.

    Runs on the CURRENT torch stream and synchronizes after each call so the
    sampler's interleaved CUDA ops never race the engine (avoids deadlocks).
    """

    def __init__(self, engine_path: str, device: torch.device):
        self.path = os.path.abspath(engine_path)
        self.device = device
        with open(self.path, "rb") as f:
            blob = f.read()
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.n_x = next(n for n in self.inputs if n == "x")
        self.n_hint = next(n for n in self.inputs if n == "hint")
        self.n_t = next(n for n in self.inputs if "time" in n)
        self.n_ctx = next(n for n in self.inputs if n == "context")
        self.n_out = self.outputs[0]
        t = self.engine.get_tensor_dtype(self.n_x)
        self.dtype = torch.float16 if t == trt.DataType.HALF else torch.float32

    def run(self, x, hint, t, context):
        dt = self.dtype
        cur = torch.cuda.current_stream()
        xb = x.to(device=self.device, dtype=dt).contiguous()
        hb = hint.to(device=self.device, dtype=dt).contiguous()
        tb = t.to(device=self.device, dtype=torch.int64).contiguous()
        cb = context.to(device=self.device, dtype=dt).contiguous()
        out = torch.empty(xb.shape, device=self.device, dtype=dt)
        self.ctx.set_tensor_address(self.n_x, xb.data_ptr())
        self.ctx.set_tensor_address(self.n_hint, hb.data_ptr())
        self.ctx.set_tensor_address(self.n_t, tb.data_ptr())
        self.ctx.set_tensor_address(self.n_ctx, cb.data_ptr())
        self.ctx.set_tensor_address(self.n_out, out.data_ptr())
        ok = self.ctx.execute_async_v3(cur.cuda_stream)
        cur.synchronize()
        if not ok:
            raise RuntimeError("CCSR TRT engine execution failed")
        # return in the input's dtype so the fp32 sampler math stays fp32
        return out.to(dtype=x.dtype)

    def release(self):
        for attr in ("ctx", "engine", "runtime"):
            try:
                delattr(self, attr)
            except Exception:
                pass


def get_engine(engine_path: str, device: torch.device):
    key = os.path.abspath(engine_path)
    if key not in _ENGINES:
        _ENGINES[key] = CCSRTRTEngine(key, device)
    return _ENGINES[key]


def release_trt_engines():
    for e in _ENGINES.values():
        try:
            e.release()
        except Exception:
            pass
    _ENGINES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
