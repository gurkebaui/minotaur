"""Low-Bit-Präzision für RTX 4060 Ti (sm_89) & H100 (sm_90):
- Pure BF16 + Stochastic Rounding (bitweises GPU-SR ohne FP32-Master-Weights)
- Ada-FP8 Linear via torch._scaled_mm (float8_e4m3fn Forward / float8_e5m2 Backward)
- COAT-Style Saved-Tensor-Quantisierung im Autograd-Graph
"""

from __future__ import annotations

import contextlib
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


def fp32_to_bf16_sr(x: torch.Tensor) -> torch.Tensor:
    """Konvertiert FP32 -> BF16 mittels unverzerrtem Stochastic Rounding (vektorisiert auf GPU)."""
    if x.dtype != torch.float32:
        x = x.float()
    bits = x.view(torch.int32)
    rand_bits = torch.randint(0, 0x10000, x.shape, device=x.device, dtype=torch.int32)
    # NaN/Inf schützen: Exponent all-ones (0x7F800000) nicht verändern
    is_finite = (bits & 0x7F800000) != 0x7F800000
    rounded = torch.where(is_finite, (bits + rand_bits) & ~0xFFFF, bits & ~0xFFFF)
    return rounded.view(torch.float32).to(torch.bfloat16)


def _quantize_fp8_tensor(x: torch.Tensor, fp8_dtype: torch.dtype, max_val: float) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=max_val, neginf=-max_val)
    amax = x_f.abs().amax().clamp(min=1e-6, max=1e6)
    scale = amax / max_val
    inv_scale = max_val / amax
    q = (x_f * inv_scale).clamp(-max_val, max_val).to(fp8_dtype)
    return q, scale


def _pad_to_16(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    m = x.shape[0]
    rem = m % 16
    if rem == 0:
        return x, m
    pad = 16 - rem
    return F.pad(x, (0, 0, 0, pad)), m


class _AdaFP8MatmulFn(torch.autograd.Function):
    """Differenzierbares FP8 Linear via torch._scaled_mm auf sm_89 / sm_90."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        m_orig, k = x_2d.shape
        n = weight.shape[0]

        use_fp8 = x_2d.is_cuda and (k % 16 == 0) and (n % 16 == 0) and m_orig > 0
        ctx.use_fp8 = use_fp8
        ctx.has_bias = bias is not None
        ctx.orig_shape = orig_shape
        ctx.x_dtype = x.dtype

        if not use_fp8:
            ctx.save_for_backward(x, weight)
            out = F.linear(x, weight.to(x.dtype), bias.to(x.dtype) if bias is not None else None)
            return out

        x_pad, m = _pad_to_16(x_2d)
        q_x, s_x = _quantize_fp8_tensor(x_pad, torch.float8_e4m3fn, _FP8_E4M3_MAX)
        q_w, s_w = _quantize_fp8_tensor(weight, torch.float8_e4m3fn, _FP8_E4M3_MAX)

        # _scaled_mm erwartet (M, K) row-major x (K, N) column-major -> q_w.t() ist column-major!
        out_pad = torch._scaled_mm(
            q_x,
            q_w.t(),
            scale_a=s_x,
            scale_b=s_w,
            out_dtype=torch.bfloat16,
        )
        out = out_pad[:m]
        if bias is not None:
            out = out + bias.to(out.dtype)
        ctx.save_for_backward(x, weight)
        return out.reshape(*orig_shape[:-1], n).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        if not ctx.use_fp8:
            x_2d = x.reshape(-1, x.shape[-1])
            go_2d = grad_out.reshape(-1, grad_out.shape[-1])
            grad_x = (go_2d @ weight.to(go_2d.dtype)).reshape(ctx.orig_shape).to(x.dtype)
            grad_w = (go_2d.t() @ x_2d.to(go_2d.dtype)).to(weight.dtype)
            grad_b = go_2d.sum(dim=0).to(weight.dtype) if ctx.has_bias else None
            return grad_x, grad_w, grad_b

        x_2d = x.reshape(-1, x.shape[-1])
        go_2d = grad_out.reshape(-1, grad_out.shape[-1])
        go_pad, m = _pad_to_16(go_2d)
        x_pad, _ = _pad_to_16(x_2d)

        # grad_x = grad_out @ weight  -> (M, N) @ (N, K)
        q_go, s_go = _quantize_fp8_tensor(go_pad, torch.float8_e5m2, _FP8_E5M2_MAX)
        # weight ist (N, K) row-major; wir brauchen (N, K) col-major -> weight.t().contiguous().t()
        w_col = weight.t().contiguous()
        q_w_col, s_w = _quantize_fp8_tensor(w_col, torch.float8_e4m3fn, _FP8_E4M3_MAX)
        grad_x_pad = torch._scaled_mm(
            q_go,
            q_w_col.t(),  # (N, K) mit col-major Stride (1, N)
            scale_a=s_go,
            scale_b=s_w,
            out_dtype=torch.bfloat16,
        )
        grad_x = grad_x_pad[:m].reshape(ctx.orig_shape).to(x.dtype)

        # grad_w = grad_out^T @ x  -> (N, M) @ (M, K), wobei M auf 16 gepaddet ist
        go_t = go_pad.t().contiguous()
        x_col = x_pad.t().contiguous()
        q_go_t, s_go_t = _quantize_fp8_tensor(go_t, torch.float8_e5m2, _FP8_E5M2_MAX)
        q_x_col, s_x_col = _quantize_fp8_tensor(x_col, torch.float8_e4m3fn, _FP8_E4M3_MAX)
        grad_w = torch._scaled_mm(
            q_go_t,
            q_x_col.t(),  # (M, K) mit col-major Stride (1, M)
            scale_a=s_go_t,
            scale_b=s_x_col,
            out_dtype=torch.bfloat16,
        ).to(weight.dtype)

        grad_b = go_2d.sum(dim=0).to(weight.dtype) if ctx.has_bias else None
        return grad_x, grad_w, grad_b


class AdaFP8Linear(nn.Module):
    """Drop-in-Ersatz für nn.Linear mit Ada-FP8 (torch._scaled_mm)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "AdaFP8Linear":
        mod = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            mod.weight.copy_(linear.weight.to(torch.bfloat16))
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(linear.bias.to(torch.bfloat16))
        return mod.to(linear.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _AdaFP8MatmulFn.apply(x, self.weight, self.bias)


def replace_linear_with_ada_fp8(
    model: nn.Module,
    exclude_names: Iterable[str] = ("lm_head", "tok_emb", "router", "gate", "aux_head", "ultramem"),
) -> int:
    """Ersetzt geeignete nn.Linear-Layer durch AdaFP8Linear (Dimensionen % 16 == 0)."""
    excluded = tuple(exclude_names)
    replaced = 0

    def _recurse(parent: nn.Module, prefix: str) -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(ex in full for ex in excluded):
                continue
            if isinstance(child, nn.Linear):
                if child.in_features % 16 == 0 and child.out_features % 16 == 0:
                    setattr(parent, name, AdaFP8Linear.from_linear(child))
                    replaced += 1
            else:
                _recurse(child, full)

    _recurse(model, "")
    return replaced


def convert_model_pure_bf16(model: nn.Module) -> None:
    """Wandelt alle trainierbaren Parameter in Pure BF16 um (spart 50% Weight-VRAM vs FP32 Master)."""
    model.to(dtype=torch.bfloat16)


class COATSavedTensorContext:
    """COAT-Prinzip: Komprimiert große gespeicherte Autograd-Aktivierungen dynamisch nach FP8 E4M3."""

    def __init__(self, enabled: bool = True, min_numel: int = 4096):
        self.enabled = enabled
        self.min_numel = min_numel
        self._ctx = None

    def _pack(self, tensor: torch.Tensor):
        if (
            not self.enabled
            or not tensor.is_cuda
            or tensor.numel() < self.min_numel
            or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return (False, tensor, None, None)
        x_f = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=_FP8_E4M3_MAX, neginf=-_FP8_E4M3_MAX)
        amax = x_f.abs().amax().clamp(min=1e-6, max=1e6)
        scale = amax / _FP8_E4M3_MAX
        q = (x_f * (_FP8_E4M3_MAX / amax)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        return (True, q, scale, tensor.dtype)

    def _unpack(self, packed):
        is_q, data, scale, orig_dtype = packed
        if not is_q:
            return data
        return (data.float() * scale).to(orig_dtype)

    def __enter__(self):
        if self.enabled:
            self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
            self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            res = self._ctx.__exit__(exc_type, exc_val, exc_tb)
            self._ctx = None
            return res
        return False
