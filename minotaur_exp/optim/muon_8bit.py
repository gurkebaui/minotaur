"""Ultra-Low-VRAM Optimizer & LOMO Fused Backward-Optimizer Hook:
- 8-bit Muon (für 2D Hidden-Matrizen) + 8-bit AdamW (für Embeddings/1D/Tabellen)
- APOLLO-Mini (Rank-1 Random Projection Optimizer für 2D-Matrizen)
- LOMOFusedHookController (register_post_accumulate_grad_hook, loop-kompatibel)
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from minotaur_exp.optim.precision import fp32_to_bf16_sr


def quantize_blockwise_int8(x: torch.Tensor, block_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Blockweise symmetrische INT8-Quantisierung (spart ~75% Optimizer-State-VRAM)."""
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (block_size - (n % block_size)) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    blocks = flat.view(-1, block_size).float()
    scales = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    q = torch.round(blocks / scales).clamp(-127, 127).to(torch.int8)
    return q, scales.squeeze(1).to(torch.float32)


def dequantize_blockwise_int8(
    q: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: torch.Size,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    deq = (q.float() * scales.float().unsqueeze(1)).reshape(-1)
    numel = math.prod(orig_shape)
    return deq[:numel].view(orig_shape).to(dtype)


def newton_schulz_5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Polar-Express / Muon 5-Step Quintic Newton-Schulz Orthogonalisierung."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.t()
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X


def _is_muon_candidate(name: str, param: nn.Parameter) -> bool:
    if param.ndim != 2:
        return False
    excluded = ("tok_emb", "lm_head", "ultramem", "router", "loop_emb", "step_emb")
    if any(ex in name for ex in excluded):
        return False
    return min(param.shape) >= 32


def _is_ultramem_table(name: str, param: nn.Parameter) -> bool:
    return "ultramem" in name and "values" in name and param.ndim == 2


class _BaseLowVRAMOptimizer:
    """Gemeinsame Basis für Muon8Bit und ApolloMini inkl. Single-Param-Update für LOMO."""

    def __init__(
        self,
        named_params: Iterable[tuple[str, nn.Parameter]],
        lr: float = 3e-4,
        muon_lr: float = 0.02,
        weight_decay: float = 0.01,
        pure_bf16_sr: bool = False,
        block_size: int = 256,
    ):
        self.lr = lr
        self.muon_lr = muon_lr
        self.weight_decay = weight_decay
        self.pure_bf16_sr = pure_bf16_sr
        self.block_size = block_size
        self.named_params: list[tuple[str, nn.Parameter]] = [
            (n, p) for n, p in named_params if p.requires_grad
        ]
        self.param_to_name: dict[int, str] = {id(p): n for n, p in self.named_params}
        self.state: dict[int, dict] = {}
        self.global_step: int = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, p in self.named_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def _apply_update_to_param(self, p: nn.Parameter, update_fp32: torch.Tensor, eff_lr: float, wd: float) -> None:
        with torch.no_grad():
            chunk_elems = 4_000_000
            if p.numel() <= chunk_elems:
                p_fp32 = p.data.float()
                if wd > 0.0:
                    p_fp32.mul_(1.0 - eff_lr * wd)
                p_fp32.add_(update_fp32, alpha=-eff_lr)
                if self.pure_bf16_sr and p.dtype == torch.bfloat16:
                    p.data.copy_(fp32_to_bf16_sr(p_fp32))
                else:
                    p.data.copy_(p_fp32.to(p.dtype))
                return

            p_flat = p.data.view(-1)
            u_flat = update_fp32.view(-1)
            for start in range(0, p_flat.numel(), chunk_elems):
                end = min(start + chunk_elems, p_flat.numel())
                c_fp32 = p_flat[start:end].float()
                if wd > 0.0:
                    c_fp32.mul_(1.0 - eff_lr * wd)
                c_fp32.add_(u_flat[start:end], alpha=-eff_lr)
                if self.pure_bf16_sr and p.dtype == torch.bfloat16:
                    p_flat[start:end].copy_(fp32_to_bf16_sr(c_fp32))
                else:
                    p_flat[start:end].copy_(c_fp32.to(p.dtype))

    def _step_adamw_8bit(self, p: nn.Parameter, grad: torch.Tensor, eff_lr: float, row_factor: bool = False) -> None:
        pid = id(p)
        st = self.state.setdefault(pid, {"step": 0})
        st["step"] += 1
        t = st["step"]
        beta1, beta2, eps = 0.9, 0.95, 1e-8

        if row_factor:
            # Ultra-kompakter Adafactor-Style State für UltraMemV2-Tabellen: nur aktive Rows in Chunks!
            if "v_row" not in st:
                st["v_row"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
            v_row = st["v_row"]
            with torch.no_grad():
                active_rows = torch.nonzero(grad.abs().amax(dim=1) > 0, as_tuple=False).squeeze(-1)
                if active_rows.numel() == 0:
                    return
                bias_corr = 1.0 - beta2**t
                row_chunk = 16_384
                for s_i in range(0, active_rows.numel(), row_chunk):
                    idx = active_rows[s_i : s_i + row_chunk]
                    g_sub = grad.index_select(0, idx).float()
                    row_sq = g_sub.pow(2).mean(dim=1)
                    vr_sub = v_row.index_select(0, idx).mul_(beta2).add_(row_sq, alpha=1.0 - beta2)
                    v_row.index_copy_(0, idx, vr_sub)
                    v_hat = (vr_sub / bias_corr).unsqueeze(1)
                    upd = g_sub / (v_hat.sqrt() + eps)
                    p_sub = p.data.index_select(0, idx).float()
                    if self.weight_decay > 0.0:
                        p_sub.mul_(1.0 - eff_lr * self.weight_decay)
                    p_sub.add_(upd, alpha=-eff_lr)
                    if self.pure_bf16_sr and p.dtype == torch.bfloat16:
                        p.data.index_copy_(0, idx, fp32_to_bf16_sr(p_sub))
                    else:
                        p.data.index_copy_(0, idx, p_sub.to(p.dtype))
            return

        g = grad.float()

        if "m_q" not in st:
            m = torch.zeros_like(g)
            v = torch.zeros_like(g)
        else:
            m = dequantize_blockwise_int8(st["m_q"], st["m_s"], p.shape, torch.float32)
            v = dequantize_blockwise_int8(st["v_q"], st["v_s"], p.shape, torch.float32).abs()

        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

        st["m_q"], st["m_s"] = quantize_blockwise_int8(m, self.block_size)
        st["v_q"], st["v_s"] = quantize_blockwise_int8(v, self.block_size)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        upd = m_hat / (v_hat.sqrt() + eps)
        self._apply_update_to_param(p, upd, eff_lr, self.weight_decay)

    def state_memory_bytes(self) -> int:
        total = 0
        for st in self.state.values():
            for val in st.values():
                if isinstance(val, torch.Tensor):
                    total += val.numel() * val.element_size()
        return total

    def step_single_param(self, p: nn.Parameter, grad: torch.Tensor, lr_scale: float = 1.0) -> None:
        raise NotImplementedError

    def step(self, lr_scale: float = 1.0) -> None:
        self.global_step += 1
        for _, p in self.named_params:
            if p.grad is not None:
                self.step_single_param(p, p.grad, lr_scale=lr_scale)


class Muon8BitOptimizer(_BaseLowVRAMOptimizer):
    """8-bit Muon (2D-Matrizen) + 8-bit AdamW (Embeddings/1D) + Row-Adafactor (UltraMemV2)."""

    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]], momentum: float = 0.95, **kwargs):
        super().__init__(named_params, **kwargs)
        self.momentum = momentum

    def step_single_param(self, p: nn.Parameter, grad: torch.Tensor, lr_scale: float = 1.0) -> None:
        name = self.param_to_name.get(id(p), "")
        if _is_ultramem_table(name, p):
            self._step_adamw_8bit(p, grad, eff_lr=self.lr * lr_scale, row_factor=True)
            return
        if not _is_muon_candidate(name, p):
            self._step_adamw_8bit(p, grad, eff_lr=self.lr * lr_scale, row_factor=False)
            return

        pid = id(p)
        st = self.state.setdefault(pid, {"step": 0})
        st["step"] += 1
        g = grad.float()
        if "buf_q" not in st:
            buf = torch.zeros_like(g)
        else:
            buf = dequantize_blockwise_int8(st["buf_q"], st["buf_s"], p.shape, torch.float32)

        buf.mul_(self.momentum).add_(g)
        st["buf_q"], st["buf_s"] = quantize_blockwise_int8(buf, self.block_size)

        # Nesterov-Blend + Newton-Schulz Orthogonalisierung
        nesterov = g + self.momentum * buf
        ortho = newton_schulz_5(nesterov, steps=5).float()
        aspect = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
        self._apply_update_to_param(p, ortho * aspect, eff_lr=self.muon_lr * lr_scale, wd=self.weight_decay)


class ApolloMiniOptimizer(_BaseLowVRAMOptimizer):
    """APOLLO-Mini: Rank-1 Random Projection Optimizer (<1% Optimizer-VRAM für 2D-Weights)."""

    def __init__(self, named_params: Iterable[tuple[str, nn.Parameter]], proj_interval: int = 200, **kwargs):
        super().__init__(named_params, **kwargs)
        self.proj_interval = proj_interval

    def step_single_param(self, p: nn.Parameter, grad: torch.Tensor, lr_scale: float = 1.0) -> None:
        name = self.param_to_name.get(id(p), "")
        if _is_ultramem_table(name, p):
            self._step_adamw_8bit(p, grad, eff_lr=self.lr * lr_scale, row_factor=True)
            return
        if not _is_muon_candidate(name, p):
            self._step_adamw_8bit(p, grad, eff_lr=self.lr * lr_scale, row_factor=False)
            return

        pid = id(p)
        st = self.state.setdefault(pid, {"step": 0, "seed": (pid & 0x7FFFFFFF)})
        st["step"] += 1
        t = st["step"]
        g = grad.float()
        rows, cols = g.shape
        beta1, beta2, eps = 0.9, 0.95, 1e-8

        # Rank-1 Projektion auf die kleinere Achse -> nur 1 Vektor State!
        proj_block = (t - 1) // self.proj_interval
        gen = torch.Generator(device=g.device).manual_seed(st["seed"] + proj_block * 9973)

        if rows >= cols:
            r_vec = torch.randn(rows, 1, device=g.device, dtype=torch.float32, generator=gen) / math.sqrt(rows)
            g_proj = (r_vec.t() @ g).squeeze(0)  # (cols,)
            proj_len = cols
        else:
            r_vec = torch.randn(1, cols, device=g.device, dtype=torch.float32, generator=gen) / math.sqrt(cols)
            g_proj = (g @ r_vec.t()).squeeze(1)  # (rows,)
            proj_len = rows

        if "m" not in st or st["m"].numel() != proj_len:
            st["m"] = torch.zeros(proj_len, device=g.device, dtype=torch.float32)
            st["v"] = torch.zeros(proj_len, device=g.device, dtype=torch.float32)

        m, v = st["m"], st["v"]
        m.mul_(beta1).add_(g_proj, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(g_proj, g_proj, value=1.0 - beta2)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        step_proj = m_hat / (v_hat.sqrt() + eps)

        # Channel/Norm-Scaling (APOLLO-Mini): skaliere volles g anhand des projizierten Ratios
        scale_ratio = step_proj.norm() / (g_proj.norm() + 1e-8)
        scale_ratio = scale_ratio.clamp(max=10.0)
        self._apply_update_to_param(p, g * scale_ratio, eff_lr=self.lr * lr_scale, wd=self.weight_decay)


class StandardAdamWWrapper:
    """Wrapper um torch.optim.AdamW mit einheitlichem Interface für Profiler."""

    def __init__(
        self,
        named_params: Iterable[tuple[str, nn.Parameter]],
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ):
        params = [p for _, p in named_params if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
        self.base_lr = lr
        self.named_params = list(named_params)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self, lr_scale: float = 1.0) -> None:
        for pg in self.opt.param_groups:
            pg["lr"] = self.base_lr * lr_scale
        self.opt.step()

    def state_memory_bytes(self) -> int:
        total = 0
        for st in self.opt.state.values():
            for val in st.values():
                if isinstance(val, torch.Tensor):
                    total += val.numel() * val.element_size()
        return total


class LOMOFusedHookController:
    """Fused Backward-Optimizer Hook (LOMO-Prinzip), sicher für Loops & Gradient Accumulation."""

    def __init__(
        self,
        optimizer: _BaseLowVRAMOptimizer,
        grad_accum_steps: int = 1,
        grad_clip: float = 1.0,
    ):
        self.optimizer = optimizer
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip = grad_clip
        self.active = False
        self.current_lr_scale = 1.0
        self._accum_counts: dict[int, int] = {}
        self._handles: list = []
        self.last_grad_norm_sq: float = 0.0
        self.hook_fires: int = 0
        self._register_hooks()

    def _register_hooks(self) -> None:
        for _, p in self.optimizer.named_params:
            if not p.requires_grad:
                continue
            pid = id(p)
            self._accum_counts[pid] = 0

            def _make_hook(param: nn.Parameter, param_id: int):
                def _hook(p_inner: nn.Parameter) -> None:
                    if not self.active or p_inner.grad is None:
                        return
                    self.hook_fires += 1
                    self._accum_counts[param_id] += 1
                    if self._accum_counts[param_id] < self.grad_accum_steps:
                        return
                    self._accum_counts[param_id] = 0
                    g = p_inner.grad
                    if self.grad_accum_steps > 1:
                        g = g / float(self.grad_accum_steps)
                    gnorm_sq = float(g.detach().float().pow(2).sum().item())
                    self.last_grad_norm_sq += gnorm_sq
                    if self.grad_clip > 0.0:
                        gnorm = math.sqrt(gnorm_sq)
                        if gnorm > self.grad_clip:
                            g = g * (self.grad_clip / (gnorm + 1e-8))
                    self.optimizer.step_single_param(param, g, lr_scale=self.current_lr_scale)
                    p_inner.grad = None

                return _hook

            self._handles.append(p.register_post_accumulate_grad_hook(_make_hook(p, pid)))

    def begin_step(self, lr_scale: float) -> None:
        self.active = True
        self.current_lr_scale = lr_scale
        self.last_grad_norm_sq = 0.0
        self.hook_fires = 0

    def end_step(self) -> float:
        self.active = False
        return math.sqrt(self.last_grad_norm_sq)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_optimizer(model: nn.Module, cfg) -> tuple[object, LOMOFusedHookController | None]:
    """Erstellt den konfigurierten Optimizer + optionalen LOMO-Hook."""
    named = list(model.named_parameters())
    pure_sr = cfg.precision_mode == "pure_bf16_sr"

    if cfg.optimizer_type == "adamw":
        opt = StandardAdamWWrapper(named, lr=cfg.lr, weight_decay=cfg.weight_decay)
        return opt, None
    elif cfg.optimizer_type == "muon_8bit":
        opt = Muon8BitOptimizer(
            named,
            lr=cfg.lr,
            muon_lr=cfg.muon_lr,
            weight_decay=cfg.weight_decay,
            pure_bf16_sr=pure_sr,
        )
    elif cfg.optimizer_type == "apollo_mini":
        opt = ApolloMiniOptimizer(
            named,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            pure_bf16_sr=pure_sr,
        )
    else:
        raise ValueError(f"Unbekannter optimizer_type: {cfg.optimizer_type!r}")

    lomo = None
    if cfg.use_lomo_hook:
        lomo = LOMOFusedHookController(
            opt,
            grad_accum_steps=cfg.grad_accum_steps,
            grad_clip=cfg.grad_clip,
        )
    return opt, lomo
