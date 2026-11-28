"""Cut Cross-Entropy (CCE) ohne (N x V)-Logit-Materialisierung, inkl. Rho-1++ Sparse-Slicing & Top-K Distill."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from cut_cross_entropy import linear_cross_entropy as _ext_linear_ce
except Exception:
    _ext_linear_ce = None


class _ChunkedCutCrossEntropyFn(torch.autograd.Function):
    """Berechnet Cross-Entropy (+ optional Top-K Distill) blockweise ohne (N x V)-Logit-VRAM."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
        token_mask: torch.Tensor | None,
        distill_ids: torch.Tensor | None,
        distill_logprobs: torch.Tensor | None,
        distill_alpha: float,
        distill_temp: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_2d = hidden.reshape(-1, hidden.shape[-1])
        y_1d = targets.reshape(-1)
        n = h_2d.shape[0]
        device = h_2d.device

        per_token_ce = torch.empty(n, device=device, dtype=torch.float32)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        has_distill = distill_ids is not None and distill_logprobs is not None
        d_ids_2d = distill_ids.reshape(n, -1) if has_distill else None
        d_lp_2d = distill_logprobs.reshape(n, -1) if has_distill else None

        with torch.no_grad():
            w_f = weight
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                h_c = h_2d[start:end]
                y_c = y_1d[start:end]
                logits_c = (h_c @ w_f.to(h_c.dtype).t()).float()
                ce_c = F.cross_entropy(logits_c, y_c, reduction="none")
                per_token_ce[start:end] = ce_c

                if has_distill:
                    # Top-K KL-Divergenz auf den Teacher-Top-K-Tokens (mit Temperatur)
                    logits_t = logits_c / distill_temp
                    lse_s = torch.logsumexp(logits_t, dim=-1, keepdim=True)
                    ids_c = d_ids_2d[start:end].long()
                    s_topk_lp = logits_t.gather(1, ids_c) - lse_s
                    t_topk_p = d_lp_2d[start:end].float().exp()
                    kl_c = (t_topk_p * (d_lp_2d[start:end].float() - s_topk_lp)).sum(dim=-1) * (distill_temp**2)
                    chunk_loss = (1.0 - distill_alpha) * ce_c + distill_alpha * kl_c
                else:
                    chunk_loss = ce_c

                if token_mask is not None:
                    m_c = token_mask[start:end].float()
                    total_loss = total_loss + (chunk_loss * m_c).sum()
                else:
                    total_loss = total_loss + chunk_loss.sum()

        denom = token_mask.float().sum().clamp(min=1.0) if token_mask is not None else float(max(1, n))
        mean_loss = total_loss / denom

        ctx.save_for_backward(h_2d, weight, y_1d, token_mask, d_ids_2d, d_lp_2d)
        ctx.chunk_size = chunk_size
        ctx.denom = denom
        ctx.orig_shape = hidden.shape
        ctx.has_distill = has_distill
        ctx.distill_alpha = distill_alpha
        ctx.distill_temp = distill_temp
        ctx.mark_non_differentiable(per_token_ce)
        return mean_loss, per_token_ce.view_as(targets)

    @staticmethod
    def backward(ctx, grad_mean_loss: torch.Tensor, _grad_per_token: torch.Tensor):
        h_2d, weight, y_1d, token_mask, d_ids_2d, d_lp_2d = ctx.saved_tensors
        n, d = h_2d.shape
        device = h_2d.device
        grad_h = torch.zeros_like(h_2d)
        grad_w = torch.zeros(weight.shape, device=device, dtype=torch.float32)
        scale = grad_mean_loss.float() / ctx.denom

        for start in range(0, n, ctx.chunk_size):
            end = min(start + ctx.chunk_size, n)
            if token_mask is not None:
                m_c = token_mask[start:end]
                active_idx = torch.nonzero(m_c, as_tuple=False).squeeze(-1)
                if active_idx.numel() == 0:
                    continue
                # Physisches Slicing für echten Rho-1++ Backward-Speedup!
                h_c = h_2d[start:end].index_select(0, active_idx)
                y_c = y_1d[start:end].index_select(0, active_idx)
                d_ids_c = d_ids_2d[start:end].index_select(0, active_idx) if ctx.has_distill else None
                d_lp_c = d_lp_2d[start:end].index_select(0, active_idx) if ctx.has_distill else None
            else:
                active_idx = None
                h_c = h_2d[start:end]
                y_c = y_1d[start:end]
                d_ids_c = d_ids_2d[start:end] if ctx.has_distill else None
                d_lp_c = d_lp_2d[start:end] if ctx.has_distill else None

            logits_c = (h_c @ weight.to(h_c.dtype).t()).float()
            probs = F.softmax(logits_c, dim=-1)
            grad_logits = probs
            grad_logits.scatter_add_(1, y_c.unsqueeze(1), torch.full((y_c.shape[0], 1), -1.0, device=device))

            if ctx.has_distill:
                grad_ce = grad_logits * (1.0 - ctx.distill_alpha)
                logits_t = logits_c / ctx.distill_temp
                probs_t = F.softmax(logits_t, dim=-1)
                t_p = d_lp_c.float().exp()
                t_sum = t_p.sum(dim=-1, keepdim=True)
                # d/dlogits KL_topk = T * (probs_t * sum(t_p) - scatter(t_p))
                grad_kl = probs_t * t_sum
                grad_kl.scatter_add_(1, d_ids_c.long(), -t_p)
                grad_logits = grad_ce + ctx.distill_alpha * ctx.distill_temp * grad_kl

            grad_logits = (grad_logits * scale).to(h_c.dtype)
            gh_c = grad_logits @ weight.to(h_c.dtype)
            if active_idx is not None:
                grad_h[start:end].index_copy_(0, active_idx, gh_c.to(grad_h.dtype))
            else:
                grad_h[start:end] = gh_c.to(grad_h.dtype)

            grad_w.addmm_(grad_logits.t().float(), h_c.float())

        return (
            grad_h.view(ctx.orig_shape),
            grad_w.to(weight.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def cut_cross_entropy_loss(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 1024,
    token_mask: torch.Tensor | None = None,
    distill_ids: torch.Tensor | None = None,
    distill_logprobs: torch.Tensor | None = None,
    distill_alpha: float = 0.5,
    distill_temp: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Berechnet CCE Loss + unmasked Per-Token-CE (für exaktes BPB- & Rho-1++-Scoring)."""
    if (
        _ext_linear_ce is not None
        and token_mask is None
        and distill_ids is None
        and hidden.is_cuda
    ):
        # Optionaler nativer cut_cross_entropy Fast-Path falls installiert
        loss = _ext_linear_ce(hidden, lm_head_weight, targets)
        with torch.no_grad():
            _, per_tok = _ChunkedCutCrossEntropyFn.apply(
                hidden.detach(), lm_head_weight.detach(), targets, chunk_size, None, None, None, 0.0, 1.0
            )
        return loss, per_tok

    return _ChunkedCutCrossEntropyFn.apply(
        hidden,
        lm_head_weight,
        targets,
        chunk_size,
        token_mask,
        distill_ids,
        distill_logprobs,
        distill_alpha,
        distill_temp,
    )


def standard_cross_entropy_loss(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard-CE mit voller (N x V)-Logit-Matrix (für Exp 0a Baseline-VRAM-Vergleich)."""
    logits = F.linear(hidden, lm_head_weight.to(hidden.dtype))
    per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), reduction="none")
    return per_tok.mean(), per_tok.view_as(targets).detach()
