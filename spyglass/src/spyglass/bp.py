"""Batched flooding min-sum belief propagation with memory, in torch.

Semantics are pinned message-for-message to the pure-numpy reference in
``tests/reference_bp.py``; the iteration schedule is: total LLR with
memory, extrinsic variable-to-check, scaled min-sum check-to-variable
with the syndrome folded into the sign, then a hard-decision syndrome
check. A shot is converged the moment its hard decision reproduces its
syndrome exactly, and converged shots freeze (their first-convergence
hard decision, posterior, and iteration count are what the result
carries).

Everything runs on padded dense tables from
:class:`spyglass.DecodingProblem`. Padded check slots carry +inf into the
minimum and a positive sign into the parity, padded variable slots read a
zeroed sentinel row, and every reduction is a ``sum``/``topk`` along a
fixed axis: no scatter atomics anywhere, which is what makes CUDA runs
bitwise reproducible.

torch is imported lazily so the package imports without it; only calling
into this module requires the ``gpu`` extra (CPU execution works with any
torch build).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._problem import DecodingProblem
from .stages import StageOutput


@dataclass(frozen=True)
class BpResult:
    hard: np.ndarray         # (B, n) uint8, first-convergence snapshot
    converged: np.ndarray    # (B,) bool
    iterations: np.ndarray   # (B,) int32, iterations to convergence (or max)
    posteriors: np.ndarray   # (B, n) float64 total LLR snapshot
    state_c2v: Optional[np.ndarray] = None  # (B, m, max_cd) when requested


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "spyglass.bp requires torch; install the 'gpu' extra "
            "(CPU-only torch builds work too)") from exc
    return torch


def bp_decode_batch(problem: DecodingProblem, syndromes: np.ndarray, *,
                    max_iter: int = 200, ms_scale: float = 0.625,
                    gamma: float = 0.0,
                    llr_prev_init: Optional[np.ndarray] = None,
                    device: str = "cpu",
                    return_state: bool = False) -> BpResult:
    """Decode a batch of syndromes with flooding min-sum.

    Args:
        problem: the decoding problem.
        syndromes: ``(B, m)`` binary.
        max_iter: iteration cap.
        ms_scale: min-sum scaling factor.
        gamma: memory strength; ``0`` is plain BP. A scalar here; the
            relay stage passes per-variable tensors through
            ``llr_prev_init`` and its own gamma handling.
        llr_prev_init: optional ``(B, n)`` starting memory (defaults to
            the channel LLRs, which makes the first iteration identical
            to memoryless BP).
        device: ``"cpu"`` or ``"cuda"``.
        return_state: also return the final check-to-variable messages
            (message-parity tests and the relay stage use this).
    """
    torch = _torch()
    dev = torch.device(device)
    dt = torch.float32

    H = problem.H
    m, n = H.shape
    syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
    B = syndromes.shape[0]
    max_cd = problem.chk_nbrs.shape[1]
    max_vd = problem.var_nbrs.shape[1]

    chk_nbrs = torch.as_tensor(problem.chk_nbrs, dtype=torch.long,
                               device=dev)          # (m, max_cd), sentinel n
    var_nbrs = torch.as_tensor(problem.var_nbrs, dtype=torch.long,
                               device=dev)          # (n, max_vd), sentinel m
    chk_edge_pos = torch.as_tensor(problem.chk_edge_pos, dtype=torch.long,
                                   device=dev)
    var_edge_pos = torch.as_tensor(problem.var_edge_pos, dtype=torch.long,
                                   device=dev)
    chk_real = chk_nbrs < n                          # (m, max_cd) bool
    llr0 = torch.as_tensor(problem.llr0, dtype=dt, device=dev)  # (n,)
    syn = torch.as_tensor(syndromes, dtype=torch.long, device=dev)  # (B, m)

    c2v = torch.zeros((B, m, max_cd), dtype=dt, device=dev)
    llr_prev = (torch.as_tensor(llr_prev_init, dtype=dt, device=dev)
                if llr_prev_init is not None
                else llr0.expand(B, n).clone())

    # frozen-state snapshots
    active = torch.ones(B, dtype=torch.bool, device=dev)
    hard_snap = torch.zeros((B, n), dtype=torch.uint8, device=dev)
    post_snap = llr0.expand(B, n).clone()
    iters = torch.full((B,), max_iter, dtype=torch.int32, device=dev)

    # flat gather index for reading c2v in variable layout: entry (v, j)
    # addresses c2v[:, var_nbrs[v, j], var_edge_pos[v, j]]; the sentinel
    # row m is appended as zeros so padded slots contribute nothing
    var_gather = (var_nbrs * max_cd + var_edge_pos).reshape(-1)  # (n*max_vd,)

    def sum_c2v_per_var(c2v_t):
        padded = torch.cat(
            [c2v_t.reshape(B, m * max_cd),
             torch.zeros((B, max_cd), dtype=dt, device=dev)], dim=1)
        gathered = padded[:, var_gather].reshape(B, n, max_vd)
        return gathered.sum(dim=-1)                  # (B, n)

    def check_syndrome(hard_t):
        padded = torch.cat(
            [hard_t.to(torch.long),
             torch.zeros((B, 1), dtype=torch.long, device=dev)], dim=1)
        per_check = padded[:, chk_nbrs.reshape(-1)]\
            .reshape(B, m, max_cd).sum(dim=-1) % 2
        return (per_check == syn).all(dim=1)         # (B,)

    inf = torch.tensor(float("inf"), dtype=dt, device=dev)

    for it in range(max_iter):
        total = gamma * llr_prev + (1.0 - gamma) * llr0 + sum_c2v_per_var(c2v)

        # v->c in check layout; padded slots become +inf (positive sign,
        # excluded from every minimum)
        total_pad = torch.cat([total, torch.full((B, 1), float("inf"),
                                                 dtype=dt, device=dev)],
                              dim=1)
        v2c = total_pad[:, chk_nbrs.reshape(-1)].reshape(B, m, max_cd) - c2v
        v2c = torch.where(chk_real, v2c, inf)

        # signs by negative counting; syndrome flips the check's parity
        neg = (v2c < 0)
        neg_total = neg.sum(dim=-1)                          # (B, m)
        parity = (neg_total + syn) % 2                       # (B, m)
        ext_neg = (parity.unsqueeze(-1) + neg.to(torch.long)) % 2
        ext_sign = 1.0 - 2.0 * ext_neg.to(dt)

        mag = v2c.abs()
        two_min, two_idx = torch.topk(mag, k=min(2, max_cd), dim=-1,
                                      largest=False)
        min1 = two_min[..., 0:1]
        min2 = two_min[..., 1:2] if max_cd > 1 else min1
        is_argmin = (torch.arange(max_cd, device=dev)
                     .view(1, 1, max_cd) == two_idx[..., 0:1])
        ext_min = torch.where(is_argmin, min2, min1)

        c2v_new = ms_scale * ext_sign * ext_min
        c2v_new = torch.where(chk_real, c2v_new, torch.zeros((), dtype=dt,
                                                             device=dev))
        # frozen shots keep their final messages
        c2v = torch.where(active.view(B, 1, 1), c2v_new, c2v)
        llr_prev = torch.where(active.view(B, 1), total, llr_prev)

        posterior = llr0 + sum_c2v_per_var(c2v)
        hard = (posterior < 0).to(torch.uint8)
        ok = check_syndrome(hard)

        newly = ok & active
        if newly.any():
            hard_snap = torch.where(newly.view(B, 1), hard, hard_snap)
            post_snap = torch.where(newly.view(B, 1), posterior, post_snap)
            iters = torch.where(newly, torch.tensor(it + 1, dtype=torch.int32,
                                                    device=dev), iters)
            active = active & ~ok
        if not active.any():
            break

    # unresolved shots carry their last posterior and hard decision
    posterior = llr0 + sum_c2v_per_var(c2v)
    hard = (posterior < 0).to(torch.uint8)
    hard_final = torch.where(active.view(B, 1), hard, hard_snap)
    post_final = torch.where(active.view(B, 1), posterior, post_snap)
    converged = ~active

    return BpResult(
        hard=hard_final.cpu().numpy().astype(np.uint8),
        converged=converged.cpu().numpy(),
        iterations=iters.cpu().numpy(),
        posteriors=post_final.cpu().numpy().astype(np.float64),
        state_c2v=(c2v.cpu().numpy() if return_state else None))


class BatchBpStage:
    """Plain batched BP as a pipeline stage."""

    def __init__(self, *, max_iter: int = 200, ms_scale: float = 0.625,
                 gamma: float = 0.0, device: str = "cpu"):
        self.name = "bp"
        self.max_iter = max_iter
        self.ms_scale = ms_scale
        self.gamma = gamma
        self.device = device

    def decode_batch(self, problem: DecodingProblem, syndromes: np.ndarray,
                     posteriors_in: Optional[np.ndarray]) -> StageOutput:
        r = bp_decode_batch(
            problem, syndromes, max_iter=self.max_iter,
            ms_scale=self.ms_scale, gamma=self.gamma, device=self.device)
        return StageOutput(
            corrections=r.hard, resolved=r.converged, posteriors=r.posteriors,
            stats={"mean_iterations": float(r.iterations.mean()),
                   "resolved_fraction": float(r.converged.mean())})
