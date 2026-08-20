"""The decoding problem: a parity-check matrix plus per-mechanism priors.

Everything downstream of this module sees only ``(H, priors)`` and the
derived index tables, never a code. Code-capacity noise, and any future
detector error model, both arrive as a :class:`DecodingProblem`.

Index-table layout. Neighbor lists are stored as PADDED DENSE arrays
rather than flat edge lists: ``chk_nbrs`` is ``(m, max_chk_deg)`` with
unused slots holding the sentinel ``n``, and ``var_nbrs`` is
``(n, max_var_deg)`` with sentinel ``m``. Check and variable degrees on
LDPC codes are small and near-uniform, so the padding waste is bounded,
and every message reduction becomes a min or sum along a fixed axis of a
contiguous array. On CUDA that choice is what makes the decoder
deterministic: no scatter/gather atomics anywhere in the hot loop.

The edge-position inverse maps close the loop between the two layouts:
``var_nbrs[chk_nbrs[c, j], chk_edge_pos[c, j]] == c`` and symmetrically
``chk_nbrs[var_nbrs[v, j], var_edge_pos[v, j]] == v`` for every real
edge.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecodingProblem:
    """A parity-check matrix, priors, and the derived index tables."""

    H: np.ndarray            # (m, n) uint8
    priors: np.ndarray       # (n,) float64 in (0, 0.5)
    llr0: np.ndarray         # (n,) float64, log((1 - p) / p)
    chk_deg: np.ndarray      # (m,) int32
    var_deg: np.ndarray      # (n,) int32
    chk_nbrs: np.ndarray     # (m, max_chk_deg) int32, sentinel n
    var_nbrs: np.ndarray     # (n, max_var_deg) int32, sentinel m
    chk_edge_pos: np.ndarray  # (m, max_chk_deg) int32
    var_edge_pos: np.ndarray  # (n, max_var_deg) int32


def build_problem(H: np.ndarray, priors: np.ndarray) -> DecodingProblem:
    """Validate ``(H, priors)`` and build the padded index tables.

    Args:
        H: ``(m, n)`` binary parity-check matrix (any integer dtype with
            entries in {0, 1}).
        priors: ``(n,)`` per-mechanism flip probabilities, each strictly
            inside ``(0, 0.5)``.

    Raises:
        ValueError: on shape mismatch, non-binary entries, or priors
            outside the open interval.
    """
    H = np.asarray(H)
    priors = np.asarray(priors, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D; got shape {H.shape}")
    if not np.isin(H, (0, 1)).all():
        raise ValueError("H must be binary (entries in {0, 1})")
    m, n = H.shape
    if priors.shape != (n,):
        raise ValueError(
            f"priors must have shape ({n},) to match H's columns; got "
            f"{priors.shape}")
    if not ((priors > 0.0) & (priors < 0.5)).all():
        raise ValueError("priors must lie strictly inside (0, 0.5)")

    H = H.astype(np.uint8)
    chk_deg = H.sum(axis=1).astype(np.int32)
    var_deg = H.sum(axis=0).astype(np.int32)
    max_cd = int(chk_deg.max()) if m else 0
    max_vd = int(var_deg.max()) if n else 0

    chk_nbrs = np.full((m, max_cd), n, dtype=np.int32)
    var_nbrs = np.full((n, max_vd), m, dtype=np.int32)
    chk_edge_pos = np.zeros((m, max_cd), dtype=np.int32)
    var_edge_pos = np.zeros((n, max_vd), dtype=np.int32)

    var_fill = np.zeros(n, dtype=np.int32)
    for c in range(m):
        for j, v in enumerate(np.flatnonzero(H[c])):
            chk_nbrs[c, j] = v
            slot = var_fill[v]
            var_nbrs[v, slot] = c
            chk_edge_pos[c, j] = slot
            var_edge_pos[v, slot] = j
            var_fill[v] += 1

    llr0 = np.log((1.0 - priors) / priors)
    return DecodingProblem(
        H=H, priors=priors, llr0=llr0, chk_deg=chk_deg, var_deg=var_deg,
        chk_nbrs=chk_nbrs, var_nbrs=var_nbrs,
        chk_edge_pos=chk_edge_pos, var_edge_pos=var_edge_pos)
