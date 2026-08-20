"""Exact most-likely-error endgame via CP-SAT.

The decoding problem in its integer-programming form: binary variable per
error mechanism, one parity constraint per check (``sum == syndrome +
2 * slack``), objective ``minimize sum(w_i x_i)`` with integer weights
derived from log-likelihood ratios. Solutions are exact minimum-weight
representatives of the syndrome's coset; the comparable quantity across
degenerate solutions is the objective value, never the support.

Soft information sharpens the solve: the stage converts the previous
stage's posterior LLRs back to probabilities (clipped away from 0 and
1/2) and weights each mechanism by its own evidence, so shots that BP
almost decoded carry strong hints into the exact search. The BP hard
decision enters as a solver hint (hints need not be feasible).

This is the backstop stage: shots reaching it are the residual of every
cheaper stage, so per-shot wall caps and a FEASIBLE-incumbent fallback
(flagged ``exact=False``) keep worst cases bounded without giving up
valid corrections.
"""

from typing import Optional

import numpy as np

from ._problem import DecodingProblem
from .stages import StageOutput

_WEIGHT_SCALE = 1000  # integer objective resolution for LLR weights
_PROB_CLIP = (1e-4, 0.5 - 1e-4)


def solve_mle(H: np.ndarray, weights: np.ndarray, syndrome: np.ndarray, *,
              time_limit_s: float = 10.0,
              hint: Optional[np.ndarray] = None,
              upper_bound: Optional[int] = None) -> Optional[np.ndarray]:
    """Minimum-weight solution of ``H x = syndrome (mod 2)``, or None."""
    x, _ = _solve_mle_ex(H, weights, syndrome, time_limit_s=time_limit_s,
                         hint=hint, upper_bound=upper_bound)
    return x


def _solve_mle_ex(H: np.ndarray, weights: np.ndarray, syndrome: np.ndarray,
                  *, time_limit_s: float = 10.0,
                  hint: Optional[np.ndarray] = None,
                  upper_bound: Optional[int] = None):
    """As :func:`solve_mle` but also reports whether optimality was
    proven within the time limit (the stage's ``exact`` flag)."""
    from ortools.sat.python import cp_model

    H = np.asarray(H, dtype=np.uint8)
    weights = np.asarray(weights)
    syndrome = np.asarray(syndrome).astype(np.int64)
    m, n = H.shape

    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    for r in range(m):
        support = [x[i] for i in np.flatnonzero(H[r])]
        if not support:
            if syndrome[r] % 2:
                return None, True  # provably infeasible
            continue
        u = model.new_int_var(0, len(support) // 2, f"u{r}")
        model.add(sum(support) == int(syndrome[r] % 2) + 2 * u)

    objective = sum(int(weights[i]) * x[i] for i in range(n))
    model.minimize(objective)
    if upper_bound is not None:
        model.add(objective <= int(upper_bound))
    if hint is not None:
        hint = np.asarray(hint).astype(int)
        for i in range(n):
            model.add_hint(x[i], int(hint[i]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = 4
    status = solver.solve(model)
    if status == cp_model.OPTIMAL:
        return (np.array([solver.value(xi) for xi in x], dtype=np.uint8),
                True)
    if status == cp_model.FEASIBLE:
        return (np.array([solver.value(xi) for xi in x], dtype=np.uint8),
                False)
    return None, status == cp_model.INFEASIBLE


class CpSatEndgame:
    """Exact MLE as the final pipeline stage."""

    def __init__(self, *, time_limit_s: float = 10.0):
        self.name = "endgame"
        self.time_limit_s = time_limit_s

    def _weights(self, problem: DecodingProblem,
                 posteriors: Optional[np.ndarray], shot: int) -> np.ndarray:
        if posteriors is None:
            llr = problem.llr0
        else:
            llr = posteriors[shot]
        p = 1.0 / (1.0 + np.exp(np.abs(llr)))
        p = np.clip(p, *_PROB_CLIP)
        w = np.round(_WEIGHT_SCALE * np.log((1.0 - p) / p)).astype(np.int64)
        # mechanisms BP believes flipped get the complementary weight so
        # the objective prefers including them
        flipped = llr < 0 if posteriors is not None else np.zeros_like(
            llr, dtype=bool)
        return np.where(flipped, np.maximum(1, _WEIGHT_SCALE // 100), w)

    def decode_batch(self, problem: DecodingProblem, syndromes: np.ndarray,
                     posteriors_in: Optional[np.ndarray]) -> StageOutput:
        syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
        B = syndromes.shape[0]
        n = problem.H.shape[1]
        corrections = np.zeros((B, n), dtype=np.uint8)
        resolved = np.zeros(B, dtype=bool)
        exact_flags = []
        for b in range(B):
            weights = self._weights(problem, posteriors_in, b)
            hint = ((posteriors_in[b] < 0).astype(np.uint8)
                    if posteriors_in is not None else None)
            x, exact = _solve_mle_ex(problem.H, weights, syndromes[b],
                                     time_limit_s=self.time_limit_s,
                                     hint=hint)
            if x is not None:
                corrections[b] = x
                resolved[b] = True
            exact_flags.append(bool(exact) if x is not None else False)
        return StageOutput(
            corrections=corrections, resolved=resolved, posteriors=None,
            stats={"exact": exact_flags,
                   "resolved_fraction": float(resolved.mean())})
