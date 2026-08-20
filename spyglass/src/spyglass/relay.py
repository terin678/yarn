"""Relay-BP ensemble stage: legs of memory-BP with randomized strengths.

Each leg reruns batched min-sum with a fresh draw of per-shot,
per-variable memory strengths from ``gamma_range``, message state reset,
and the TOTAL-LLR memory carried over from the previous leg (that carry
is the relay). A shot is accepted at the first leg whose hard decision
reproduces its syndrome. Negative strengths are deliberate: anti-memory
kicks oscillating shots out of the traps that defeated the plain stage.

The ensemble idea follows the Relay-BP lineage the telescoping
architecture cites; every constant here (leg count, iteration split, the
strength range) is this implementation's own.
"""

from typing import Optional

import numpy as np

from ._problem import DecodingProblem
from .bp import bp_decode_batch
from .stages import StageOutput


class RelayBpStage:
    """Ensemble of randomized-memory BP legs over a shrinking residual."""

    def __init__(self, *, num_legs: int = 12, leg_max_iter: int = 60,
                 gamma_range: tuple = (-0.25, 0.65),
                 seed: Optional[int] = None, ms_scale: float = 0.625,
                 device: str = "cpu"):
        self.name = "relay"
        self.num_legs = num_legs
        self.leg_max_iter = leg_max_iter
        self.gamma_range = gamma_range
        self.seed = 0 if seed is None else int(seed)
        self.ms_scale = ms_scale
        self.device = device

    def _draw_gamma(self, batch: int, n: int, leg: int):
        """Per-(shot, variable) strengths, drawn on CPU for cross-device
        reproducibility, seeded by (seed, leg)."""
        import torch
        g = torch.Generator()
        g.manual_seed(self.seed * 1_000_003 + leg)
        lo, hi = self.gamma_range
        return lo + (hi - lo) * torch.rand((batch, n), generator=g)

    def decode_batch(self, problem: DecodingProblem, syndromes: np.ndarray,
                     posteriors_in: Optional[np.ndarray]) -> StageOutput:
        syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
        B = syndromes.shape[0]
        n = problem.H.shape[1]

        corrections = np.zeros((B, n), dtype=np.uint8)
        posteriors = np.tile(problem.llr0, (B, 1))
        resolved = np.zeros(B, dtype=bool)
        leg_of = np.full(B, -1, dtype=np.int32)

        residual = np.arange(B)
        llr_prev = (np.asarray(posteriors_in, dtype=np.float64)
                    if posteriors_in is not None else None)
        legs_run = 0

        for leg in range(self.num_legs):
            if residual.size == 0:
                break
            legs_run = leg + 1
            gamma = self._draw_gamma(residual.size, n, leg).numpy()
            r = bp_decode_batch(
                problem, syndromes[residual], max_iter=self.leg_max_iter,
                ms_scale=self.ms_scale, gamma=gamma,
                llr_prev_init=llr_prev, device=self.device,
                return_state=True)

            hit = r.converged
            accepted = residual[hit]
            corrections[accepted] = r.hard[hit]
            posteriors[accepted] = r.posteriors[hit]
            resolved[accepted] = True
            leg_of[accepted] = leg

            # unresolved shots keep this leg's best effort and carry
            # their memory into the next leg
            missed = residual[~hit]
            corrections[missed] = r.hard[~hit]
            posteriors[missed] = r.posteriors[~hit]
            llr_prev = r.state_llr_prev[~hit]
            residual = missed

        return StageOutput(
            corrections=corrections, resolved=resolved,
            posteriors=posteriors,
            stats={"legs_used": legs_run,
                   "leg_of_resolution": leg_of.tolist(),
                   "resolved_fraction": float(resolved.mean())})
