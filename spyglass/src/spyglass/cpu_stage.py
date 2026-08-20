"""Serial-schedule BP stage over the ldpc package.

The serial (layered) message schedule breaks the symmetric trapping sets
that defeat flooding BP on short-cycle quantum codes, which is this
stage's whole reason to sit between the batched GPU stages and the exact
endgame. One decoder is built per problem and reused across the (small)
residual it sees; every reported success is re-verified against the
syndrome with plain numpy rather than trusting the library's converged
flag.
"""

from typing import Optional

import numpy as np

from ._problem import DecodingProblem
from .stages import StageOutput


class SerialBpStage:
    """ldpc.BpDecoder with a serial schedule, as a pipeline stage."""

    def __init__(self, *, max_iter: int = 200,
                 bp_method: str = "product_sum"):
        # product-sum by default: the serial minimum-sum combination in
        # the backing library failed to converge on trivial single-error
        # syndromes in testing, and the schedule, not the update rule, is
        # what this stage is here for
        self.name = "serial_bp"
        self.max_iter = max_iter
        self.bp_method = bp_method
        self._decoder = None
        self._decoder_key = None

    def _get_decoder(self, problem: DecodingProblem):
        from ldpc import BpDecoder
        key = id(problem)
        if self._decoder is None or self._decoder_key != key:
            self._decoder = BpDecoder(
                problem.H,
                error_channel=problem.priors.tolist(),
                max_iter=self.max_iter,
                bp_method=self.bp_method,
                schedule="serial")
            self._decoder_key = key
        return self._decoder

    def decode_batch(self, problem: DecodingProblem, syndromes: np.ndarray,
                     posteriors_in: Optional[np.ndarray]) -> StageOutput:
        syndromes = np.atleast_2d(np.asarray(syndromes, dtype=np.uint8))
        decoder = self._get_decoder(problem)
        B = syndromes.shape[0]
        n = problem.H.shape[1]
        corrections = np.zeros((B, n), dtype=np.uint8)
        resolved = np.zeros(B, dtype=bool)
        for b in range(B):
            corr = np.asarray(decoder.decode(syndromes[b]),
                              dtype=np.uint8)
            corrections[b] = corr
            # defensive: the syndrome check is ours, not the library's
            resolved[b] = bool(
                np.array_equal(problem.H @ corr % 2, syndromes[b]))
        return StageOutput(
            corrections=corrections, resolved=resolved, posteriors=None,
            stats={"resolved_fraction": float(resolved.mean())})
