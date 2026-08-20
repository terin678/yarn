"""Code-capacity noise helpers.

Uniform independent bit flips are the only noise model this module knows;
it exists so examples and tests have a one-line way to make problems and
shots. Composing the two CSS directions (decode X errors against Hz,
Z errors against Hx) is the caller's job by design: the decoder itself
never sees a CSS pair, which is what keeps its interface ready for a
detector error model later (a DEM is just a different ``(H, priors)``).
"""

import numpy as np

from ._problem import DecodingProblem, build_problem


def code_capacity_problem(H: np.ndarray, p: float) -> DecodingProblem:
    """A :class:`DecodingProblem` with uniform flip probability ``p``."""
    H = np.asarray(H)
    return build_problem(H, np.full(H.shape[1], float(p)))


def sample_code_capacity(H: np.ndarray, p: float, shots: int,
                         rng: np.random.Generator):
    """Sample iid bit-flip errors and their syndromes.

    Returns:
        ``(errors, syndromes)`` with shapes ``(shots, n)`` and
        ``(shots, m)``, both uint8.
    """
    H = np.asarray(H, dtype=np.uint8)
    errors = (rng.random((shots, H.shape[1])) < p).astype(np.uint8)
    syndromes = errors @ H.T % 2
    return errors, syndromes.astype(np.uint8)
