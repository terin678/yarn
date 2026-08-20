"""Serial-schedule BP stage over the ldpc package."""

import numpy as np
import pytest

pytest.importorskip("ldpc")

from spyglass import build_problem
from spyglass.cpu_stage import SerialBpStage

pytestmark = pytest.mark.ldpc


def _steane():
    return np.array([[0, 0, 0, 1, 1, 1, 1],
                     [0, 1, 1, 0, 0, 1, 1],
                     [1, 0, 1, 0, 1, 0, 1]], dtype=np.uint8)


def test_serial_resolves_all_steane_singles():
    H = _steane()
    problem = build_problem(H, np.full(7, 0.05))
    syndromes = np.stack([H @ np.eye(7, dtype=np.uint8)[i] % 2
                          for i in range(7)])
    out = SerialBpStage(max_iter=100).decode_batch(problem, syndromes, None)
    assert out.resolved.all()
    assert np.array_equal(out.corrections @ H.T % 2, syndromes)


def test_defensive_recheck_catches_forced_nonconvergence():
    H = _steane()
    problem = build_problem(H, np.full(7, 0.05))
    # all-ones is a hard syndrome; one iteration cannot resolve it, and
    # the stage must say so rather than trust the library
    syndromes = np.ones((1, 3), dtype=np.uint8)
    out = SerialBpStage(max_iter=1).decode_batch(problem, syndromes, None)
    if out.resolved[0]:  # if it somehow did, the correction must be valid
        assert np.array_equal(out.corrections[0] @ H.T % 2, syndromes[0])
    else:
        assert not np.array_equal(out.corrections[0] @ H.T % 2,
                                  syndromes[0])
