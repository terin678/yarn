"""Sanity referee for the numpy BP oracle itself, so later torch parity
tests inherit a trustworthy reference."""

import numpy as np
import pytest

from spyglass import build_problem
from reference_bp import reference_min_sum

pytestmark = pytest.mark.fast


def test_zero_syndrome_is_a_no_op(ring_repetition_H):
    p = build_problem(ring_repetition_H, np.full(5, 0.1))
    hard, converged, _ = reference_min_sum(p, np.zeros(5, dtype=np.uint8),
                                           max_iter=20)
    assert converged
    assert hard.sum() == 0


def test_ring_repetition_weight_one_errors(ring_repetition_H):
    p = build_problem(ring_repetition_H, np.full(5, 0.05))
    for i in range(5):
        e = np.zeros(5, dtype=np.uint8)
        e[i] = 1
        syndrome = ring_repetition_H @ e % 2
        hard, converged, _ = reference_min_sum(p, syndrome, max_iter=30)
        assert converged, i
        assert np.array_equal(ring_repetition_H @ hard % 2, syndrome)


def test_steane_weight_one_errors_reproduce_syndromes(steane_Hz):
    p = build_problem(steane_Hz, np.full(7, 0.05))
    for i in range(7):
        e = np.zeros(7, dtype=np.uint8)
        e[i] = 1
        syndrome = steane_Hz @ e % 2
        hard, converged, _ = reference_min_sum(p, syndrome, max_iter=50)
        assert converged, i
        assert np.array_equal(steane_Hz @ hard % 2, syndrome)


def test_steane_logical_accuracy_and_the_all_checks_failure(steane_Hz,
                                                           steane_Lz):
    """Flooding min-sum corrects 6 of 7 single errors logically; the bit
    sitting in ALL THREE checks converges to a syndrome-valid but
    logically WRONG correction (girth-4 overshoot). Pinned deliberately:
    this shot class is why the pipeline escalates past plain BP, and a
    later stage rescuing exactly this case is an end-to-end test target
    for the pipeline-wiring session."""
    p = build_problem(steane_Hz, np.full(7, 0.05))
    outcomes = []
    for i in range(7):
        e = np.zeros(7, dtype=np.uint8)
        e[i] = 1
        hard, converged, _ = reference_min_sum(
            p, steane_Hz @ e % 2, max_iter=50)
        outcomes.append(converged
                        and (steane_Lz @ (hard ^ e) % 2 == 0).all())
    assert outcomes.count(True) == 6
    # the failing bit is the all-checks column, and no other
    all_checks_bit = int(np.flatnonzero(steane_Hz.sum(axis=0) == 3)[0])
    assert outcomes[all_checks_bit] is np.False_ or not outcomes[all_checks_bit]
