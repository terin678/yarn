"""Exact MLE endgame vs brute force, plus the payoff test: the endgame
rescues the shot class that defeats min-sum."""

import itertools

import numpy as np
import pytest

pytest.importorskip("ortools")

from spyglass import build_problem
from spyglass.endgame import CpSatEndgame, solve_mle

pytestmark = pytest.mark.fast


def _brute_force_min_objective(H, weights, syndrome):
    n = H.shape[1]
    best = None
    for bits in itertools.product((0, 1), repeat=n):
        x = np.array(bits, dtype=np.uint8)
        if np.array_equal(H @ x % 2, syndrome):
            obj = float(weights @ x)
            best = obj if best is None else min(best, obj)
    return best


def test_matches_brute_force_on_random_instances():
    rng = np.random.default_rng(42)
    checked = 0
    while checked < 20:
        m = int(rng.integers(2, 8))
        n = int(rng.integers(m, 12))
        H = (rng.random((m, n)) < 0.4).astype(np.uint8)
        if H.sum() == 0:
            continue
        weights = rng.integers(1, 500, size=n).astype(np.int64)
        true_e = (rng.random(n) < 0.3).astype(np.uint8)
        syndrome = H @ true_e % 2
        x = solve_mle(H, weights, syndrome, time_limit_s=10)
        assert x is not None
        assert np.array_equal(H @ x % 2, syndrome)
        assert float(weights @ x) == _brute_force_min_objective(
            H, weights, syndrome)
        checked += 1


def test_infeasible_syndrome_returns_none():
    H = np.array([[1, 1, 0],
                  [0, 0, 0]], dtype=np.uint8)  # dead check row
    syndrome = np.array([0, 1], dtype=np.uint8)
    assert solve_mle(H, np.ones(3, dtype=np.int64), syndrome,
                     time_limit_s=5) is None


def test_posterior_weighting_breaks_ties():
    # two single-bit solutions to the same syndrome; the cheaper weight
    # must win each way around
    H = np.array([[1, 1]], dtype=np.uint8)
    syndrome = np.array([1], dtype=np.uint8)
    x = solve_mle(H, np.array([10, 500]), syndrome, time_limit_s=5)
    assert x.tolist() == [1, 0]
    x = solve_mle(H, np.array([500, 10]), syndrome, time_limit_s=5)
    assert x.tolist() == [0, 1]


def test_endgame_stage_rescues_the_all_checks_shot(steane_Hz, steane_Lz):
    """The S8a-pinned failure: min-sum miscorrects the all-checks bit into
    the logical coset. Exact MLE at uniform weights finds the weight-1
    error, closing the escalation narrative."""
    problem = build_problem(steane_Hz, np.full(7, 0.05))
    all_checks_bit = int(np.flatnonzero(steane_Hz.sum(axis=0) == 3)[0])
    e = np.zeros(7, dtype=np.uint8)
    e[all_checks_bit] = 1
    syndrome = (steane_Hz @ e % 2)[None]

    stage = CpSatEndgame(time_limit_s=10)
    out = stage.decode_batch(problem, syndrome, None)
    assert out.resolved[0]
    assert np.array_equal(out.corrections[0], e)
    assert (steane_Lz @ (out.corrections[0] ^ e) % 2 == 0).all()


def test_stage_reports_exact_flag_and_objectives(steane_Hz):
    problem = build_problem(steane_Hz, np.full(7, 0.05))
    syndromes = np.eye(3, dtype=np.uint8)  # three weight-1-decodable shots
    out = CpSatEndgame(time_limit_s=10).decode_batch(problem, syndromes,
                                                     None)
    assert out.resolved.all()
    assert out.stats["exact"] == [True, True, True]
