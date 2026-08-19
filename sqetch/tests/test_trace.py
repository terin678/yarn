"""Per-batch trace observability on estimate_distance -- skipped without CUDA.

The trace records (trials_done_after_batch, batch_best, running_best) per
completed batch, host-side only. It exists so callers can study convergence
without re-paying the null-space and packing cost of repeated small runs.
Traces are a function of (seed, batch_size) jointly: batch seeds derive from
seed + trials_done, so a different batch_size is a different trial sequence.
"""

import math

import numpy as np
import pytest


def _have_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


pytestmark = pytest.mark.skipif(not _have_cuda(), reason="CUDA / PyTorch not available")


def _steane_inputs():
    H = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    L = np.ones((1, 7), dtype=np.uint8)
    return H, L


def _run(record_trace, num_trials=2_000, batch_size=500, d_target=None, seed=11):
    from sqetch import estimate_distance

    H, L = _steane_inputs()
    return estimate_distance(
        H, L,
        num_trials=num_trials,
        batch_size=batch_size,
        d_target=d_target,
        k_sub=1,
        seed=seed,
        record_trace=record_trace,
    )


def test_trace_none_by_default():
    from sqetch import estimate_distance

    H, L = _steane_inputs()
    result = estimate_distance(H, L, num_trials=500, k_sub=1, seed=3)
    assert result.trace is None


def test_trace_batch_count_matches_ceil_trials_over_batch():
    result = _run(record_trace=True, num_trials=2_000, batch_size=500)
    assert result.trace is not None
    assert len(result.trace) == math.ceil(2_000 / 500)


def test_trace_running_best_monotone_and_matches_best_weight():
    result = _run(record_trace=True)
    running = [entry[2] for entry in result.trace]
    assert all(a >= b for a, b in zip(running, running[1:]))
    assert running[-1] == result.best_weight


def test_trace_last_entry_trials_equals_trials_run():
    result = _run(record_trace=True)
    assert result.trace[-1][0] == result.trials_run


def test_trace_batch_best_never_below_running_best_history():
    result = _run(record_trace=True)
    running = None
    for trials, batch_best, running_best in result.trace:
        expected = batch_best if running is None else min(running, batch_best)
        assert running_best == expected
        running = running_best


def test_trace_deterministic_same_seed_same_batch_size():
    a = _run(record_trace=True)
    b = _run(record_trace=True)
    assert a.trace == b.trace


def test_trace_truncates_at_early_stop():
    # d_target above the Steane distance (3) fires the early stop on the
    # first batch that sees any logical below it; the trace must end at the
    # batch where the loop broke, not run the full budget.
    result = _run(record_trace=True, num_trials=100_000, batch_size=500, d_target=7)
    assert result.found
    assert result.trace[-1][0] == result.trials_run
    assert result.trials_run < 100_000


def test_trace_differs_across_batch_size():
    # Same seed, different batch_size: a different base-seed sequence, so
    # the trial universe differs. This documents the seed semantics rather
    # than asserting any particular values.
    a = _run(record_trace=True, num_trials=2_000, batch_size=500)
    b = _run(record_trace=True, num_trials=2_000, batch_size=250)
    assert [e[0] for e in a.trace] != [e[0] for e in b.trace]
