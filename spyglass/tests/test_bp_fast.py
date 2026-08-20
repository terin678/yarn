"""Torch BP vs the numpy oracle, on CPU: message parity, outcome parity
(including the pinned min-sum failure, which torch must REPRODUCE), and
the memory term's effect."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyglass import build_problem
from spyglass.bp import BatchBpStage, bp_decode_batch
from reference_bp import reference_min_sum

pytestmark = pytest.mark.fast


def _oracle_messages_to_array(problem, messages):
    """Map the oracle's {(\"c2v\", c, v): value} dict onto the padded
    (m, max_cd) layout."""
    m, max_cd = problem.chk_nbrs.shape
    out = np.zeros((m, max_cd))
    for c in range(m):
        for j in range(problem.chk_deg[c]):
            out[c, j] = messages[("c2v", c, problem.chk_nbrs[c, j])]
    return out


@pytest.mark.parametrize("fixture_name", ["ring_repetition_H", "steane_Hz"])
def test_message_parity_with_oracle_at_five_iters(request, fixture_name):
    H = request.getfixturevalue(fixture_name)
    n = H.shape[1]
    problem = build_problem(H, np.full(n, 0.08))
    # a deliberately hard syndrome (all ones) so five iterations do not
    # converge and messages are compared mid-flight
    syndrome = np.ones(H.shape[0], dtype=np.uint8)

    _, _, oracle_msgs = reference_min_sum(problem, syndrome, max_iter=5,
                                          gamma=0.0)
    oracle_c2v = _oracle_messages_to_array(problem, oracle_msgs)

    result = bp_decode_batch(problem, syndrome[None], max_iter=5, gamma=0.0,
                             device="cpu", return_state=True)
    torch_c2v = result.state_c2v[0]
    assert np.allclose(torch_c2v, oracle_c2v, atol=1e-5), fixture_name


def test_outcome_parity_on_all_steane_single_errors(steane_Hz):
    problem = build_problem(steane_Hz, np.full(7, 0.05))
    for i in range(7):
        e = np.zeros(7, dtype=np.uint8)
        e[i] = 1
        syndrome = steane_Hz @ e % 2
        o_hard, o_conv, _ = reference_min_sum(problem, syndrome, max_iter=50)
        r = bp_decode_batch(problem, syndrome[None], max_iter=50,
                            device="cpu")
        assert bool(r.converged[0]) == o_conv, i
        # parity means reproducing the oracle EXACTLY, including the
        # pinned all-checks miscorrection
        assert np.array_equal(r.hard[0], o_hard), i


def test_zero_syndrome_converges_immediately(steane_Hz):
    problem = build_problem(steane_Hz, np.full(7, 0.05))
    r = bp_decode_batch(problem, np.zeros((1, 3), dtype=np.uint8),
                        max_iter=50, device="cpu")
    assert r.converged[0] and r.hard[0].sum() == 0
    assert r.iterations[0] <= 1


def test_batched_shots_are_independent(steane_Hz):
    problem = build_problem(steane_Hz, np.full(7, 0.05))
    syndromes = []
    for i in range(7):
        e = np.zeros(7, dtype=np.uint8)
        e[i] = 1
        syndromes.append(steane_Hz @ e % 2)
    batch = np.stack(syndromes)
    r_batch = bp_decode_batch(problem, batch, max_iter=50, device="cpu")
    for i in range(7):
        r_one = bp_decode_batch(problem, batch[i][None], max_iter=50,
                                device="cpu")
        assert np.array_equal(r_batch.hard[i], r_one.hard[0]), i
        assert r_batch.converged[i] == r_one.converged[0], i


def test_memory_term_changes_trajectories(ring_repetition_H):
    # the all-ones syndrome on the ring code is INFEASIBLE (every real
    # syndrome has even weight), so BP can never converge and freeze;
    # memory's influence starts at iteration 3, so 8 iterations separate
    # the trajectories cleanly
    problem = build_problem(ring_repetition_H, np.full(5, 0.05))
    syndrome = np.ones((1, 5), dtype=np.uint8)
    r0 = bp_decode_batch(problem, syndrome, max_iter=8, gamma=0.0,
                         device="cpu", return_state=True)
    r5 = bp_decode_batch(problem, syndrome, max_iter=8, gamma=0.5,
                         device="cpu", return_state=True)
    assert not r0.converged[0] and not r5.converged[0]
    assert not np.allclose(r0.state_c2v, r5.state_c2v)


def test_posteriors_shape_and_finite(steane_Hz):
    problem = build_problem(steane_Hz, np.full(7, 0.05))
    r = bp_decode_batch(problem, np.ones((4, 3), dtype=np.uint8),
                        max_iter=10, device="cpu")
    assert r.posteriors.shape == (4, 7)
    assert np.isfinite(r.posteriors).all()


def test_stage_adapter_resolves_shots(ring_repetition_H):
    from spyglass import TelescopingDecoder
    from spyglass.noise import sample_code_capacity
    problem_H = ring_repetition_H
    _, syndromes = sample_code_capacity(problem_H, 0.05, 100,
                                        np.random.default_rng(5))
    dec = TelescopingDecoder(problem_H, np.full(5, 0.05),
                             stages=[BatchBpStage(max_iter=50,
                                                  device="cpu")])
    result = dec.decode_batch(syndromes)
    assert result.converged.mean() > 0.5
    assert (result.stage[result.converged] == 0).all()
