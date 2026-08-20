"""DecodingProblem construction: validation, padded neighbor tables, and
the edge-position inverse maps everything downstream leans on."""

import numpy as np
import pytest

from spyglass import build_problem

pytestmark = pytest.mark.fast

H_SMALL = np.array([[1, 1, 0],
                    [0, 1, 1]], dtype=np.uint8)


def test_shapes_and_llr():
    p = build_problem(H_SMALL, np.full(3, 0.1))
    assert p.H.shape == (2, 3)
    assert np.allclose(p.llr0, np.log(0.9 / 0.1))
    assert p.chk_deg.tolist() == [2, 2]
    assert p.var_deg.tolist() == [1, 2, 1]


def test_padded_neighbor_tables_and_sentinels():
    p = build_problem(H_SMALL, np.full(3, 0.1))
    # chk_nbrs rows list the variable indices of each check, sentinel n=3
    assert p.chk_nbrs.tolist() == [[0, 1], [1, 2]]
    # var_nbrs rows list the check indices of each variable, sentinel m=2
    assert p.var_nbrs.shape == (3, 2)
    assert p.var_nbrs[0].tolist() == [0, 2]   # var 0: check 0, then padding
    assert p.var_nbrs[1].tolist() == [0, 1]
    assert p.var_nbrs[2].tolist() == [1, 2]


def test_edge_position_inverse_maps_roundtrip():
    rng = np.random.default_rng(11)
    H = (rng.random((6, 10)) < 0.3).astype(np.uint8)
    H[:, 0] |= 1  # ensure no empty column 0 edge cases
    p = build_problem(H, np.full(10, 0.05))
    m, n = H.shape
    # for every real edge seen from the check side, the inverse map must
    # point back to the same edge seen from the variable side
    for c in range(m):
        for j in range(p.chk_deg[c]):
            v = p.chk_nbrs[c, j]
            slot = p.chk_edge_pos[c, j]
            assert p.var_nbrs[v, slot] == c
    for v in range(n):
        for j in range(p.var_deg[v]):
            c = p.var_nbrs[v, j]
            slot = p.var_edge_pos[v, j]
            assert p.chk_nbrs[c, slot] == v


def test_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        build_problem(H_SMALL, np.full(4, 0.1))       # length mismatch
    with pytest.raises(ValueError):
        build_problem(H_SMALL, np.full(3, 0.7))       # prior >= 0.5
    with pytest.raises(ValueError):
        build_problem(H_SMALL, np.full(3, 0.0))       # prior <= 0
    with pytest.raises(ValueError):
        build_problem(np.array([[0, 2, 0]]), np.full(3, 0.1))  # non-binary


def test_frozen():
    p = build_problem(H_SMALL, np.full(3, 0.1))
    with pytest.raises(Exception):
        p.H = None
