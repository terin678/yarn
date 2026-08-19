"""Tests for the table-backed GroupData construction seam.

GroupData.from_tables / from_npz build instances from explicit
multiplication and inverse tables with no GAP involvement, and
core.native_groups constructs tables for small standard groups directly.

The referee here is a set of hand-derived tables for C2, C3, and S3,
written out from the group definitions (permutation composition applies
the left factor first, matching GAP's convention) and depending on no
implementation under test. Minted npz fixtures produced by GAP, when
present under tests/fixtures/groups/, form a third independent arm.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from core.group import (
    GroupData,
    build_group,
    canonicalize,
    dagger,
    element_order,
    is_self_dagger,
    left_rep,
    right_rep,
    ring_mul,
)
from core import native_groups

pytestmark = pytest.mark.fast

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "groups"

# ── hand-derived referee tables ────────────────────────────────────

C2_ELEMS = ["()", "(1,2)"]
C2_MULT = [[0, 1], [1, 0]]
C2_INV = [0, 1]

C3_ELEMS = ["()", "(1,2,3)", "(1,3,2)"]
C3_MULT = [[(i + j) % 3 for j in range(3)] for i in range(3)]
C3_INV = [0, 2, 1]

# S3 with elements indexed e, (12), (13), (23), (123), (132); the product
# a*b applies a first, then b.
S3_ELEMS = ["()", "(1,2)", "(1,3)", "(2,3)", "(1,2,3)", "(1,3,2)"]
S3_MULT = [
    [0, 1, 2, 3, 4, 5],
    [1, 0, 4, 5, 2, 3],
    [2, 5, 0, 4, 3, 1],
    [3, 4, 5, 0, 1, 2],
    [4, 3, 1, 2, 5, 0],
    [5, 2, 3, 1, 0, 4],
]
S3_INV = [0, 1, 2, 3, 5, 4]


def hand_s3():
    return GroupData.from_tables(S3_ELEMS, S3_MULT, S3_INV, structure="S3")


def check_axioms(gd):
    n = gd.n
    mult = gd.mult
    # closure and the Latin-square property
    for i in range(n):
        assert sorted(mult[i]) == list(range(n))
        assert sorted(mult[j][i] for j in range(n)) == list(range(n))
    # identity at index 0
    for i in range(n):
        assert mult[0][i] == i and mult[i][0] == i
    # inverses land on the identity, both sides
    for g in range(n):
        assert mult[g][gd.inv[g]] == 0
        assert mult[gd.inv[g]][g] == 0
    # associativity: exhaustive for small n, spot sample otherwise
    triples = (itertools.product(range(n), repeat=3) if n <= 12
               else itertools.islice(
                   itertools.product(range(1, n, 3), repeat=3), 200))
    for a, b, c in triples:
        assert mult[mult[a][b]][c] == mult[a][mult[b][c]]


def isomorphic(gd_a, gd_b):
    """Brute-force isomorphism test, viable for tiny groups only."""
    if gd_a.n != gd_b.n:
        return False
    n = gd_a.n
    orders_a = [element_order(g, gd_a) for g in range(n)]
    orders_b = [element_order(g, gd_b) for g in range(n)]
    if sorted(orders_a) != sorted(orders_b):
        return False
    for perm in itertools.permutations(range(1, n)):
        p = (0,) + perm
        if any(orders_a[g] != orders_b[p[g]] for g in range(n)):
            continue
        if all(p[gd_a.mult[i][j]] == gd_b.mult[p[i]][p[j]]
               for i in range(n) for j in range(n)):
            return True
    return False


# ── the import itself must be GAP-free ─────────────────────────────

class TestGapFreeImport:
    def test_core_group_imported_without_gappy(self):
        # this module imported core.group at collection time; if gappy had
        # been touched, the import would have failed in this venv, and the
        # module set must not contain it either
        assert "core.group" in sys.modules
        assert "gappy" not in sys.modules


# ── from_tables on the hand-derived referee ────────────────────────

class TestFromTablesHandDerived:
    @pytest.mark.parametrize("elems,mult,inv", [
        (C2_ELEMS, C2_MULT, C2_INV),
        (C3_ELEMS, C3_MULT, C3_INV),
        (S3_ELEMS, S3_MULT, S3_INV),
    ])
    def test_axioms(self, elems, mult, inv):
        check_axioms(GroupData.from_tables(elems, mult, inv))

    def test_attribute_contract(self):
        gd = hand_s3()
        for attr in ("n", "gap_expr", "is_abelian", "structure", "elem_strs",
                     "mult", "inv", "identity", "commutator",
                     "commutator_order", "abelianization_order", "coset_id",
                     "factors", "decompose_table", "compose_table",
                     "kron_perm"):
            assert hasattr(gd, attr), attr
        assert gd.n == 6
        assert gd.identity == 0
        assert gd.gap_expr is None
        assert gd.structure == "S3"
        repr(gd)  # must not raise

    def test_abelian_detection(self):
        assert GroupData.from_tables(C3_ELEMS, C3_MULT, C3_INV).is_abelian
        assert not hand_s3().is_abelian

    def test_commutator_subgroup_of_s3_is_a3(self):
        gd = hand_s3()
        assert gd.commutator == frozenset({0, 4, 5})
        assert gd.commutator_order == 3
        assert gd.abelianization_order == 2

    def test_coset_partition_of_s3(self):
        gd = hand_s3()
        cosets = {}
        for g, cid in enumerate(gd.coset_id):
            cosets.setdefault(cid, set()).add(g)
        assert sorted(map(sorted, cosets.values())) == [[0, 4, 5], [1, 2, 3]]

    def test_identity_must_be_index_zero(self):
        # C2 with the rows swapped puts the identity at index 1
        with pytest.raises(ValueError):
            GroupData.from_tables(["(1,2)", "()"], [[1, 0], [0, 1]], [1, 0])

    def test_inconsistent_tables_rejected(self):
        with pytest.raises(ValueError):
            GroupData.from_tables(C2_ELEMS, [[0, 1], [1, 1]], C2_INV)
        with pytest.raises(ValueError):
            GroupData.from_tables(C2_ELEMS, C2_MULT, [0, 0])

    def test_decompose_raises_without_tables(self):
        gd = hand_s3()
        with pytest.raises(Exception):
            gd.decompose(1)


# ── ring / representation functions on table-backed instances ──────

class TestContractFunctions:
    def test_ring_mul_matches_hand_products(self):
        gd = hand_s3()
        # single supports reduce to the multiplication table
        for i in range(6):
            for j in range(6):
                assert ring_mul((i,), (j,), gd) == (S3_MULT[i][j],)
        # {(12), (13)} * {(123)} = {(12)(123), (13)(123)} = {(13), (23)}
        assert ring_mul((1, 2), (4,), gd) == (2, 3)

    def test_ring_mul_f2_cancellation(self):
        gd = hand_s3()
        # {e, (12)} * {e, (12)} = {e, (12), (12), e} = empty over F2
        assert ring_mul((0, 1), (0, 1), gd) == ()

    def test_dagger_uses_inverse_table(self):
        gd = hand_s3()
        assert dagger((1, 4), gd) == canonicalize((1, 5))
        assert is_self_dagger((1, 2, 3), gd)
        assert not is_self_dagger((4,), gd)

    def test_element_orders(self):
        gd = hand_s3()
        assert [element_order(g, gd) for g in range(6)] == [1, 2, 2, 2, 3, 3]

    def test_left_rep_is_a_homomorphism(self):
        gd = hand_s3()
        for g, h in ((1, 4), (2, 3), (4, 5)):
            lhs = (left_rep((g,), gd) @ left_rep((h,), gd)) % 2
            rhs = left_rep((S3_MULT[g][h],), gd)
            assert np.array_equal(lhs, rhs)

    def test_left_and_right_reps_commute(self):
        gd = hand_s3()
        L = left_rep((1,), gd)
        R = right_rep((4,), gd)
        assert np.array_equal((L @ R) % 2, (R @ L) % 2)


# ── native constructors ────────────────────────────────────────────

class TestNativeConstructors:
    def test_cyclic(self):
        gd = native_groups.cyclic(10)
        check_axioms(gd)
        assert gd.n == 10 and gd.is_abelian
        assert gd.commutator_order == 1
        assert sorted(element_order(g, gd) for g in range(10)) == \
            sorted((10 // np.gcd(g, 10) if g else 1) for g in range(10))

    def test_dihedral_order_8(self):
        gd = native_groups.dihedral(8)
        check_axioms(gd)
        assert gd.n == 8 and not gd.is_abelian
        assert gd.commutator_order == 2
        assert gd.abelianization_order == 4

    def test_symmetric_3_isomorphic_to_hand_table(self):
        gd = native_groups.symmetric(3)
        check_axioms(gd)
        assert isomorphic(gd, hand_s3())

    def test_direct_product_kron_tables(self):
        gd = native_groups.direct_product(native_groups.cyclic(2),
                                          native_groups.cyclic(3))
        check_axioms(gd)
        assert gd.n == 6 and gd.is_abelian
        assert gd.decompose_table is not None
        for g in range(6):
            assert gd.compose(gd.decompose(g)) == g
        # native ordering IS Kronecker order
        assert gd.kron_perm == list(range(6))


# ── npz round-trip and schema ──────────────────────────────────────

class TestNpz:
    def test_roundtrip(self, tmp_path):
        gd = hand_s3()
        path = tmp_path / "s3.npz"
        native_groups.to_npz(gd, path, provenance={
            "gap_expr": "SymmetricGroup(3)", "structure": "S3",
            "gap_version": "hand", "minted": "test"})
        gd2 = GroupData.from_npz(path)
        assert gd2.elem_strs == gd.elem_strs
        assert gd2.mult == gd.mult
        assert gd2.inv == gd.inv
        assert gd2.structure == "S3"
        assert gd2.gap_expr == "SymmetricGroup(3)"
        check_axioms(gd2)

    def test_missing_field_rejected(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, elem_strs=np.array(C2_ELEMS), mult=np.array(C2_MULT))
        with pytest.raises((ValueError, KeyError)):
            GroupData.from_npz(path)


# ── build_group config seam ────────────────────────────────────────

class _Cfg:
    def __init__(self, gap_expr=None, native=None, table=None):
        self.gap_expr = gap_expr
        self.native = native
        self.table = table


class TestBuildGroup:
    def test_native_spec(self):
        gd = build_group(_Cfg(native="S3"))
        check_axioms(gd)
        assert gd.n == 6
        assert "gappy" not in sys.modules

    def test_native_product_spec(self):
        gd = build_group(_Cfg(native="C2xC3"))
        assert gd.n == 6 and gd.is_abelian
        assert gd.decompose_table is not None

    def test_table_path(self, tmp_path):
        path = tmp_path / "c3.npz"
        native_groups.to_npz(GroupData.from_tables(C3_ELEMS, C3_MULT, C3_INV),
                             path, provenance={"structure": "C3"})
        gd = build_group(_Cfg(table=str(path)))
        assert gd.n == 3 and gd.is_abelian

    def test_gap_expr_path_still_routes_to_gap(self):
        # in this venv gappy is absent, so the legacy path must fail with
        # the import error rather than being silently rerouted
        with pytest.raises(ImportError):
            build_group(_Cfg(gap_expr="SymmetricGroup(3)"))

    def test_exactly_one_source_required(self):
        with pytest.raises(ValueError):
            build_group(_Cfg())
        with pytest.raises(ValueError):
            build_group(_Cfg(native="S3", table="x.npz"))


# ── loader accepts the new group keys ──────────────────────────────

class TestLoaderGroupKeys:
    def test_native_key_parses(self):
        from search.configs.loader import _group_from_dict
        g = _group_from_dict({"native": "C10"})
        assert g.native == "C10" and g.gap_expr is None

    def test_unknown_key_still_rejected(self):
        from search.configs.loader import _group_from_dict
        with pytest.raises(ValueError):
            _group_from_dict({"native": "C10", "bogus": 1})

    def test_gap_expr_key_unchanged(self):
        from search.configs.loader import _group_from_dict
        g = _group_from_dict({"gap_expr": "CyclicGroup(4)", "tag": "c4"})
        assert g.gap_expr == "CyclicGroup(4)" and g.tag == "c4"


# ── three-arm comparison against minted GAP tables, when present ────

MINTED = sorted(FIXTURE_DIR.glob("*.npz")) if FIXTURE_DIR.exists() else []


@pytest.mark.skipif(not MINTED, reason="no minted group fixtures present")
class TestMintedFixtures:
    def minted(self, stem):
        path = FIXTURE_DIR / f"{stem}.npz"
        if not path.exists():
            pytest.skip(f"{stem} not minted")
        return GroupData.from_npz(path)

    def test_minted_tables_satisfy_axioms(self):
        for path in MINTED:
            check_axioms(GroupData.from_npz(path))

    def test_minted_s3_isomorphic_to_hand_and_native(self):
        gd = self.minted("s3")
        assert isomorphic(gd, hand_s3())
        assert isomorphic(gd, native_groups.symmetric(3))

    def test_minted_c10_matches_native_invariants(self):
        gd = self.minted("c10")
        nat = native_groups.cyclic(10)
        assert gd.n == nat.n and gd.is_abelian
        assert sorted(element_order(g, gd) for g in range(gd.n)) == \
            sorted(element_order(g, nat) for g in range(nat.n))

    def test_minted_d8_matches_native_invariants(self):
        gd = self.minted("d8")
        nat = native_groups.dihedral(8)
        assert gd.n == nat.n and not gd.is_abelian
        assert gd.commutator_order == nat.commutator_order
        assert sorted(element_order(g, gd) for g in range(gd.n)) == \
            sorted(element_order(g, nat) for g in range(nat.n))
