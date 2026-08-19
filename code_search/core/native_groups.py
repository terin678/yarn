"""Native constructors for small standard groups, no GAP required.

Each constructor builds explicit element lists and multiplication/inverse
tables and hands them to ``GroupData.from_tables``. Element index 0 is
always the identity. Conventions:

- ``cyclic(n)``: elements r^0..r^(n-1) in exponent order.
- ``dihedral(order)``: order must be even, 2m; elements r^0..r^(m-1) then
  s·r^0..s·r^(m-1), with r^m = s^2 = e and s·r·s = r^(-1). The product
  convention matches the package's mult table: mult[i][j] = g_i * g_j.
- ``symmetric(n)``: all permutations of {1..n} in lexicographic one-line
  order (identity first); products apply the LEFT factor first, matching
  GAP's permutation composition.
- ``direct_product(*factors)``: Kronecker (row-major) element order over
  the factor indices, so ``kron_perm`` is the identity permutation, with
  ``decompose_table``/``compose_table`` populated.
- ``build_native(spec)``: resolves spec strings ``"Cn"``, ``"Dn"``
  (n = order, even), ``"Sn"``, and ``"x"``-joined products of those,
  e.g. ``"C2xC3"``.

npz minting/loading lives here too (``to_npz``; ``GroupData.from_npz``
reads them) so GAP-minted tables and native tables share one format.
"""

import itertools
import json
import re

import numpy as np

from .group import GroupData


def _from_elements(elems, compose, invert, label, elem_str):
    """Build GroupData from an element list, a compose fn, and an invert fn.

    ``elems`` must put the identity first; index lookups are by equality.
    """
    index = {e: i for i, e in enumerate(elems)}
    n = len(elems)
    mult = [[index[compose(elems[i], elems[j])] for j in range(n)]
            for i in range(n)]
    inv = [index[invert(elems[i])] for i in range(n)]
    return GroupData.from_tables([elem_str(e) for e in elems], mult, inv,
                                 structure=label)


def cyclic(n: int) -> GroupData:
    if n < 1:
        raise ValueError("cyclic order must be positive")
    return _from_elements(
        list(range(n)),
        compose=lambda a, b: (a + b) % n,
        invert=lambda a: (-a) % n,
        label=f"C{n}",
        elem_str=lambda a: "()" if a == 0 else f"r^{a}")


def dihedral(order: int) -> GroupData:
    """Dihedral group of the given ORDER (must be even), matching GAP's
    DihedralGroup(order) sizing convention."""
    if order < 2 or order % 2:
        raise ValueError("dihedral order must be even and >= 2")
    m = order // 2
    # element (f, k) = s^f · r^k;  (f1,k1)·(f2,k2) applies the group law
    # with s·r^k = r^(-k)·s:  s^f1 r^k1 · s^f2 r^k2 = s^(f1+f2) r^(k2 ± k1)
    elems = [(f, k) for f in (0, 1) for k in range(m)]

    def compose(a, b):
        f1, k1 = a
        f2, k2 = b
        return ((f1 + f2) % 2, (k2 + (k1 if not f2 else -k1)) % m)

    def invert(a):
        f, k = a
        return (f, k % m) if f else (0, (-k) % m)

    return _from_elements(
        elems, compose, invert, label=f"D{order}",
        elem_str=lambda e: ("()" if e == (0, 0)
                            else ("s" if e[1] == 0 else f"s*r^{e[1]}") if e[0]
                            else f"r^{e[1]}"))


def symmetric(n: int) -> GroupData:
    if n < 1:
        raise ValueError("symmetric degree must be positive")
    elems = sorted(itertools.permutations(range(n)))

    def compose(a, b):
        # apply a first, then b (GAP's convention)
        return tuple(b[a[i]] for i in range(n))

    def invert(a):
        out = [0] * n
        for i, v in enumerate(a):
            out[v] = i
        return tuple(out)

    def elem_str(p):
        # cycle notation, 1-based, identity as "()"
        seen, cycles = set(), []
        for start in range(n):
            if start in seen or p[start] == start:
                seen.add(start)
                continue
            cyc, x = [], start
            while x not in seen:
                seen.add(x)
                cyc.append(x + 1)
                x = p[x]
            cycles.append("(" + ",".join(map(str, cyc)) + ")")
        return "".join(cycles) if cycles else "()"

    return _from_elements(elems, compose, invert, label=f"S{n}",
                          elem_str=elem_str)


def direct_product(*factors: GroupData) -> GroupData:
    """Direct product in Kronecker (row-major) element order, so kron_perm
    is the identity and decompose/compose tables come for free."""
    if len(factors) < 2:
        raise ValueError("direct_product needs at least two factors")
    sizes = [f.n for f in factors]
    tuples = list(itertools.product(*[range(s) for s in sizes]))
    index = {t: i for i, t in enumerate(tuples)}
    n = len(tuples)
    mult = [[index[tuple(f.mult[a[x]][b[x]] for x, f in enumerate(factors))]
             for b in tuples] for a in tuples]
    inv = [index[tuple(f.inv[a[x]] for x, f in enumerate(factors))]
           for a in tuples]
    elem_strs = ["[" + ",".join(f.elem_strs[a[x]]
                                for x, f in enumerate(factors)) + "]"
                 for a in tuples]
    structure = " x ".join(f.structure or "?" for f in factors)
    return GroupData.from_tables(
        elem_strs, mult, inv, structure=structure, factors=list(factors),
        decompose_table=list(tuples),
        compose_table={t: i for i, t in enumerate(tuples)},
        kron_perm=list(range(n)))


_SPEC_RE = re.compile(r"^([CDS])(\d+)$")


def build_native(spec: str) -> GroupData:
    """Resolve a native spec string: Cn, Dn (n = order), Sn, or an
    "x"-joined product such as "C2xC3"."""
    parts = spec.replace(" ", "").split("x")
    built = []
    for part in parts:
        m = _SPEC_RE.match(part)
        if not m:
            raise ValueError(
                f"unrecognized native group spec {part!r} in {spec!r}; "
                f"expected Cn, Dn, or Sn")
        kind, num = m.group(1), int(m.group(2))
        built.append({"C": cyclic, "D": dihedral, "S": symmetric}[kind](num))
    return built[0] if len(built) == 1 else direct_product(*built)


def to_npz(gd: GroupData, path, provenance: dict | None = None) -> None:
    """Mint a GroupData's tables to npz (elem_strs, mult, inv, provenance)."""
    prov = dict(provenance or {})
    prov.setdefault("gap_expr", gd.gap_expr)
    prov.setdefault("structure", gd.structure)
    np.savez(path,
             elem_strs=np.array(gd.elem_strs, dtype="U"),
             mult=np.array(gd.mult, dtype=np.int64),
             inv=np.array(gd.inv, dtype=np.int64),
             provenance=np.array(json.dumps(prov)))
