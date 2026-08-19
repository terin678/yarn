"""Mint npz group tables from GAP, one time, wherever a gap binary exists.

Drives the ``gap`` executable directly (no gappy/libgap needed), so it runs
on any machine with ``apt-get install gap`` even when the package itself
cannot. The minted npz files feed ``GroupData.from_npz`` and the fixture
tests under tests/fixtures/groups/; provenance (gap_expr, structure, GAP
version) is embedded so the mint never needs repeating.

Usage:
    python scripts/mint_group_tables.py [out_dir]

Default out_dir is tests/fixtures/groups/ next to this script's package.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# perm-group constructors give clean cycle-notation element strings
SPECS = [
    ("s3", "SymmetricGroup(3)"),
    ("c10", "CyclicGroup(IsPermGroup, 10)"),
    ("d8", "DihedralGroup(IsPermGroup, 8)"),
]

GAP_PROGRAM = """
G := {gap_expr};;
els := Elements(G);;
n := Size(G);;
Print("BEGIN_JSON\\n");
Print("{{\\"n\\": ", n, ", ");
Print("\\"gap_version\\": \\"", GAPInfo.Version, "\\", ");
Print("\\"structure\\": \\"", StructureDescription(G), "\\", ");
Print("\\"elem_strs\\": [");
for i in [1..n] do
  if i > 1 then Print(", "); fi;
  Print("\\"", String(els[i]), "\\"");
od;
Print("], \\"mult\\": [");
for i in [1..n] do
  if i > 1 then Print(", "); fi;
  Print("[");
  for j in [1..n] do
    if j > 1 then Print(", "); fi;
    Print(Position(els, els[i]*els[j]) - 1);
  od;
  Print("]");
od;
Print("], \\"inv\\": [");
for i in [1..n] do
  if i > 1 then Print(", "); fi;
  Print(Position(els, Inverse(els[i])) - 1);
od;
Print("]}}\\n");
Print("END_JSON\\n");
QUIT;
"""


def mint(gap_expr: str) -> dict:
    with tempfile.NamedTemporaryFile(
            "w", suffix=".g", delete=False, encoding="utf-8") as f:
        f.write(GAP_PROGRAM.format(gap_expr=gap_expr))
        path = f.name
    out = subprocess.run(["gap", "-q", "--norepl", path],
                         capture_output=True, text=True, timeout=300)
    Path(path).unlink()
    m = re.search(r"BEGIN_JSON\s*(.*?)\s*END_JSON", out.stdout, re.S)
    if not m:
        raise RuntimeError(
            f"gap produced no JSON for {gap_expr}:\n{out.stdout}\n{out.stderr}")
    return json.loads(m.group(1))


def main():
    default = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "groups"
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, gap_expr in SPECS:
        data = mint(gap_expr)
        prov = {"gap_expr": gap_expr, "structure": data["structure"],
                "gap_version": data["gap_version"], "minted_by": "mint_group_tables.py"}
        np.savez(out_dir / f"{stem}.npz",
                 elem_strs=np.array(data["elem_strs"], dtype="U"),
                 mult=np.array(data["mult"], dtype=np.int64),
                 inv=np.array(data["inv"], dtype=np.int64),
                 provenance=np.array(json.dumps(prov)))
        print(f"{stem}: n={data['n']} structure={data['structure']} "
              f"gap={data['gap_version']} -> {out_dir / (stem + '.npz')}")


if __name__ == "__main__":
    main()
