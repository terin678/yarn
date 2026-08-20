"""Decode code-capacity shots on a shipped processor code, end to end.

Run from the repository root (needs the gpu extra, or a CPU torch build):

    python spyglass/examples/quickstart.py
"""

from pathlib import Path

import numpy as np

from spyglass import TelescopingDecoder
from spyglass.noise import sample_code_capacity

CODE = Path(__file__).resolve().parents[2] / "processor_codes" / "mitten" \
    / "[[150,30,10]]"
P = 0.02
SHOTS = 5_000
SEED = 1


def decode_direction(name, H, L):
    errors, syndromes = sample_code_capacity(
        H, P, SHOTS, np.random.default_rng(SEED))
    dec = TelescopingDecoder(H, np.full(H.shape[1], P), seed=SEED)
    result = dec.decode_batch(syndromes)

    residual = (errors ^ result.corrections) @ L.T % 2
    logical_failures = int((residual.any(axis=1)).sum())
    print(f"\n{name}: {SHOTS} shots at p={P}")
    print(f"  resolved: {int(result.converged.sum())}/{SHOTS}")
    print(f"  logical failures: {logical_failures} "
          f"(rate {logical_failures / SHOTS:.2e})")
    t = result.telemetry
    for stage, seen, done, secs in zip(t.stage_names, t.shots_in,
                                       t.shots_resolved, t.wall_seconds):
        print(f"  {stage:>10}: saw {seen:>6}  resolved {done:>6}  "
              f"{secs:8.3f}s")


def main():
    Hx = np.load(CODE / "Hx.npy").astype(np.uint8)
    Hz = np.load(CODE / "Hz.npy").astype(np.uint8)
    Lx = np.load(CODE / "Lx.npy").astype(np.uint8)
    Lz = np.load(CODE / "Lz.npy").astype(np.uint8)
    # X errors flip Z checks and are witnessed by Z logicals, and vice
    # versa; composing the two directions is the caller's job by design
    decode_direction("X errors (Hz syndrome)", Hz, Lz)
    decode_direction("Z errors (Hx syndrome)", Hx, Lx)


if __name__ == "__main__":
    main()
