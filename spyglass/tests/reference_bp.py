"""Pure-numpy flooding min-sum reference, test-only.

Deliberately slow and obvious: dictionaries of per-edge messages, no
padding tricks, no vectorization. This is the oracle the production torch
implementation must match message-for-message, so it favors readability
over everything.
"""

import numpy as np


def reference_min_sum(problem, syndrome, *, max_iter=50, ms_scale=0.625,
                      gamma=0.0):
    """Flooding min-sum with optional memory on one syndrome.

    Returns:
        ``(hard, converged, messages)`` where ``messages`` maps
        ``("c2v", c, v)`` to the final check-to-variable message (for
        message-level parity tests).
    """
    H = problem.H
    m, n = H.shape
    edges = [(c, v) for c in range(m) for v in np.flatnonzero(H[c])]
    c2v = {(c, v): 0.0 for c, v in edges}
    llr_prev = problem.llr0.astype(np.float64).copy()
    syndrome = np.asarray(syndrome)

    hard = np.zeros(n, dtype=np.uint8)
    for _ in range(max_iter):
        # total LLR with memory, then extrinsic variable-to-check
        total = (gamma * llr_prev
                 + (1.0 - gamma) * problem.llr0
                 + np.array([sum(c2v[(c, v)] for c in np.flatnonzero(H[:, v]))
                             for v in range(n)]))
        v2c = {(c, v): total[v] - c2v[(c, v)] for c, v in edges}

        # check-to-variable min-sum with syndrome sign
        for c in range(m):
            nbrs = list(np.flatnonzero(H[c]))
            for v in nbrs:
                others = [v2c[(c, u)] for u in nbrs if u != v]
                sign = -1.0 if syndrome[c] else 1.0
                for x in others:
                    sign *= 1.0 if x >= 0 else -1.0
                mag = min(abs(x) for x in others) if others else 0.0
                c2v[(c, v)] = ms_scale * sign * mag

        llr_prev = total
        total_new = (problem.llr0
                     + np.array([sum(c2v[(c, v)]
                                     for c in np.flatnonzero(H[:, v]))
                                 for v in range(n)]))
        hard = (total_new < 0).astype(np.uint8)
        if np.array_equal(H @ hard % 2, syndrome):
            return hard, True, {("c2v", c, v): c2v[(c, v)] for c, v in edges}

    return hard, False, {("c2v", c, v): c2v[(c, v)] for c, v in edges}
