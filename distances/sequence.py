"""Sequence edit distance.

Distances between schedules based on ordered activity-type sequences.
Sequences are extracted by ``distances.data.activity_sequences`` — each is a
list[str] drawn from ACTIVITY_TYPES.

The edit distance uses dynamic programming with configurable substitution
costs and is normalised by the length of the longer sequence so the result
lies in [0, 1]:

* 0 when both sequences are identical.
* 1 when the sequences are completely dissimilar and have equal length.

A semantic substitution cost matrix (``DEFAULT_COST_MATRIX``) assigns lower
costs to substitutions between functionally related activity types
(e.g. work ↔ education costs 0.5).  Insertion and deletion always cost 1.

Public API
----------
DEFAULT_COST_MATRIX : dict[tuple[str, str], float]
edit_distance(seq1, seq2, cost_matrix=None) -> float
pairwise_sequence_distance(sequences, cost_matrix=None, n_jobs=1) -> np.ndarray
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from distances.data import ACTIVITY_TYPES

# ---------------------------------------------------------------------------
# Semantic substitution cost matrix
# ---------------------------------------------------------------------------

_REDUCED: dict[frozenset, float] = {
    # frozenset({"work", "education"}): 0.5,
    # frozenset({"leisure", "visit"}): 0.5,
    # frozenset({"leisure", "shop"}): 0.5,
    # frozenset({"escort", "other"}): 0.5,
}


def _build_default_cost_matrix() -> dict[tuple[str, str], float]:
    costs: dict[tuple[str, str], float] = {}
    for a in ACTIVITY_TYPES:
        for b in ACTIVITY_TYPES:
            costs[(a, b)] = 0.0 if a == b else _REDUCED.get(frozenset({a, b}), 1.0)
    return costs


DEFAULT_COST_MATRIX: dict[tuple[str, str], float] = _build_default_cost_matrix()

# ---------------------------------------------------------------------------
# Scalar distance
# ---------------------------------------------------------------------------


def edit_distance(
    seq1: list[str],
    seq2: list[str],
    cost_matrix: dict[tuple[str, str], float] | None = None,
) -> float:
    """Normalised edit distance between two activity-type sequences.

    Parameters
    ----------
    seq1, seq2 : list[str]
        Activity-type sequences (elements from ACTIVITY_TYPES).
    cost_matrix : dict[(str, str), float] | None
        Substitution costs.  If None, unit costs are used (every substitution
        costs 1).  Pass ``DEFAULT_COST_MATRIX`` for the semantic variant.
        Insertion/deletion always cost 1 regardless.

    Returns
    -------
    float
        Edit distance normalised by max(len(seq1), len(seq2)), in [0, 1].
        Returns 0.0 when both sequences are empty.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 and m == 0:
        return 0.0
    denom = float(max(n, m))

    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    dp[:, 0] = np.arange(n + 1, dtype=np.float64)
    dp[0, :] = np.arange(m + 1, dtype=np.float64)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = seq1[i - 1], seq2[j - 1]
            if a == b:
                sub = 0.0
            elif cost_matrix is not None:
                sub = cost_matrix.get((a, b), 1.0)
            else:
                sub = 1.0
            dp[i, j] = min(
                dp[i - 1, j] + 1.0,
                dp[i, j - 1] + 1.0,
                dp[i - 1, j - 1] + sub,
            )

    return float(dp[n, m] / denom)


# ---------------------------------------------------------------------------
# Pairwise distance
# ---------------------------------------------------------------------------


def pairwise_sequence_distance(
    sequences: list[list[str]],
    cost_matrix: dict[tuple[str, str], float] | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Pairwise normalised edit distances for all sequences.

    Parameters
    ----------
    sequences : list[list[str]]
        Activity-type sequences.
    cost_matrix : dict[(str, str), float] | None
        Substitution costs.  See ``edit_distance``.
    n_jobs : int
        Number of parallel workers (joblib).  -1 = all CPUs.

    Returns
    -------
    np.ndarray, shape (N, N), float64
        Symmetric distance matrix, zero diagonal.
    """
    n = len(sequences)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(edit_distance)(sequences[i], sequences[j], cost_matrix)
        for i, j in pairs
    )

    D = np.zeros((n, n), dtype=np.float64)
    for (i, j), d in zip(pairs, results):
        D[i, j] = d
        D[j, i] = d
    return D
