"""Composite schedule distance.

Combines the three component distances — participation, sequence, timing —
into a single normalised distance.  Component distances are weighted and
summed; weights need not sum to 1 (they are normalised internally).

Pairwise computation over a full dataset is expensive: each pair requires
an edit-distance DP solve plus two array comparisons.  The pairwise function
parallelises with joblib and supports caching the result to disk as a
compressed NumPy archive so it can be reused across training runs.

Public API
----------
composite_distance(part1, part2, seq1, seq2, time1, time2, weights) -> float
pairwise_composite_distance(activities, weights, n_jobs, cache_path)
    -> tuple[list[str], np.ndarray]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from distances.data import activity_sequences, participation_matrix, time_use_matrix
from distances.participation import participation_distance
from distances.sequence import DEFAULT_COST_MATRIX, edit_distance
from distances.timing import timing_distance


def composite_distance(
    part1: np.ndarray,
    part2: np.ndarray,
    seq1: list[str],
    seq2: list[str],
    time1: np.ndarray,
    time2: np.ndarray,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> float:
    """Weighted sum of the three normalised component distances.

    Parameters
    ----------
    part1, part2 : np.ndarray, shape (9,)
        Participation vectors from ``participation_matrix``.
    seq1, seq2 : list[str]
        Activity-type sequences from ``activity_sequences``.
    time1, time2 : np.ndarray, shape (T,)
        Time-use rows from ``time_use_matrix``.
    weights : tuple[float, float, float]
        (w_participation, w_sequence, w_timing).  Need not sum to 1.

    Returns
    -------
    float
        Composite distance in [0, 1].
    """
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    d_part = participation_distance(part1, part2)
    d_seq = edit_distance(seq1, seq2, DEFAULT_COST_MATRIX)
    d_time = timing_distance(time1, time2)
    return float(w[0] * d_part + w[1] * d_seq + w[2] * d_time)


def pairwise_composite_distance(
    activities: pl.DataFrame,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    n_jobs: int = -1,
    cache_path: str | Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Compute the full pairwise composite distance matrix for a dataset.

    Extracts participation, sequence, and time-use representations from
    *activities*, then computes upper-triangle pairwise composite distances
    in parallel.  The result is optionally cached to a compressed ``.npz``
    file for reuse across training runs.

    Parameters
    ----------
    activities : pl.DataFrame
        Activities table (``load_activities`` output).
    weights : tuple[float, float, float]
        Component weights — see ``composite_distance``.
    n_jobs : int
        Parallel workers.  -1 = all CPUs.
    cache_path : str | Path | None
        If provided, save/load the matrix from this ``.npz`` file.

    Returns
    -------
    pids : list[str]
        Person identifiers in row/column order.
    D : np.ndarray, shape (N, N)
        Symmetric composite distance matrix, zero diagonal.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return list(data["pids"]), data["D"]

    pids_p, part_mat = participation_matrix(activities)
    pids_s, seqs = activity_sequences(activities)
    pids_t, time_mat = time_use_matrix(activities, resolution=1)

    assert pids_p == pids_s == pids_t, "pid ordering mismatch between extractors"
    pids = pids_p
    n = len(pids)

    w = np.array(weights, dtype=np.float64)
    w /= w.sum()

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def _compute(i: int, j: int) -> float:
        d_part = participation_distance(part_mat[i], part_mat[j])
        d_seq = edit_distance(seqs[i], seqs[j], DEFAULT_COST_MATRIX)
        d_time = timing_distance(time_mat[i], time_mat[j])
        return float(w[0] * d_part + w[1] * d_seq + w[2] * d_time)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute)(i, j) for i, j in pairs
    )

    D = np.zeros((n, n), dtype=np.float64)
    for (i, j), d in zip(pairs, results):
        D[i, j] = d
        D[j, i] = d

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, pids=np.array(pids), D=D)

    return pids, D
