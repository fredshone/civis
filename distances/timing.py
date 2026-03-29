"""Timing distance.

Distances between schedules based on *when* activities happen, not just which.

Time-use vectors are extracted by ``distances.data.time_use_matrix`` — each
row is a shape-(T,) int32 array of ACTIVITY_TYPES indices, one value per
time bin.  Use ``resolution=1`` for per-minute precision (T=1440).

Two variants are provided:

Hamming timing distance
    Fraction of time bins where two time-use rows disagree, normalised to
    [0, 1].  Fast; captures gross timing mismatches.

Wasserstein (earth-mover) timing distance
    For a single chosen activity type, treats the occupied time bins as a
    1D distribution over the day and computes the 1D Wasserstein distance
    using ``scipy.stats.wasserstein_distance``.  Normalised by T.
    Captures *where* in the day an activity shifts.

Public API
----------
timing_distance(row1, row2) -> float
activity_timing_distance(row1, row2, activity_type_idx) -> float
pairwise_timing_distance(matrix) -> np.ndarray
pairwise_activity_timing_distance(matrix, activity_type_idx, n_jobs) -> np.ndarray
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wasserstein_distance as _wasserstein


# ---------------------------------------------------------------------------
# Scalar distances
# ---------------------------------------------------------------------------


def timing_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    """Fraction of time bins where two time-use rows disagree (Hamming).

    Parameters
    ----------
    row1, row2 : np.ndarray, shape (T,)
        Time-use vectors (integer activity-type indices per bin).

    Returns
    -------
    float
        Hamming distance normalised by T, in [0, 1].
    """
    t = len(row1)
    return float(np.sum(row1 != row2) / t)


def activity_timing_distance(
    row1: np.ndarray,
    row2: np.ndarray,
    activity_type_idx: int,
) -> float:
    """Wasserstein distance between per-day timing distributions of one activity.

    Treats the time bins occupied by *activity_type_idx* as a 1D probability
    distribution over the day and computes the earth-mover distance.

    Parameters
    ----------
    row1, row2 : np.ndarray, shape (T,)
        Time-use vectors (integer activity-type indices per bin).
    activity_type_idx : int
        Index into ACTIVITY_TYPES for the activity of interest.

    Returns
    -------
    float
        1D Wasserstein distance normalised by T, in [0, 1].
        Returns 0.0 when neither schedule contains the activity.
        Returns 1.0 when exactly one schedule contains the activity
        (maximum mismatch — one distribution has all its mass, the other
        has none).
    """
    t = len(row1)
    bins = np.arange(t, dtype=np.float64)

    w1 = (row1 == activity_type_idx).astype(np.float64)
    w2 = (row2 == activity_type_idx).astype(np.float64)
    s1, s2 = w1.sum(), w2.sum()

    if s1 == 0.0 and s2 == 0.0:
        return 0.0
    if s1 == 0.0 or s2 == 0.0:
        return 1.0

    w1 /= s1
    w2 /= s2
    return float(_wasserstein(bins, bins, w1, w2) / t)


# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------


def pairwise_timing_distance(matrix: np.ndarray) -> np.ndarray:
    """Pairwise Hamming timing distances for all rows in *matrix*.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, T)
        Time-use matrix as returned by ``time_use_matrix``.

    Returns
    -------
    np.ndarray, shape (N, N), float64
        Symmetric distance matrix, zero diagonal.
    """
    t = matrix.shape[1]
    disagree = matrix[:, np.newaxis, :] != matrix[np.newaxis, :, :]
    return disagree.sum(axis=2).astype(np.float64) / t


def pairwise_activity_timing_distance(
    matrix: np.ndarray,
    activity_type_idx: int,
    n_jobs: int = 1,
) -> np.ndarray:
    """Pairwise Wasserstein timing distances for a single activity type.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, T)
        Time-use matrix.
    activity_type_idx : int
        Activity type index (see ACTIVITY_TYPES).
    n_jobs : int
        Parallel workers (joblib).  -1 = all CPUs.

    Returns
    -------
    np.ndarray, shape (N, N), float64
        Symmetric distance matrix, zero diagonal.
    """
    n = matrix.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(activity_timing_distance)(matrix[i], matrix[j], activity_type_idx)
        for i, j in pairs
    )

    D = np.zeros((n, n), dtype=np.float64)
    for (i, j), d in zip(pairs, results):
        D[i, j] = d
        D[j, i] = d
    return D
