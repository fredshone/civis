"""Activity participation distance.

Distances between schedules based on fractional activity-type participation
vectors.  Participation vectors are float64 arrays of length 9 (one entry per
``ACTIVITY_TYPES``, summing to 1.0) as produced by
``distances.data.participation_matrix``.

The distance is half the L1 norm, equivalently the total variation distance
between two discrete probability distributions.  It lies in [0, 1]:

* 0 when both schedules have identical participation fractions.
* 1 when the two schedules share no activity types at all.

Public API
----------
participation_distance(v1, v2) -> float
pairwise_participation_distance(matrix) -> np.ndarray shape (N, N)
"""

from __future__ import annotations

import numpy as np


def participation_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """L1 distance between two participation vectors, normalised to [0, 1].

    Equivalent to the total variation distance between the two distributions.

    Parameters
    ----------
    v1, v2 : np.ndarray, shape (9,)
        Participation vectors as returned by a single row of
        ``participation_matrix``.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    return float(np.sum(np.abs(v1 - v2)) / 2.0)


def pairwise_participation_distance(matrix: np.ndarray) -> np.ndarray:
    """Pairwise participation distances for all persons in a matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (N, 9)
        Participation matrix as returned by ``participation_matrix``.

    Returns
    -------
    np.ndarray, shape (N, N), float64
        Symmetric distance matrix where ``D[i, j]`` equals
        ``participation_distance(matrix[i], matrix[j])``.  Diagonal is zero.
    """
    diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]  # (N, N, 9)
    return np.sum(np.abs(diff), axis=2) / 2.0
