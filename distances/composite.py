"""Composite schedule distance helpers.

The project now assumes a fixed composite made from participation vectors,
sequence 2-gram vectors, and timing matrices.

Public API
----------
composite_distance(part1, part2, seq2gram1, seq2gram2, time1, time2, weights) -> float
"""

from __future__ import annotations

import numpy as np
from distances.participation import participation_distance
from distances.timing import timing_distance


def composite_distance(
    part1: np.ndarray,
    part2: np.ndarray,
    seq2gram1: np.ndarray,
    seq2gram2: np.ndarray,
    time1: np.ndarray,
    time2: np.ndarray,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> float:
    """Weighted sum of the three normalised component distances.

    Parameters
    ----------
    part1, part2 : np.ndarray, shape (9,)
        Participation vectors from ``participation_matrix``.
    seq2gram1, seq2gram2 : np.ndarray, shape (81,)
        Normalised activity transition 2-gram vectors.
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
    d_seq = float(np.sum(np.abs(seq2gram1 - seq2gram2)) / 2.0)
    d_time = timing_distance(time1, time2)
    return float(w[0] * d_part + w[1] * d_seq + w[2] * d_time)
