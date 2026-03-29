# civis

Attribute Embedding Learning for Human Activity Schedules.

## distances.data

Loading and array extraction for activity schedule data.

```python
from distances.data import (
    load_activities, load_attributes,
    participation_matrix, time_use_matrix, activity_sequences,
)

acts  = load_activities("data/activities.csv")
attrs = load_attributes("data/attributes.csv")

# (N, 9) float64 — fraction of day spent in each activity type, rows sum to 1
pids, part = participation_matrix(acts)

# (N, 144) int32 — majority activity-type index per 10-min bin (default)
pids, tuse = time_use_matrix(acts)
pids, tuse = time_use_matrix(acts, resolution=5)   # (N, 288) bins

# list of ordered activity-type string lists
pids, seqs = activity_sequences(acts)
```

Activity types are indexed by `ACTIVITY_TYPES`:
`home, work, education, leisure, medical, escort, other, visit, shop`.

For exploration, `python -m distances.data [activities.csv] [attributes.csv]`
prints summary statistics.

## distances.sequence

Normalised edit distance between activity-type sequences.

```python
from distances.sequence import edit_distance, pairwise_sequence_distance, DEFAULT_COST_MATRIX

# Unit costs (every substitution = 1, normalised by max sequence length)
d = edit_distance(["home", "work", "home"], ["home", "education", "leisure", "home"])

# Semantic costs — work↔education, leisure↔visit, etc. cost 0.5
d = edit_distance(seq1, seq2, cost_matrix=DEFAULT_COST_MATRIX)

# Pairwise matrix for a list of sequences (parallelised with joblib)
D = pairwise_sequence_distance(sequences, cost_matrix=DEFAULT_COST_MATRIX, n_jobs=-1)
```

## distances.timing

Timing distances capturing *when* activities happen.

```python
from distances.timing import (
    timing_distance, activity_timing_distance,
    pairwise_timing_distance, pairwise_activity_timing_distance,
)
from distances.data import time_use_matrix

# (N, 1440) int32 — majority activity-type index per minute
pids, tuse = time_use_matrix(acts, resolution=1)

# Hamming: fraction of minutes that disagree, in [0, 1]
d = timing_distance(tuse[0], tuse[1])
D = pairwise_timing_distance(tuse)

# Wasserstein: earth-mover distance for one activity type's daily timing
d = activity_timing_distance(tuse[0], tuse[1], activity_type_idx=1)  # 1 = work
D = pairwise_activity_timing_distance(tuse, activity_type_idx=1, n_jobs=-1)
```

## distances.composite

Weighted combination of all three distance components.

```python
from distances.composite import composite_distance, pairwise_composite_distance

# Scalar composite for a single pair
d = composite_distance(part1, part2, seq1, seq2, time1, time2, weights=(1/3, 1/3, 1/3))

# Full pairwise matrix (parallelised; optional disk cache)
pids, D = pairwise_composite_distance(
    acts,
    weights=(0.4, 0.3, 0.3),
    n_jobs=-1,
    cache_path="experiments/distances.npz",
)
```
