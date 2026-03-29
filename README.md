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
