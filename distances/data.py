"""Schedule data structures and loading utilities.

Schedules are represented entirely as Polars DataFrames; derived array
representations are obtained via extractor functions that go directly from
DataFrame to NumPy arrays.

Input tables
------------
activities
    Columns: pid (Utf8), seq (Int32), act (Utf8), zone (Utf8),
    start (Int32), end (Int32).  Times are minutes since midnight [0, 1440].

attributes
    Columns: pid (Utf8), hid (Utf8), age (Int32), hh_size (Int32),
    hh_income (Float64), sex, dwelling, ownership, vehicles (Int32),
    disability, education, can_wfh, occupation, race, has_licence,
    relationship, employment, country, source (Utf8), year (Int32),
    month, day, hh_zone, weight (Float64), access_egress_distance (Float64),
    max_temp_c (Float64), rain, avg_speed (Float64).

Constants
---------
ACTIVITY_TYPES
    Canonical tuple of the nine activity-type strings.  Index positions
    are used as integer codes in ``time_use_matrix``.

Public API
----------
load_activities(path) -> pl.DataFrame
load_attributes(path) -> pl.DataFrame
participation_matrix(activities) -> tuple[list[str], np.ndarray]
time_use_matrix(activities, resolution) -> tuple[list[str], np.ndarray]
activity_sequences(activities) -> tuple[list[str], list[list[str]]]
sequence_2gram_matrix(activities) -> tuple[list[str], np.ndarray]
sequence_2gram_matrix_from_sequences(sequences) -> np.ndarray
print_summary(activities, attributes) -> None
plot_schedules(activities, n, show) -> matplotlib.figure.Figure
plot_activity_frequencies(activities, show) -> matplotlib.figure.Figure
plot_duration_distributions(activities, show) -> matplotlib.figure.Figure
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ACTIVITY_TYPES: tuple[str, ...] = (
    "home",
    "work",
    "education",
    "leisure",
    "medical",
    "escort",
    "other",
    "visit",
    "shop",
)

_ACT_INDEX: dict[str, int] = {a: i for i, a in enumerate(ACTIVITY_TYPES)}

_ACT_COLOURS: dict[str, str] = {
    "home": "#4e79a7",
    "work": "#f28e2b",
    "education": "#59a14f",
    "leisure": "#76b7b2",
    "medical": "#e15759",
    "escort": "#ff9da7",
    "other": "#b07aa1",
    "visit": "#edc948",
    "shop": "#9c755f",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_activities(path: str | Path) -> pl.DataFrame:
    """Load an activities CSV or Parquet file and return a typed DataFrame.

    Parameters
    ----------
    path:
        Path to a CSV or Parquet file with columns: pid, seq, act, zone, start, end.

    Returns
    -------
    pl.DataFrame
        pid (Utf8), seq (Int32), act (Utf8), zone (Utf8),
        start (Int32), end (Int32).

    Raises
    ------
    ValueError
        If the file extension is not .csv or .parquet.
    """
    path = Path(path)
    schema = {
        "pid": pl.Utf8,
        "seq": pl.Int32,
        "act": pl.Utf8,
        "zone": pl.Utf8,
        "start": pl.Int32,
        "end": pl.Int32,
    }
    if path.suffix == ".csv":
        return pl.read_csv(path, schema_overrides=schema)
    elif path.suffix == ".parquet":
        return pl.read_parquet(path).cast(schema)
    else:
        raise ValueError(
            f"Unsupported format: {path.suffix!r}. Expected .csv or .parquet"
        )


def load_attributes(path: str | Path) -> pl.DataFrame:
    """Load an attributes CSV or Parquet file and return a typed DataFrame.

    Parameters
    ----------
    path:
        Path to a CSV or Parquet file matching the foundata attributes schema.

    Returns
    -------
    pl.DataFrame
        String columns as Utf8; numeric columns cast where unambiguous.
        Empty strings and ``"void"`` are treated as null (CSV only; Parquet
        stores nulls natively).

    Raises
    ------
    ValueError
        If the file extension is not .csv or .parquet.
    """
    path = Path(path)
    schema = {
        "hid": pl.Utf8,
        "pid": pl.Utf8,
        "age": pl.Int32,
        "hh_size": pl.Int32,
        "hh_income": pl.Float64,
        "sex": pl.Utf8,
        "dwelling": pl.Utf8,
        "ownership": pl.Utf8,
        "vehicles": pl.Int32,
        "disability": pl.Utf8,
        "education": pl.Utf8,
        "can_wfh": pl.Utf8,
        "occupation": pl.Utf8,
        "race": pl.Utf8,
        "has_licence": pl.Utf8,
        "relationship": pl.Utf8,
        "employment": pl.Utf8,
        "country": pl.Utf8,
        "source": pl.Utf8,
        "year": pl.Int32,
        "month": pl.Utf8,
        "day": pl.Utf8,
        "hh_zone": pl.Utf8,
        "weight": pl.Float64,
        "access_egress_distance": pl.Float64,
        "max_temp_c": pl.Float64,
        "rain": pl.Utf8,
        "avg_speed": pl.Float64,
    }
    if path.suffix == ".csv":
        return pl.read_csv(path, schema_overrides=schema, null_values=["", "void"])
    elif path.suffix == ".parquet":
        return pl.read_parquet(path).cast(schema)
    else:
        raise ValueError(
            f"Unsupported format: {path.suffix!r}. Expected .csv or .parquet"
        )


# ---------------------------------------------------------------------------
# Extractor functions: DataFrame -> arrays
# ---------------------------------------------------------------------------


def participation_matrix(
    activities: pl.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Fractional-duration participation vectors for all persons.

    Parameters
    ----------
    activities:
        DataFrame as returned by :func:`load_activities`.

    Returns
    -------
    pids : list[str]
        Person IDs in sorted order, length N.
    matrix : np.ndarray, shape (N, 9), float64
        Row ``i`` is the participation vector for ``pids[i]``: fraction of
        1440 minutes spent in each activity type (indexed by
        ``ACTIVITY_TYPES``).  Each row sums to 1.0.
    """
    df = activities.with_columns((pl.col("end") - pl.col("start")).alias("duration"))
    agg_exprs = [
        pl.col("duration").filter(pl.col("act") == act).sum().alias(act)
        for act in ACTIVITY_TYPES
    ]
    grouped = df.group_by("pid").agg(agg_exprs).sort("pid")
    pids: list[str] = grouped["pid"].to_list()
    mat = grouped.select(list(ACTIVITY_TYPES)).to_numpy().astype(np.float64)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return pids, mat / row_sums


def time_use_matrix(
    activities: pl.DataFrame,
    resolution: int = 10,
) -> tuple[list[str], np.ndarray]:
    """Per-bin activity-type index matrix for all persons.

    Each bin of ``resolution`` minutes is assigned the activity type with
    the most minutes in that bin (ties broken by lower ``ACTIVITY_TYPES``
    index).

    Parameters
    ----------
    activities:
        DataFrame as returned by :func:`load_activities`.
    resolution:
        Bin width in minutes.  Must divide 1440 exactly.  Default 10
        gives 144 bins; use 1 for per-minute precision.

    Returns
    -------
    pids : list[str]
        Person IDs in sorted order, length N.
    matrix : np.ndarray, shape (N, 1440 // resolution), int32
        Entry ``[i, b]`` is the ``ACTIVITY_TYPES`` index of the majority
        activity in bin ``b`` (minutes ``[b*resolution, (b+1)*resolution)``)
        for person ``pids[i]``.  Uncovered minutes default to index 0 (home).
    """
    if 1440 % resolution != 0:
        raise ValueError(f"resolution={resolution} does not divide 1440 exactly")

    pids: list[str] = activities["pid"].unique().sort().to_list()
    n = len(pids)

    pid_df = pl.DataFrame({"pid": pids, "_pid_idx": list(range(n))})
    act_df = pl.DataFrame(
        {"act": list(ACTIVITY_TYPES), "_act_idx": list(range(len(ACTIVITY_TYPES)))}
    )
    enriched = (
        activities.join(pid_df, on="pid")
        .join(act_df, on="act", how="left")
        .with_columns(pl.col("_act_idx").fill_null(0))
        .select(["_pid_idx", "start", "end", "_act_idx"])
    )

    pid_indices = enriched["_pid_idx"].to_numpy()
    starts = enriched["start"].to_numpy()
    ends = enriched["end"].to_numpy()
    act_indices = enriched["_act_idx"].to_numpy()

    # Build at 1-minute resolution, then downsample
    mat_1min = np.zeros((n, 1440), dtype=np.int32)
    for pid_idx, start, end, act_idx in zip(pid_indices, starts, ends, act_indices):
        mat_1min[pid_idx, start:end] = act_idx

    if resolution == 1:
        return pids, mat_1min

    n_bins = 1440 // resolution
    n_acts = len(ACTIVITY_TYPES)
    # shape (N, n_bins, resolution) — count minutes per activity type per bin
    bins = mat_1min.reshape(n, n_bins, resolution)
    counts = np.zeros((n, n_bins, n_acts), dtype=np.int32)
    for a in range(n_acts):
        counts[:, :, a] = (bins == a).sum(axis=2)
    return pids, counts.argmax(axis=2).astype(np.int32)


def activity_sequences(
    activities: pl.DataFrame,
) -> tuple[list[str], list[list[str]]]:
    """Ordered activity-type sequences for all persons.

    Parameters
    ----------
    activities:
        DataFrame as returned by :func:`load_activities`.

    Returns
    -------
    pids : list[str]
        Person IDs in sorted order, length N.
    sequences : list[list[str]]
        ``sequences[i]`` is the ordered list of activity-type strings for
        ``pids[i]``, sorted by start time.
    """
    grouped = (
        activities.sort(["pid", "start"])
        .group_by("pid", maintain_order=True)
        .agg(pl.col("act"))
        .sort("pid")
    )
    pids: list[str] = grouped["pid"].to_list()
    seqs: list[list[str]] = grouped["act"].to_list()
    return pids, seqs


def sequence_2gram_matrix_from_sequences(
    sequences: list[list[str]],
) -> np.ndarray:
    """Normalised 2-gram transition vectors from activity sequences.

    Parameters
    ----------
    sequences:
        Ordered activity-type sequences per person.

    Returns
    -------
    np.ndarray, shape (N, 81), float64
        Row ``i`` contains the normalised counts of all activity 2-grams for
        person ``i`` using the canonical ``ACTIVITY_TYPES × ACTIVITY_TYPES``
        vocabulary in row-major order.
    """
    n = len(sequences)
    n_acts = len(ACTIVITY_TYPES)
    n_features = n_acts * n_acts
    mat = np.zeros((n, n_features), dtype=np.float64)

    for i, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
        for prev_act, next_act in zip(seq[:-1], seq[1:]):
            prev_idx = _ACT_INDEX.get(prev_act)
            next_idx = _ACT_INDEX.get(next_act)
            if prev_idx is None or next_idx is None:
                continue
            feature_idx = prev_idx * n_acts + next_idx
            mat[i, feature_idx] += 1.0

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return mat / row_sums


def sequence_2gram_matrix(
    activities: pl.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Build per-person normalised 2-gram transition vectors.

    Parameters
    ----------
    activities:
        DataFrame as returned by :func:`load_activities`.

    Returns
    -------
    pids : list[str]
        Person IDs in sorted order, length N.
    matrix : np.ndarray, shape (N, 81), float64
        Normalised activity 2-gram vectors for each person.
    """
    pids, sequences = activity_sequences(activities)
    return pids, sequence_2gram_matrix_from_sequences(sequences)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_summary(activities: pl.DataFrame, attributes: pl.DataFrame) -> None:
    """Print summary statistics for activities and attributes to stdout."""
    n_persons = activities["pid"].n_unique()
    n_rows = len(activities)
    print(f"Activities: {n_rows:,} rows, {n_persons:,} persons")

    seq_lengths = activities.group_by("pid").agg(pl.len().alias("n"))["n"]
    print(
        f"Sequence length: min={seq_lengths.min()}, "
        f"mean={seq_lengths.mean():.1f}, "
        f"max={seq_lengths.max()}"
    )

    print("\nActivity type frequencies:")
    act_counts = (
        activities.group_by("act")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    for row in act_counts.iter_rows(named=True):
        pct = 100 * row["count"] / n_rows
        print(f"  {row['act']:12s}  {row['count']:>10,}  ({pct:.1f}%)")

    print(f"\nAttributes: {len(attributes):,} rows")
    if "source" in attributes.columns:
        print("Source counts:")
        for row in (
            attributes.group_by("source")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .iter_rows(named=True)
        ):
            print(f"  {str(row['source']):10s}  {row['count']:>10,}")

    print("\nAttribute missingness (% null):")
    for col in attributes.columns:
        null_pct = 100 * attributes[col].null_count() / len(attributes)
        if null_pct > 0:
            print(f"  {col:30s}  {null_pct:.1f}%")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_schedules(
    activities: pl.DataFrame,
    n: int = 20,
    show: bool = True,
):
    """Plot a random sample of schedules as horizontal Gantt bar charts.

    Parameters
    ----------
    activities:
        DataFrame as returned by :func:`load_activities`.
    n:
        Number of persons to plot (randomly sampled if more exist).
    show:
        If ``True``, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    all_pids = activities["pid"].unique().to_list()
    if len(all_pids) > n:
        rng = np.random.default_rng()
        sample_pids = rng.choice(all_pids, size=n, replace=False).tolist()
    else:
        sample_pids = all_pids

    subset = activities.filter(pl.col("pid").is_in(sample_pids)).sort(["pid", "start"])

    fig, ax = plt.subplots(figsize=(14, max(4, len(sample_pids) * 0.4)))
    pid_order = sorted(set(subset["pid"].to_list()))
    pid_y = {p: i for i, p in enumerate(pid_order)}

    seen_acts: set[str] = set()
    for row in subset.iter_rows(named=True):
        act = row["act"]
        ax.barh(
            pid_y[row["pid"]],
            row["end"] - row["start"],
            left=row["start"],
            height=0.8,
            color=_ACT_COLOURS.get(act, "#aaaaaa"),
            label=act if act not in seen_acts else "_nolegend_",
        )
        seen_acts.add(act)

    ax.legend(loc="lower right", fontsize=8)
    ax.set_yticks(range(len(pid_order)))
    ax.set_yticklabels(pid_order, fontsize=6)
    ax.set_xlabel("Minutes since midnight")
    ax.set_xlim(0, 1440)
    ax.set_title(f"Sample of {len(pid_order)} schedules")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_activity_frequencies(
    activities: pl.DataFrame,
    show: bool = True,
):
    """Bar chart of activity-type frequencies, optionally broken down by source.

    Parameters
    ----------
    activities:
        DataFrame as returned by :func:`load_activities`, optionally with
        a ``source`` column joined from attributes.
    show:
        If ``True``, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    acts = list(ACTIVITY_TYPES)
    fig, ax = plt.subplots(figsize=(12, 5))

    if "source" not in activities.columns:
        counts_map = {
            row["act"]: row["count"]
            for row in activities.group_by("act")
            .agg(pl.len().alias("count"))
            .iter_rows(named=True)
        }
        ax.bar(acts, [counts_map.get(a, 0) for a in acts])
        ax.set_title("Activity type frequencies")
    else:
        sources = activities["source"].drop_nulls().unique().sort().to_list()
        x = np.arange(len(acts))
        width = 0.8 / len(sources)
        for i, src in enumerate(sources):
            counts_map = {
                row["act"]: row["count"]
                for row in activities.filter(pl.col("source") == src)
                .group_by("act")
                .agg(pl.len().alias("count"))
                .iter_rows(named=True)
            }
            ax.bar(
                x + i * width, [counts_map.get(a, 0) for a in acts], width, label=src
            )
        ax.set_xticks(x + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(acts, rotation=30, ha="right")
        ax.set_title("Activity type frequencies by source")
        ax.legend()

    ax.set_ylabel("Count")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_duration_distributions(
    activities: pl.DataFrame,
    show: bool = True,
):
    """Violin plots of duration distributions per activity type.

    Parameters
    ----------
    activities:
        DataFrame as returned by :func:`load_activities`.
    show:
        If ``True``, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    df = activities.with_columns((pl.col("end") - pl.col("start")).alias("duration"))
    labels = []
    data = []
    for act_type in ACTIVITY_TYPES:
        durs = df.filter(pl.col("act") == act_type)["duration"].to_numpy()
        if len(durs) > 0:
            data.append(durs)
            labels.append(act_type)

    fig, ax = plt.subplots(figsize=(12, 5))
    if data:
        parts = ax.violinplot(data, showmedians=True)
        for pc, act_type in zip(parts["bodies"], labels):
            pc.set_facecolor(_ACT_COLOURS.get(act_type, "#aaaaaa"))
            pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Duration distributions per activity type")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
