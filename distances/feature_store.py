"""Feature precomputation and versioned storage for schedule representations.

Extracts participation, sequence, sequence 2-gram, and timing features once and persists them
with content-based versioning.  Enables multiple distance metrics to reuse
the same precomputed base features.

Public API
----------
FeatureManifest
build_schedule_features
load_schedule_features
feature_manifest_hash
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np
import polars as pl

from distances.data import (
    activity_sequences,
    participation_matrix,
    sequence_2gram_matrix_from_sequences,
    time_use_matrix,
)


@dataclass
class FeatureManifest:
    """Metadata and versioning for a precomputed feature set.

    Parameters
    ----------
    data_hash : str
        SHA256 hash of the activities DataFrame (row-sorted).
    feature_schema : str
        Description of features extracted (e.g. "participation-9d, sequence-ordered, timing-1440-bins").
    extraction_timestamp : str
        ISO 8601 timestamp when features were computed.
    n_persons : int
        Number of unique persons in the feature set.
    n_activity_records : int
        Number of activity records processed.
    timing_resolution : int
        Bin width in minutes for timing features.
    feature_dir : str
        Directory where feature artifacts are stored (relative or absolute).
    """

    data_hash: str
    feature_schema: str
    extraction_timestamp: str
    n_persons: int
    n_activity_records: int
    timing_resolution: int
    feature_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FeatureManifest":
        return FeatureManifest(**d)


@dataclass
class ScheduleFeatures:
    """Lightweight container for precomputed schedule features.

    Parameters
    ----------
    pids : list[str]
        Person identifiers in sorted order.
    participation : np.ndarray, shape (N, 9), float64
        Fractional activity participation vectors.
    sequences : list[list[str]]
        Ordered activity-type sequences.
    sequence_2grams : np.ndarray, shape (N, 81), float64
        Normalised activity transition 2-gram vectors.
    timing : np.ndarray, shape (N, T), int32
        Time-use binary matrices (activity type per bin).
    manifest : FeatureManifest
        Metadata and versioning info.
    """

    pids: list[str]
    participation: np.ndarray
    sequences: list[list[str]]
    sequence_2grams: np.ndarray
    timing: np.ndarray
    manifest: FeatureManifest


def _compute_data_hash(activities: pl.DataFrame) -> str:
    """Compute a deterministic hash of the activities data.

    Sorts by pid and activity sequence to ensure reproducibility.
    """
    sorted_df = activities.sort(["pid", "seq"])
    data_repr = json.dumps(
        sorted_df.rows(named=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    data_bytes = data_repr.encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()


def build_schedule_features(
    activities: pl.DataFrame,
    feature_dir: str | Path,
    timing_resolution: int = 1,
    overwrite: bool = False,
) -> ScheduleFeatures:
    """Extract and save schedule features with versioning.

    Parameters
    ----------
    activities : pl.DataFrame
        Activities DataFrame (from ``load_activities``).
    feature_dir : str | Path
        Directory where feature artifacts will be saved.
    timing_resolution : int
        Bin width in minutes for time-use matrix.
    overwrite : bool
        If False and features exist, load cached version.
        If True, recompute and overwrite.

    Returns
    -------
    ScheduleFeatures
        Extracted features and manifest.
    """
    from datetime import datetime, timezone

    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = feature_dir / "manifest.json"

    # Check if cached features exist
    if manifest_path.exists() and not overwrite:
        return load_schedule_features(feature_dir)

    # Compute data hash for versioning
    data_hash = _compute_data_hash(activities)

    # Extract features
    pids_p, part_mat = participation_matrix(activities)
    pids_s, seqs = activity_sequences(activities)
    seq2_mat = sequence_2gram_matrix_from_sequences(seqs)
    pids_t, time_mat = time_use_matrix(activities, resolution=timing_resolution)

    if not (pids_p == pids_s == pids_t):
        raise ValueError("pid ordering mismatch between extractors")

    pids = pids_p
    n_persons = len(pids)
    n_records = len(activities)

    feature_schema = (
        f"participation-9d, sequence-ordered, sequence-2gram-81d, "
        f"timing-{time_mat.shape[1]}-bins-res{timing_resolution}min"
    )

    manifest = FeatureManifest(
        data_hash=data_hash,
        feature_schema=feature_schema,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        n_persons=n_persons,
        n_activity_records=n_records,
        timing_resolution=timing_resolution,
        feature_dir=str(feature_dir),
    )

    # Save features as compressed npz
    features_path = feature_dir / "features.npz"
    np.savez_compressed(
        features_path,
        pids=np.array(pids, dtype=object),
        participation=part_mat.astype(np.float64),
        sequence_2grams=seq2_mat.astype(np.float64),
        timing=time_mat.astype(np.int32),
    )

    # Save sequences as JSON (lists of strings are not numpy-native)
    sequences_path = feature_dir / "sequences.json"
    with open(sequences_path, "w") as f:
        json.dump(seqs, f)

    # Save manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    return ScheduleFeatures(
        pids=pids,
        participation=part_mat,
        sequences=seqs,
        sequence_2grams=seq2_mat,
        timing=time_mat,
        manifest=manifest,
    )


def load_schedule_features(feature_dir: str | Path) -> ScheduleFeatures:
    """Load precomputed features from a feature store directory.

    Parameters
    ----------
    feature_dir : str | Path
        Directory containing manifest.json, features.npz, sequences.json.

    Returns
    -------
    ScheduleFeatures
        Loaded features and manifest.

    Raises
    ------
    FileNotFoundError
        If manifest or feature files are missing.
    """
    feature_dir = Path(feature_dir)

    # Load manifest
    manifest_path = feature_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest_dict = json.load(f)
    manifest = FeatureManifest.from_dict(manifest_dict)

    # Load features
    features_path = feature_dir / "features.npz"
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found at {features_path}")

    data = np.load(features_path, allow_pickle=True)
    pids = list(data["pids"])
    participation = data["participation"].astype(np.float64)
    timing = data["timing"].astype(np.int32)

    # Load sequences
    sequences_path = feature_dir / "sequences.json"
    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences not found at {sequences_path}")

    with open(sequences_path, "r") as f:
        sequences = json.load(f)

    if "sequence_2grams" in data:
        sequence_2grams = data["sequence_2grams"].astype(np.float64)
    else:
        sequence_2grams = sequence_2gram_matrix_from_sequences(sequences)

    return ScheduleFeatures(
        pids=pids,
        participation=participation,
        sequences=sequences,
        sequence_2grams=sequence_2grams,
        timing=timing,
        manifest=manifest,
    )


def feature_manifest_hash(manifest: FeatureManifest) -> str:
    """Compute a deterministic hash of a feature manifest.

    Used to validate distance cache compatibility.
    """
    manifest_str = json.dumps(manifest.to_dict(), sort_keys=True)
    return hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
