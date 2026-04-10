"""Data loading and manifest helpers for FakeAVCeleb."""

from .fakeavceleb import (
    FakeAVCelebDataset,
    generate_stratified_splits,
    load_fakeavceleb_metadata,
    read_manifest,
    save_records_to_csv,
    save_split_artifacts,
    summarize_counts,
    validate_video_paths,
)

__all__ = [
    "FakeAVCelebDataset",
    "generate_stratified_splits",
    "load_fakeavceleb_metadata",
    "read_manifest",
    "save_records_to_csv",
    "save_split_artifacts",
    "summarize_counts",
    "validate_video_paths",
]
