"""Metadata parsing, manifest generation, and dataset utilities for FakeAVCeleb."""

from __future__ import annotations

import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Sequence

from torch.utils.data import Dataset

from src.config import DEFAULT_SPLIT_CONFIG, SplitConfig

RAW_TO_NORMALIZED_CATEGORY = {
    ("A", "RealVideo-RealAudio"): ("RARV", 0, "none"),
    ("B", "RealVideo-FakeAudio"): ("RAFV", 1, "audio"),
    ("C", "FakeVideo-RealAudio"): ("FARV", 1, "visual"),
    ("D", "FakeVideo-FakeAudio"): ("FAFV", 1, "both"),
}

MANIFEST_FIELDNAMES = [
    "sample_index",
    "source_id",
    "target1",
    "target2",
    "method",
    "category_code",
    "type_name",
    "category",
    "binary_label",
    "manipulation_target",
    "race",
    "gender",
    "file_name",
    "relative_dir",
    "video_path",
    "dataset_split",
]


def normalize_headers(fieldnames: Sequence[str] | None) -> List[str]:
    """Rename the trailing empty header into a stable column name."""

    if not fieldnames:
        raise ValueError("Metadata CSV has no headers.")

    normalized = []
    for name in fieldnames:
        clean_name = (name or "").strip()
        normalized.append(clean_name if clean_name else "relative_dir")
    return normalized


def _clean_row(raw_row: MutableMapping[str, str]) -> Dict[str, str]:
    return {str(key).strip(): (value or "").strip() for key, value in raw_row.items()}


def normalize_label(category_code: str, type_name: str) -> tuple[str, int, str]:
    """Map FakeAVCeleb raw labels into project-level labels."""

    key = (category_code.strip(), type_name.strip())
    if key not in RAW_TO_NORMALIZED_CATEGORY:
        raise ValueError(
            f"Unsupported category/type combination: category={category_code}, type={type_name}"
        )
    return RAW_TO_NORMALIZED_CATEGORY[key]


def _relative_dir_variants(relative_dir: str) -> List[Path]:
    """Generate plausible relative directory variants seen across FakeAVCeleb layouts."""

    base = Path(relative_dir)
    variants = [base]

    if base.parts and base.parts[0] == "FakeAVCeleb" and len(base.parts) > 1:
        variants.append(Path(*base.parts[1:]))

    deduped: List[Path] = []
    seen = set()
    for variant in variants:
        key = variant.as_posix()
        if key not in seen:
            deduped.append(variant)
            seen.add(key)
    return deduped


def resolve_video_path(
    metadata_path: str | Path,
    relative_dir: str,
    file_name: str,
    dataset_root: str | Path | None = None,
) -> Path:
    """Resolve the sample video path across common FakeAVCeleb directory layouts."""

    metadata_parent = Path(metadata_path).expanduser().resolve().parent
    anchors: List[Path] = []

    if dataset_root is not None:
        anchors.append(Path(dataset_root).expanduser().resolve())
    anchors.append(metadata_parent)

    candidates: List[Path] = []
    for anchor in anchors:
        for relative_variant in _relative_dir_variants(relative_dir):
            candidate = anchor / relative_variant / file_name
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Deterministic fallback when the underlying files are not present locally.
    return candidates[0]


def _normalize_record(
    row: Dict[str, str],
    metadata_path: str | Path,
    sample_index: int,
    dataset_root: str | Path | None = None,
) -> Dict[str, str]:
    required_keys = [
        "source",
        "target1",
        "target2",
        "method",
        "category",
        "type",
        "race",
        "gender",
        "path",
        "relative_dir",
    ]
    missing = [key for key in required_keys if key not in row]
    if missing:
        raise ValueError(f"Metadata row is missing columns: {missing}")

    normalized_category, binary_label, manipulation_target = normalize_label(
        row["category"], row["type"]
    )
    video_path = resolve_video_path(
        metadata_path,
        row["relative_dir"],
        row["path"],
        dataset_root=dataset_root,
    )

    return {
        "sample_index": str(sample_index),
        "source_id": row["source"],
        "target1": row["target1"],
        "target2": row["target2"],
        "method": row["method"],
        "category_code": row["category"],
        "type_name": row["type"],
        "category": normalized_category,
        "binary_label": str(binary_label),
        "manipulation_target": manipulation_target,
        "race": row["race"],
        "gender": row["gender"],
        "file_name": row["path"],
        "relative_dir": row["relative_dir"],
        "video_path": str(video_path),
        "dataset_split": "",
    }


def load_fakeavceleb_metadata(
    metadata_path: str | Path,
    dataset_root: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> List[Dict[str, str]]:
    """Parse FakeAVCeleb metadata CSV into a canonical manifest-like structure."""

    metadata_path = Path(metadata_path).expanduser().resolve()
    records: List[Dict[str, str]] = []
    malformed_rows = 0
    raw_category_counts: Counter[str] = Counter()
    raw_type_counts: Counter[str] = Counter()

    with metadata_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = normalize_headers(reader.fieldnames)

        for sample_index, raw_row in enumerate(reader):
            row = _clean_row(raw_row)
            if not any(row.values()):
                continue
            raw_category_counts[row.get("category", "")] += 1
            raw_type_counts[row.get("type", "")] += 1
            try:
                record = _normalize_record(
                    row,
                    metadata_path,
                    sample_index,
                    dataset_root=dataset_root,
                )
            except ValueError as exc:
                malformed_rows += 1
                if logger:
                    logger.warning("Skipping malformed metadata row %s: %s", sample_index, exc)
                continue
            records.append(record)

    if logger:
        logger.info("Parsed %s metadata rows from %s", len(records), metadata_path)
        logger.info("Malformed rows skipped: %s", malformed_rows)
        logger.info("Raw category counts: %s", dict(sorted(raw_category_counts.items())))
        logger.info("Raw type counts: %s", dict(sorted(raw_type_counts.items())))
        logger.info("Normalized category counts: %s", summarize_counts(records, "category"))
    return records


def summarize_counts(records: Iterable[MutableMapping[str, str]], field: str) -> Dict[str, int]:
    """Return ordered counts for a selected record field."""

    counts = Counter(record[field] for record in records)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def validate_video_paths(
    records: Iterable[MutableMapping[str, str]],
    logger: logging.Logger | None = None,
) -> List[str]:
    """Collect missing video paths for diagnostics."""

    missing = []
    for record in records:
        video_path = Path(record["video_path"])
        if not video_path.exists():
            missing.append(str(video_path))

    if logger:
        if missing:
            logger.warning("Detected %s missing video paths.", len(missing))
        else:
            logger.info("All video paths resolved successfully.")
    return missing


def save_records_to_csv(
    records: Iterable[MutableMapping[str, str]],
    output_path: str | Path,
    fieldnames: Sequence[str] = MANIFEST_FIELDNAMES,
) -> Path:
    """Persist manifest-style records to CSV."""

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})

    return output_path


def read_manifest(manifest_path: str | Path) -> List[Dict[str, str]]:
    """Read a manifest or split CSV back into memory."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_clean_row(row) for row in reader]


def _split_counts(total: int, config: SplitConfig) -> tuple[int, int, int]:
    train_count = int(total * config.train_ratio)
    val_count = int(total * config.val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def generate_stratified_splits(
    records: Sequence[MutableMapping[str, str]],
    split_config: SplitConfig = DEFAULT_SPLIT_CONFIG,
    seed: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    """Create row-level stratified splits by normalized category."""

    split_config.validate()
    rng = random.Random(seed)
    grouped_records: Dict[str, List[MutableMapping[str, str]]] = defaultdict(list)

    for record in records:
        grouped_records[record["category"]].append(record)

    splits: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}

    for category, items in grouped_records.items():
        shuffled_items = list(items)
        rng.shuffle(shuffled_items)

        train_count, val_count, _ = _split_counts(len(shuffled_items), split_config)

        for index, item in enumerate(shuffled_items):
            if index < train_count:
                split_name = "train"
            elif index < train_count + val_count:
                split_name = "val"
            else:
                split_name = "test"

            record_copy = dict(item)
            record_copy["dataset_split"] = split_name
            splits[split_name].append(record_copy)

    for split_records in splits.values():
        rng.shuffle(split_records)

    return splits


def save_split_artifacts(
    splits: Dict[str, Sequence[MutableMapping[str, str]]],
    artifacts_dir: str | Path,
) -> Dict[str, Path]:
    """Save train, validation, and test split CSVs."""

    artifacts_path = Path(artifacts_dir).expanduser().resolve()
    split_dir = artifacts_path / "splits"

    saved_paths = {}
    for split_name, records in splits.items():
        saved_paths[split_name] = save_records_to_csv(records, split_dir / f"{split_name}.csv")
    return saved_paths


class FakeAVCelebDataset(Dataset):
    """Metadata-backed dataset scaffold for future multimodal loading."""

    def __init__(
        self,
        split_csv: str | Path,
        strict: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.split_csv = Path(split_csv).expanduser().resolve()
        self.records = read_manifest(self.split_csv)
        self.logger = logger

        missing_paths = validate_video_paths(self.records, logger=logger)
        if missing_paths and strict:
            raise FileNotFoundError(
                f"{len(missing_paths)} video paths listed in {self.split_csv} do not exist."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        return {
            "video_path": record["video_path"],
            "label": int(record["binary_label"]),
            "category": record["category"],
            "manipulation_target": record["manipulation_target"],
            "source_id": record["source_id"],
        }
