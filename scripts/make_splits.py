"""Generate reproducible stratified train/val/test splits from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import generate_stratified_splits, read_manifest, save_split_artifacts, summarize_counts
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to the canonical manifest.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for generated artifacts.",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level. Example: INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("make_splits", args.artifacts_dir, args.log_level)

    logger.info("Starting split generation.")
    logger.info("Resolved manifest path: %s", args.manifest.expanduser().resolve())
    logger.info("Split seed: %s", args.split_seed)
    logger.info("Log file: %s", log_path)

    records = read_manifest(args.manifest)
    logger.info("Loaded %s manifest rows.", len(records))
    logger.info("Category distribution before split: %s", summarize_counts(records, "category"))

    splits = generate_stratified_splits(records, seed=args.split_seed)
    saved_paths = save_split_artifacts(splits, args.artifacts_dir)

    for split_name, split_records in splits.items():
        logger.info(
            "%s split size: %s | category distribution: %s",
            split_name,
            len(split_records),
            summarize_counts(split_records, "category"),
        )
        logger.info("Saved %s split to %s", split_name, saved_paths[split_name])


if __name__ == "__main__":
    main()
