"""Build a canonical FakeAVCeleb manifest from the provided metadata CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import (
    load_fakeavceleb_metadata,
    save_records_to_csv,
    summarize_counts,
    validate_video_paths,
)
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root path.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="Path to meta_data.csv.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--strict-path-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if any resolved video path is missing.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level. Example: INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("build_manifest", args.artifacts_dir, args.log_level)

    logger.info("Starting manifest build.")
    logger.info("Resolved dataset root: %s", args.dataset_root.resolve() if args.dataset_root else "N/A")
    logger.info("Resolved metadata path: %s", args.metadata_path.expanduser().resolve())
    logger.info("Log file: %s", log_path)

    records = load_fakeavceleb_metadata(
        args.metadata_path,
        dataset_root=args.dataset_root,
        logger=logger,
    )
    logger.info("Manifest row count: %s", len(records))
    logger.info("Manifest category counts: %s", summarize_counts(records, "category"))

    missing_paths = validate_video_paths(records, logger=logger)
    if args.strict_path_check and missing_paths:
        raise FileNotFoundError(
            f"Strict path check failed. Missing {len(missing_paths)} video files."
        )

    manifest_path = args.artifacts_dir.expanduser().resolve() / "manifests" / "full_manifest.csv"
    save_records_to_csv(records, manifest_path)
    logger.info("Saved manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
