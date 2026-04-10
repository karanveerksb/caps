"""Smoke-test the metadata-backed FakeAVCeleb DataLoader."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from src.data import FakeAVCelebDataset
from src.utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, required=True, help="Path to a split CSV file.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for generated artifacts.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the smoke test.")
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
    logger, log_path = setup_logger("check_dataloader", args.artifacts_dir, args.log_level)

    logger.info("Starting DataLoader smoke test.")
    logger.info("Resolved split path: %s", args.split_csv.expanduser().resolve())
    logger.info("Batch size: %s", args.batch_size)
    logger.info("Log file: %s", log_path)

    dataset = FakeAVCelebDataset(args.split_csv, strict=args.strict_path_check, logger=logger)
    if len(dataset) == 0:
        raise ValueError("Split CSV contains no rows.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(dataloader))
    batch_size = len(batch["video_path"])

    logger.info("Loaded batch size: %s", batch_size)
    logger.info("Batch keys: %s", sorted(batch.keys()))


if __name__ == "__main__":
    main()
