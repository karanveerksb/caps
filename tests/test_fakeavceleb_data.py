"""Unit tests for FakeAVCeleb metadata parsing and split generation."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from src.data.fakeavceleb import (
    FakeAVCelebDataset,
    generate_stratified_splits,
    load_fakeavceleb_metadata,
    normalize_headers,
    normalize_label,
    read_manifest,
    resolve_video_path,
    save_records_to_csv,
)


def _write_metadata_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "source,target1,target2,method,category,type,race,gender,path,",
                "id0001,-,-,real,A,RealVideo-RealAudio,African,men,00001.mp4,FakeAVCeleb/RealVideo-RealAudio/African/men/id0001",
                "id0002,id1000,-,wav2lip,B,RealVideo-FakeAudio,Asian,women,00002.mp4,FakeAVCeleb/RealVideo-FakeAudio/Asian/women/id0002",
                "id0003,id1001,-,fsgan,C,FakeVideo-RealAudio,Caucasian,men,00003.mp4,FakeAVCeleb/FakeVideo-RealAudio/Caucasian/men/id0003",
                "id0004,id1002,id2002,faceswap,D,FakeVideo-FakeAudio,Caucasian,women,00004.mp4,FakeAVCeleb/FakeVideo-FakeAudio/Caucasian/women/id0004",
            ]
        ),
        encoding="utf-8",
    )


def _make_split_records() -> list[dict[str, str]]:
    records = []
    categories = ["RARV", "RAFV", "FARV", "FAFV"]
    for category_index, category in enumerate(categories):
        for item_index in range(10):
            records.append(
                {
                    "sample_index": str(category_index * 10 + item_index),
                    "source_id": f"id{category_index:02d}{item_index:02d}",
                    "target1": "-",
                    "target2": "-",
                    "method": "real",
                    "category_code": chr(ord("A") + category_index),
                    "type_name": {
                        "RARV": "RealVideo-RealAudio",
                        "RAFV": "RealVideo-FakeAudio",
                        "FARV": "FakeVideo-RealAudio",
                        "FAFV": "FakeVideo-FakeAudio",
                    }[category],
                    "category": category,
                    "binary_label": "0" if category == "RARV" else "1",
                    "manipulation_target": {
                        "RARV": "none",
                        "RAFV": "audio",
                        "FARV": "visual",
                        "FAFV": "both",
                    }[category],
                    "race": "African",
                    "gender": "men",
                    "file_name": f"{item_index:05d}.mp4",
                    "relative_dir": f"FakeAVCeleb/{category}/id{category_index:02d}",
                    "video_path": f"/tmp/{category}_{item_index:02d}.mp4",
                    "dataset_split": "",
                }
            )
    return records


def test_normalize_headers_renames_trailing_blank_column() -> None:
    assert normalize_headers(["source", "path", ""]) == ["source", "path", "relative_dir"]


def test_normalize_label_maps_expected_categories() -> None:
    assert normalize_label("A", "RealVideo-RealAudio") == ("RARV", 0, "none")
    assert normalize_label("B", "RealVideo-FakeAudio") == ("RAFV", 1, "audio")
    assert normalize_label("C", "FakeVideo-RealAudio") == ("FARV", 1, "visual")
    assert normalize_label("D", "FakeVideo-FakeAudio") == ("FAFV", 1, "both")


def test_resolve_video_path_uses_metadata_parent() -> None:
    metadata_path = Path("/data/FakeAVCeleb_v1.2/meta_data.csv")
    resolved = resolve_video_path(
        metadata_path,
        "FakeAVCeleb/RealVideo-RealAudio/African/men/id0001",
        "00001.mp4",
    )
    assert resolved == Path(
        "/data/FakeAVCeleb_v1.2/FakeAVCeleb/RealVideo-RealAudio/African/men/id0001/00001.mp4"
    )


def test_resolve_video_path_prefers_dataset_root_when_provided() -> None:
    metadata_path = Path("/data/FakeAVCeleb_v1.2/meta_data.csv")
    resolved = resolve_video_path(
        metadata_path,
        "FakeAVCeleb/RealVideo-RealAudio/African/men/id0001",
        "00001.mp4",
        dataset_root="/workspace/try/dataset",
    )
    assert resolved == Path(
        "/workspace/try/dataset/FakeAVCeleb/RealVideo-RealAudio/African/men/id0001/00001.mp4"
    )


def test_load_metadata_parses_expected_fields(tmp_path: Path) -> None:
    metadata_path = tmp_path / "meta_data.csv"
    _write_metadata_csv(metadata_path)

    records = load_fakeavceleb_metadata(metadata_path)

    assert len(records) == 4
    assert records[0]["relative_dir"] == "FakeAVCeleb/RealVideo-RealAudio/African/men/id0001"
    assert records[1]["category"] == "RAFV"
    assert records[2]["binary_label"] == "1"
    assert records[3]["manipulation_target"] == "both"


def test_load_metadata_uses_dataset_root_when_supplied(tmp_path: Path) -> None:
    metadata_path = tmp_path / "meta_data.csv"
    _write_metadata_csv(metadata_path)

    records = load_fakeavceleb_metadata(metadata_path, dataset_root="/workspace/try/dataset")

    assert records[0]["video_path"] == (
        "/workspace/try/dataset/FakeAVCeleb/RealVideo-RealAudio/African/men/id0001/00001.mp4"
    )


def test_generate_stratified_splits_preserves_all_rows() -> None:
    records = _make_split_records()

    splits = generate_stratified_splits(records, seed=123)
    combined = splits["train"] + splits["val"] + splits["test"]

    assert len(combined) == len(records)
    assert sorted(record["sample_index"] for record in combined) == sorted(
        record["sample_index"] for record in records
    )
    assert len(splits["train"]) == 28
    assert len(splits["val"]) == 4
    assert len(splits["test"]) == 8


def test_dataset_and_dataloader_return_metadata_batch(tmp_path: Path) -> None:
    all_records = _make_split_records()
    records = [all_records[0], all_records[10], all_records[20], all_records[30]]
    for record in records:
        video_path = Path(record["video_path"])
        video_path.write_text("placeholder", encoding="utf-8")

    split_csv = tmp_path / "train.csv"
    save_records_to_csv(records, split_csv)

    dataset = FakeAVCelebDataset(split_csv, strict=True)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert len(dataset) == 4
    assert sorted(batch.keys()) == [
        "category",
        "label",
        "manipulation_target",
        "source_id",
        "video_path",
    ]
    assert batch["label"].tolist() == [0, 1]


def test_saved_manifest_round_trip(tmp_path: Path) -> None:
    records = _make_split_records()[:3]
    manifest_path = tmp_path / "manifest.csv"
    save_records_to_csv(records, manifest_path)

    loaded = read_manifest(manifest_path)

    assert loaded[0]["source_id"] == records[0]["source_id"]
    assert loaded[-1]["video_path"] == records[-1]["video_path"]
