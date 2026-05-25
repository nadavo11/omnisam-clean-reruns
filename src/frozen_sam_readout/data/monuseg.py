"""MoNuSeg binary benchmark adapter backed by the Hugging Face mirror.

Extracted from rwtd_sam3.data.monuseg_binary. Uses the official AutoSAM-comparable
challenge split:

- training uses only the 30 official challenge images (excludes the 7 extra
  Hugging Face ``tissue == 0`` train rows)
- testing uses the 14 official challenge test images
- each binary foreground mask is the union of all annotated nucleus instances
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .common import boundary_from_region_masks


MONUSEG_DATASET_ID = "monuseg"
MONUSEG_HF_DATASET_NAME = "RationAI/MoNuSeg"
MONUSEG_SUPPORTED_SPLITS = ("train", "test", "all")
MONUSEG_TISSUE_LABELS: dict[int, str] = {
    0: "unknown",
    1: "breast",
    2: "kidney",
    3: "liver",
    4: "prostate",
    5: "bladder",
    6: "colon",
    7: "stomach",
}


@dataclass(frozen=True)
class MonusegBinaryOverview:
    dataset_id: str
    dataset_name: str
    split: str
    num_examples: int
    patient_ids: tuple[str, ...]
    evaluation_view: str
    train_selection_policy: str
    cache_dir: Optional[str]


@dataclass(frozen=True)
class MonusegBinarySample:
    index: int
    dataset_id: str
    split: str
    crop_name: str
    image: Any
    boundary_mask: np.ndarray
    texture_a_mask: np.ndarray  # nucleus
    texture_b_mask: np.ndarray  # background
    texture_a: str
    texture_b: str
    evaluation_view: str
    grade_label: Optional[str]

    @property
    def height(self) -> int:
        return int(self.image.size[1])

    @property
    def width(self) -> int:
        return int(self.image.size[0])


@lru_cache(maxsize=8)
def _load_monuseg_dataset(dataset_name: str, cache_dir: Optional[str]):
    from datasets import load_dataset

    return load_dataset(dataset_name, cache_dir=cache_dir)


def load_monuseg_binary_overview(
    *,
    split: str = "test",
    dataset_name: str = MONUSEG_HF_DATASET_NAME,
    cache_dir: Optional[str] = None,
) -> MonusegBinaryOverview:
    records = _resolve_monuseg_records(
        split=split, dataset_name=dataset_name, cache_dir=cache_dir
    )
    return MonusegBinaryOverview(
        dataset_id=MONUSEG_DATASET_ID,
        dataset_name=dataset_name,
        split=split,
        num_examples=len(records),
        patient_ids=tuple(str(record["patient"]) for record in records),
        evaluation_view="direct_foreground",
        train_selection_policy="official_challenge_train_excludes_tissue_0_unknown",
        cache_dir=cache_dir,
    )


def iter_monuseg_binary_samples(
    *,
    split: str = "test",
    dataset_name: str = MONUSEG_HF_DATASET_NAME,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    start_index: int = 0,
) -> Iterator[MonusegBinarySample]:
    records = _resolve_monuseg_records(
        split=split, dataset_name=dataset_name, cache_dir=cache_dir
    )
    max_items = (
        max(0, len(records) - start_index)
        if limit is None
        else min(int(limit), max(0, len(records) - start_index))
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_monuseg_binary_sample(record=records[index], split_index=index)


def _resolve_monuseg_records(
    *, split: str, dataset_name: str, cache_dir: Optional[str]
) -> tuple[dict[str, Any], ...]:
    if split not in MONUSEG_SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported MoNuSeg split '{split}'. Expected one of: {MONUSEG_SUPPORTED_SPLITS}"
        )
    dataset = _load_monuseg_dataset(dataset_name, cache_dir)
    train_split = dataset["train"]
    test_split = dataset["test"]
    train_patients = list(train_split["patient"])
    train_tissues = list(train_split["tissue"])
    test_patients = list(test_split["patient"])
    train_records = tuple(
        {
            "split": "train",
            "row_index": int(i),
            "patient": str(train_patients[i]),
            "tissue": int(train_tissues[i]),
            "dataset_name": str(dataset_name),
            "cache_dir": cache_dir,
        }
        for i in range(len(train_patients))
        if int(train_tissues[i]) != 0
    )
    test_records = tuple(
        {
            "split": "test",
            "row_index": int(i),
            "patient": str(test_patients[i]),
            "tissue": 0,
            "dataset_name": str(dataset_name),
            "cache_dir": cache_dir,
        }
        for i in range(len(test_patients))
    )
    if split == "train":
        return train_records
    if split == "test":
        return test_records
    return train_records + test_records


def _decode_monuseg_binary_sample(*, record: dict[str, Any], split_index: int) -> MonusegBinarySample:
    dataset = _load_monuseg_dataset(str(record["dataset_name"]), record.get("cache_dir"))
    row = dataset[str(record["split"])][int(record["row_index"])]
    image = row["image"].convert("RGB")
    h, w = int(image.size[1]), int(image.size[0])
    nucleus_mask = np.zeros((h, w), dtype=bool)
    for instance in row["instances"]:
        nucleus_mask |= np.asarray(instance, dtype=np.uint8) > 0
    background_mask = np.logical_not(nucleus_mask)
    if not nucleus_mask.any():
        raise ValueError(
            f"MoNuSeg sample '{row['patient']}' has no positive nucleus pixels."
        )
    boundary_mask = boundary_from_region_masks(nucleus_mask, background_mask)
    tissue_index = int(row["tissue"])
    tissue_label = MONUSEG_TISSUE_LABELS.get(tissue_index, f"tissue_{tissue_index}")
    return MonusegBinarySample(
        index=int(split_index),
        dataset_id=MONUSEG_DATASET_ID,
        split=str(record["split"]),
        crop_name=str(row["patient"]),
        image=image,
        boundary_mask=np.asarray(boundary_mask, dtype=bool),
        texture_a_mask=np.asarray(nucleus_mask, dtype=bool),
        texture_b_mask=np.asarray(background_mask, dtype=bool),
        texture_a="nucleus",
        texture_b="background",
        evaluation_view="direct_foreground",
        grade_label=tissue_label,
    )
