"""Local GlaS binary benchmark adapter.

Extracted from rwtd_sam3.data.glas_binary. Resolves an AutoSAM-style flat
directory layout (or an extracted Warwick archive containing one).
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

from .common import boundary_from_region_masks


LOGGER = logging.getLogger(__name__)

GLAS_DATASET_ID = "glas"
SUPPORTED_IMAGE_SUFFIXES = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
GLAS_SUPPORTED_SPLITS = ("train", "test", "all")
_IMAGE_STEM_PATTERN = re.compile(r"^(train|testA|testB)_(\d+)$")
_MASK_STEM_PATTERN = re.compile(r"^(train|testA|testB)_(\d+)_anno$")


@dataclass(frozen=True)
class GlasBinaryOverview:
    dataset_id: str
    dataset_root: Path
    data_dir: Path
    split: str
    num_examples: int
    image_ids: tuple[str, ...]
    evaluation_view: str
    grade_csv_path: Optional[Path]


@dataclass(frozen=True)
class GlasBinarySample:
    index: int
    dataset_id: str
    split: str
    crop_name: str
    image: Image.Image
    boundary_mask: np.ndarray
    texture_a_mask: np.ndarray  # gland
    texture_b_mask: np.ndarray  # background
    texture_a: str
    texture_b: str
    evaluation_view: str
    grade_label: Optional[str]

    @property
    def height(self) -> int:
        return self.image.size[1]

    @property
    def width(self) -> int:
        return self.image.size[0]


def load_glas_binary_overview(dataset_root, split: str = "test") -> GlasBinaryOverview:
    if split not in GLAS_SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported GlaS split '{split}'. Expected: {GLAS_SUPPORTED_SPLITS}"
        )
    resolved_root, data_dir = resolve_glas_binary_dirs(dataset_root)
    image_ids = _discover_image_ids(data_dir=data_dir, split=split)
    grade_csv_path = _resolve_grade_csv_path(resolved_root, data_dir)
    return GlasBinaryOverview(
        dataset_id=GLAS_DATASET_ID,
        dataset_root=resolved_root,
        data_dir=data_dir,
        split=split,
        num_examples=len(image_ids),
        image_ids=image_ids,
        evaluation_view="direct_foreground",
        grade_csv_path=grade_csv_path,
    )


def iter_glas_binary_samples(
    dataset_root,
    *,
    split: str = "test",
    limit: Optional[int] = None,
    start_index: int = 0,
) -> Iterator[GlasBinarySample]:
    overview = load_glas_binary_overview(dataset_root, split=split)
    max_items = (
        max(0, overview.num_examples - start_index)
        if limit is None
        else min(limit, max(0, overview.num_examples - start_index))
    )
    for index in range(start_index, start_index + max_items):
        yield _decode_glas_sample(overview, index)


def resolve_glas_binary_dirs(dataset_root) -> tuple[Path, Path]:
    requested_root = Path(dataset_root).expanduser().resolve()
    if not requested_root.is_dir():
        raise FileNotFoundError(f"GlaS root not found: {requested_root}")
    candidates = [requested_root] + [p for p in requested_root.rglob("*") if p.is_dir()]
    best_dir, best_count = None, 0
    for cand in candidates:
        c = _count_matched_pairs(cand)
        if c > best_count:
            best_dir, best_count = cand, c
    if best_dir is None or best_count < 1:
        raise FileNotFoundError(
            f"No matched GlaS image/mask pairs found under '{requested_root}'."
        )
    return requested_root, best_dir


def _decode_glas_sample(overview: GlasBinaryOverview, index: int) -> GlasBinarySample:
    if index < 0 or index >= overview.num_examples:
        raise IndexError(f"GlaS index {index} out of range ({overview.num_examples}).")
    image_id = overview.image_ids[index]
    image_path = _find_image_path(overview.data_dir, image_id)
    mask_path = overview.data_dir / f"{image_id}_anno.bmp"
    image = Image.open(image_path).convert("RGB")
    gland = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    background = ~gland
    boundary = boundary_from_region_masks(gland, background)
    return GlasBinarySample(
        index=index,
        dataset_id=overview.dataset_id,
        split="train" if image_id.startswith("train_") else "test",
        crop_name=image_id,
        image=image,
        boundary_mask=np.asarray(boundary, dtype=bool),
        texture_a_mask=np.asarray(gland, dtype=bool),
        texture_b_mask=np.asarray(background, dtype=bool),
        texture_a="gland",
        texture_b="background",
        evaluation_view=overview.evaluation_view,
        grade_label=_load_grade_label(overview.grade_csv_path, image_id),
    )


def _count_matched_pairs(data_dir: Path) -> int:
    if not data_dir.is_dir():
        return 0
    image_stems, mask_stems = set(), set()
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_IMAGE_SUFFIXES and _IMAGE_STEM_PATTERN.match(stem):
            image_stems.add(stem)
        elif suffix == ".bmp":
            m = _MASK_STEM_PATTERN.match(stem)
            if m:
                mask_stems.add(f"{m.group(1)}_{m.group(2)}")
    return len(image_stems & mask_stems)


def _discover_image_ids(*, data_dir: Path, split: str) -> tuple[str, ...]:
    image_ids = set()
    for path in data_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        m = _IMAGE_STEM_PATTERN.match(path.stem)
        if m is None:
            continue
        if not _split_matches_prefix(split=split, prefix=m.group(1)):
            continue
        if (data_dir / f"{path.stem}_anno.bmp").is_file():
            image_ids.add(path.stem)
    if not image_ids:
        raise FileNotFoundError(
            f"No matched GlaS pairs in '{data_dir}' for split '{split}'."
        )
    return tuple(sorted(image_ids, key=_natural_sort_key))


def _find_image_path(data_dir: Path, image_id: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        c = data_dir / f"{image_id}{suffix}"
        if c.is_file():
            return c
    raise FileNotFoundError(f"Image '{image_id}' not found in {data_dir}")


def _split_matches_prefix(*, split: str, prefix: str) -> bool:
    if split == "all":
        return prefix in {"train", "testA", "testB"}
    if split == "train":
        return prefix == "train"
    return prefix in {"testA", "testB"}


def _resolve_grade_csv_path(resolved_root: Path, data_dir: Path) -> Optional[Path]:
    for candidate in (data_dir / "Grade.csv", resolved_root / "Grade.csv"):
        if candidate.is_file():
            return candidate
    return None


def _load_grade_label(grade_csv_path: Optional[Path], image_id: str) -> Optional[str]:
    if grade_csv_path is None:
        return None
    try:
        with grade_csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if row and row[0].strip() == image_id and len(row) > 1:
                    return row[1].strip()
    except OSError:
        return None
    return None


def _natural_sort_key(value: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", value)]
