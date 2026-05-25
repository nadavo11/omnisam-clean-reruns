"""Dataset registry mapping dataset names to streaming iterators."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator

from .glas import iter_glas_binary_samples
from .monuseg import iter_monuseg_binary_samples


DATASET_REGISTRY: Dict[str, Callable[..., Iterator[Any]]] = {
    "monuseg": iter_monuseg_binary_samples,
    "glas": iter_glas_binary_samples,
}


def build_dataset_iter(name: str, **kwargs) -> Iterator[Any]:
    """Build a streaming sample iterator for a registered dataset name."""

    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
