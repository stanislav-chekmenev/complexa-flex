"""Test-only `StructureDataset` shim that bypasses CIF parsing.

The shim hands back a saved `Data` fixture verbatim, then applies the
configured `atom37_transforms` so the integration test exercises the real
pLDDT transform end-to-end without depending on AF2-DB files on disk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset

from proteinfoundation.datasets.transforms import Data


class FixtureStructureDataset(Dataset):
    """Loads a single saved `Data` fixture and applies atom37 transforms.

    Mirrors the public surface of `StructureDataset` used by the data module's
    collate path: `__len__`, `__getitem__`, and the side-effect of running the
    atom37 transform list. The atomarray pipeline is intentionally skipped.
    """

    def __init__(
        self,
        fixture_path: str | Path,
        atom37_transforms: list[Callable] | None = None,
        length: int = 1,
    ):
        self._fixture: Data = torch.load(fixture_path, weights_only=False)
        self.atom37_transforms = atom37_transforms or []
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Data:
        data = _clone_data(self._fixture)
        for transform in self.atom37_transforms:
            data = transform(data)
        return data


def _clone_data(data: Data) -> Data:
    new = Data()
    for key in data.keys():
        value = getattr(data, key)
        if isinstance(value, torch.Tensor):
            setattr(new, key, value.clone())
        else:
            setattr(new, key, value)
    return new
