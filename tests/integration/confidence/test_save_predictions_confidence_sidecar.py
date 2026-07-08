"""``save_predictions`` writes confidence columns + a sidecar CSV (CPU-only).

Confirms the per-sample rewards CSV carries the confidence scalars (via the
existing rewards-dict loop) and that a standalone
``confidence_scores_{job_id}.csv`` is written next to the rewards CSV (under
``root_path/..``), keyed by ``metadata_tag``.
"""

from __future__ import annotations

import os

import pandas as pd
import torch

from proteinfoundation.generate import save_predictions
from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY


def _batch_pred(b: int = 2, n: int = 6) -> dict:
    coors = torch.randn(b, n, 37, 3) * 3.0
    ipae = torch.tensor([5.0, 8.0])[:b]
    return {
        "coors": coors,
        "residue_type": torch.randint(0, 20, (b, n)),
        "chain_index": torch.zeros(b, n, dtype=torch.long),
        "mask": torch.ones(b, n, dtype=torch.bool),
        "metadata_tag": [f"bon_orig0_r{r}" for r in range(b)],
        "sample_type": ["final"] * b,
        "rewards": {
            TOTAL_REWARD_KEY: -ipae,
            "confidence_ipae": ipae,
            "confidence_complex_plddt": torch.tensor([0.92, 0.7])[:b],
            "provisional_success": torch.tensor([1.0, 0.0])[:b],
            "elapsed_gpu_hours": torch.tensor([0.01, 0.02])[:b],
        },
    }


def test_save_predictions_writes_confidence_columns_and_sidecar(tmp_path) -> None:
    root = tmp_path / "inf_0"
    root.mkdir()
    job_id = 7

    _pdb_paths, df = save_predictions(str(root), [_batch_pred()], job_id=job_id)

    for col in (
        "total_reward",
        "confidence_ipae",
        "confidence_complex_plddt",
        "provisional_success",
        "elapsed_gpu_hours",
        "metadata_tag",
    ):
        assert col in df.columns, f"missing rewards CSV column {col}"
    assert (df["total_reward"].values == (-df["confidence_ipae"]).values).all()

    sidecar = os.path.join(str(root), "..", f"confidence_scores_{job_id}.csv")
    assert os.path.exists(sidecar), "sidecar CSV not written"
    sc = pd.read_csv(sidecar)
    assert list(sc["metadata_tag"]) == ["bon_orig0_r0", "bon_orig0_r1"]
    for col in (
        "confidence_ipae",
        "confidence_complex_plddt",
        "provisional_success",
        "elapsed_gpu_hours",
    ):
        assert col in sc.columns, f"missing sidecar column {col}"
