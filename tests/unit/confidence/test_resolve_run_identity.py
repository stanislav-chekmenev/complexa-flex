"""Unit tests for `_resolve_run_identity`.

The resolver decides the WandB run id and display name for each launch:
- `cfg.resume_id` (when set) → reuse verbatim; keep `run_name` as display.
- Otherwise fork a fresh suffix appended to `run_name`. Suffix precedence:
  `COMPLEXA_RUN_FORK_ID` env var → `SLURM_JOB_ID` → `secrets.token_hex(4)`.

The suffix must be deterministic across DDP ranks for a given launch — this
is what `SLURM_JOB_ID` (or an explicit env override) guarantees.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf


def test_resume_id_returned_verbatim_with_base_run_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPLEXA_RUN_FORK_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    from proteinfoundation.confidence.train_confidence import _resolve_run_identity

    cfg = OmegaConf.create({"run_name": "pae-distill-teddymer", "resume_id": "existing-run-xyz"})
    wandb_id, wandb_name = _resolve_run_identity(cfg)

    assert wandb_id == "existing-run-xyz"
    assert wandb_name == "pae-distill-teddymer"


def test_fork_id_env_takes_precedence_over_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPLEXA_RUN_FORK_ID", "deadbeef")
    monkeypatch.setenv("SLURM_JOB_ID", "99999")

    from proteinfoundation.confidence.train_confidence import _resolve_run_identity

    cfg = OmegaConf.create({"run_name": "pae-distill-teddymer", "resume_id": None})
    wandb_id, wandb_name = _resolve_run_identity(cfg)

    assert wandb_id == "pae-distill-teddymer-deadbeef"
    assert wandb_name == "pae-distill-teddymer-deadbeef"


def test_slurm_job_id_used_when_fork_id_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPLEXA_RUN_FORK_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "54152")

    from proteinfoundation.confidence.train_confidence import _resolve_run_identity

    cfg = OmegaConf.create({"run_name": "pae-distill-teddymer", "resume_id": None})
    wandb_id, wandb_name = _resolve_run_identity(cfg)

    assert wandb_id == "pae-distill-teddymer-54152"
    assert wandb_name == "pae-distill-teddymer-54152"


def test_token_fallback_when_no_env_or_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPLEXA_RUN_FORK_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    from proteinfoundation.confidence.train_confidence import _resolve_run_identity

    cfg = OmegaConf.create({"run_name": "pae-distill-teddymer", "resume_id": None})
    wandb_id, wandb_name = _resolve_run_identity(cfg)

    assert wandb_id == wandb_name
    assert wandb_id.startswith("pae-distill-teddymer-")
    # secrets.token_hex(4) produces 8 hex chars
    suffix = wandb_id[len("pae-distill-teddymer-"):]
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


def test_missing_resume_id_key_is_treated_as_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPLEXA_RUN_FORK_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")

    from proteinfoundation.confidence.train_confidence import _resolve_run_identity

    # No `resume_id` key at all (use .get default).
    cfg = OmegaConf.create({"run_name": "pae-distill-teddymer"})
    wandb_id, wandb_name = _resolve_run_identity(cfg)

    assert wandb_id == "pae-distill-teddymer-12345"
    assert wandb_name == "pae-distill-teddymer-12345"
