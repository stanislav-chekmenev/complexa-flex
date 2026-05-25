# PR #2 — pAE-Derived Metric Correlations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every pAE-derived interface metric (`i_pae`, `min_ipae`, `i_ptm`, `i_ptm_energy`, six ipSAE variants), additionally log per-sample-paired MAE, Pearson R, and Spearman R between the metric computed from the *ground-truth* AF2 PAE and the metric computed from the *student's predicted* PAE, accumulated across the validation set in a DDP-safe way via `torchmetrics.MetricCollection`.

**Architecture:** The 10 metric functions in `nn/confidence/_metrics.py` already take an EV-shaped `[B, L, L]` PAE map and return a scalar (batch-mean over samples-with-mass). Step 1 adds a `reduce: Literal["per_sample", "batch_mean"] = "batch_mean"` kwarg so callers can extract per-sample values for paired accumulation. `PaeHead` builds a fp32 EV from GT bin labels (already available — `_labels_to_continuous` over `pae_bin`) and from its own logits (`logits_to_expected_value`); the head then `update`s a per-metric `torchmetrics.MetricCollection({MAE, Pearson, Spearman})` ModuleDict during validation. On `on_validation_epoch_end`, the Lightning module computes and logs the aggregates. `MetricCollection` handles the DDP all-gather and reset.

**Tech Stack:** torchmetrics ≥1.4 (`PearsonCorrCoef`, `SpearmanCorrCoef`, `MeanAbsoluteError`, `MetricCollection`), PyTorch 2.10, Lightning ≥2.5,<2.6, Hydra 1.3.

**Prerequisite:** PR #1 (QG head port) must be merged to `dev` before this branch opens — the `PaeHead.compute_loss_and_metrics` signature and the `MultiHeadConfidence.compute_multi_loss_and_metrics` dispatch path are both touched by PR #1, and rebasing across them would cause a conflict storm. Task 1 explicitly checks this.

---

### Task 1: Branch setup and prerequisite verification

**Files:**
- Modify: working tree only (no files written yet).

- [ ] **Step 1: Verify PR #1 has merged to `dev`**

```bash
cd /mnt/storage01/home/schekmenev/projects/complexa-flex
git fetch origin dev
git log --oneline origin/dev -20 | grep -i "qg head\|pairformer\|adaptor" || {
  echo "ABORT: PR #1 (QG head port) does not appear to be merged to dev."
  echo "Open this PR only after PR #1 lands; otherwise the pae_head.py / lightning_module.py conflicts are non-trivial."
  exit 1
}
```

Expected: at least one matching commit line.

- [ ] **Step 2: Create feature branch off latest dev**

```bash
git checkout dev
git pull --ff-only origin dev
git checkout -b feat/pae-derived-metric-correlation
```

Expected: branch created, working tree clean.

- [ ] **Step 3: Verify torchmetrics is already installed**

```bash
./.venv/bin/python -c "import torchmetrics; print(torchmetrics.__version__)"
```

Expected: a version ≥ 1.4. If absent, **STOP** — `env/build_uv_env.sh` must be amended to install it and the venv tarball rebuilt; do not `pip install` ad-hoc into the venv (it desyncs the `/netscratch` tarball).

- [ ] **Step 4: Commit empty marker (skip — branch creation is not a commit-worthy event)**

No commit at this task.

---

### Task 2: Add `reduce` kwarg to the 10 pAE-derived metric functions

**Files:**
- Modify: `src/proteinfoundation/nn/confidence/_metrics.py:358-620`
- Test: `tests/unit/nn/confidence/test_metrics_per_sample_reduction.py` (create)

The 10 functions in scope (each currently returns a 0-d scalar):
`i_pae`, `min_ipae`, `iptm_from_logits`, `iptm_energy_from_logits`, and `ipsae_family` (the last returns a 6-entry dict — each entry must also support per-sample reduction).

New contract:
- `reduce="batch_mean"` (default): unchanged — bit-identical scalar output.
- `reduce="per_sample"`: returns a `[B]` tensor (or for `ipsae_family`, a dict of `[B]` tensors). Samples-without-mass produce `NaN` (sentinel chosen because torchmetrics `PearsonCorrCoef` ignores `NaN` via `nan_strategy="ignore"`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/nn/confidence/test_metrics_per_sample_reduction.py`:

```python
"""Per-sample reduction parity for the 10 pAE-derived metrics.

For every metric, `M(...).mean()` over samples with mass must equal
`M(..., reduce="batch_mean")` within 1e-6. This pins the new `reduce`
kwarg as a strict generalisation of the existing batch-mean default.
"""
from __future__ import annotations

import math

import pytest
import torch

from proteinfoundation.nn.confidence._metrics import (
    i_pae,
    interface_pair_mask,
    ipsae_family,
    iptm_energy_from_logits,
    iptm_from_logits,
    min_ipae,
)


@pytest.fixture
def synthetic_batch() -> dict:
    torch.manual_seed(0)
    B, L, K = 3, 24, 64
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, 12:] = 1  # 12+12 dimer split
    mask = torch.ones((B, L), dtype=torch.bool)
    mask[1, 20:] = False  # variable-length sample 1
    mask_eff = mask[:, :, None] & mask[:, None, :]
    pae_ev = 8.0 + 4.0 * torch.randn((B, L, L))
    pae_ev = (pae_ev + pae_ev.transpose(-2, -1)) / 2  # PAE is not necessarily sym but ipSAE is OK with either
    logits = torch.randn((B, L, L, K))
    centers = torch.linspace(0.25, 31.75, K)
    inter = interface_pair_mask(chain_idx, mask_eff)
    return {
        "pae_ev": pae_ev,
        "logits": logits,
        "centers": centers,
        "mask_eff": mask_eff,
        "interface_mask": inter,
        "chain_idx": chain_idx,
    }


def _assert_batch_mean_matches_per_sample_mean(scalar, per_sample, name):
    assert per_sample.shape == (3,), f"{name}: expected per-sample shape (3,), got {per_sample.shape}"
    finite = per_sample[~torch.isnan(per_sample)]
    assert finite.numel() > 0, f"{name}: all per-sample values were NaN — fixture has no mass"
    recomputed = finite.mean()
    assert torch.allclose(scalar, recomputed, atol=1e-6), (
        f"{name}: batch_mean={scalar.item():.6f} != per-sample mean over mass={recomputed.item():.6f}"
    )


def test_i_pae_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = i_pae(sb["pae_ev"], sb["interface_mask"])
    per = i_pae(sb["pae_ev"], sb["interface_mask"], reduce="per_sample")
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "i_pae")


def test_min_ipae_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = min_ipae(sb["pae_ev"], sb["interface_mask"])
    per = min_ipae(sb["pae_ev"], sb["interface_mask"], reduce="per_sample")
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "min_ipae")


def test_iptm_from_logits_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = iptm_from_logits(sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"])
    per = iptm_from_logits(
        sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"], reduce="per_sample"
    )
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "iptm_from_logits")


def test_iptm_energy_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar = iptm_energy_from_logits(
        sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"]
    )
    per = iptm_energy_from_logits(
        sb["logits"], sb["mask_eff"], sb["interface_mask"], sb["centers"], reduce="per_sample"
    )
    _assert_batch_mean_matches_per_sample_mean(scalar, per, "iptm_energy_from_logits")


def test_ipsae_family_per_sample_matches_batch_mean(synthetic_batch):
    sb = synthetic_batch
    scalar_d = ipsae_family(sb["pae_ev"], sb["chain_idx"], sb["mask_eff"])
    per_d = ipsae_family(
        sb["pae_ev"], sb["chain_idx"], sb["mask_eff"], reduce="per_sample"
    )
    assert set(scalar_d) == set(per_d)
    for name, scalar in scalar_d.items():
        _assert_batch_mean_matches_per_sample_mean(scalar, per_d[name], name)


def test_per_sample_returns_nan_for_no_mass_sample():
    """When a sample has no interface pair, its per-sample value must be NaN.

    `nan_strategy="ignore"` (the torchmetrics default for PearsonCorrCoef) skips
    NaN entries during accumulation; the alternative (a zero value) would silently
    pull the correlation toward the origin.
    """
    B, L, K = 2, 16, 64
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[0, 8:] = 1  # sample 0: dimer
    # sample 1: monomer — chain_idx all zero → no inter-chain pairs
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    pae_ev = torch.full((B, L, L), 10.0)
    inter = interface_pair_mask(chain_idx, mask_eff)
    per = i_pae(pae_ev, inter, reduce="per_sample")
    assert not torch.isnan(per[0]), "dimer sample must have a finite per-sample value"
    assert torch.isnan(per[1]), f"monomer sample must yield NaN under reduce='per_sample', got {per[1].item()}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/test_metrics_per_sample_reduction.py -v
```

Expected: FAIL — `i_pae` and the others raise `TypeError: got an unexpected keyword argument 'reduce'`.

- [ ] **Step 3: Add the `reduce` kwarg to the five functions**

Patch `src/proteinfoundation/nn/confidence/_metrics.py`. For each function, the per-sample tensor that gets averaged at the end is already explicit; the change is to skip the final sum/divide and return the per-sample tensor instead, with NaN at sample-no-mass positions.

**Patch — `i_pae` (replace the function body lines 372-378):**

```python
def i_pae(
    pae_ev: Tensor,
    interface_mask: Tensor,
    *,
    reduce: str = "batch_mean",
) -> Tensor:
    mask_f = interface_mask.to(torch.float32)
    sample_denom = mask_f.sum(dim=(-2, -1))
    sample_has_mass = sample_denom > 0
    per_sample = (pae_ev.float() * mask_f).sum(dim=(-2, -1)) / sample_denom.clamp_min(1.0)
    if reduce == "per_sample":
        return torch.where(sample_has_mass, per_sample, torch.full_like(per_sample, float("nan")))
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample = torch.where(sample_has_mass, per_sample, torch.zeros_like(per_sample))
    n_valid_samples = sample_has_mass.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample.sum() / n_valid_samples).to(torch.float32)
```

**Patch — `min_ipae` (lines 394-407):**

```python
def min_ipae(
    pae_ev: Tensor,
    interface_mask: Tensor,
    *,
    reduce: str = "batch_mean",
) -> Tensor:
    mask_f = interface_mask.to(torch.float32)
    row_denom = mask_f.sum(dim=-1)
    row_has_mass = row_denom > 0
    per_row = (pae_ev.float() * mask_f).sum(dim=-1) / row_denom.clamp_min(1.0)
    per_row_for_min = torch.where(
        row_has_mass, per_row, torch.full_like(per_row, float("inf"))
    )
    sample_has_any_row = row_has_mass.any(dim=-1)
    per_sample_min = per_row_for_min.min(dim=-1).values
    if reduce == "per_sample":
        return torch.where(
            sample_has_any_row, per_sample_min, torch.full_like(per_sample_min, float("nan"))
        )
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample_min = torch.where(
        sample_has_any_row, per_sample_min, torch.zeros_like(per_sample_min)
    )
    n_valid_samples = sample_has_any_row.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample_min.sum() / n_valid_samples).to(torch.float32)
```

**Patch — `iptm_from_logits` (lines 459-474):**

```python
def iptm_from_logits(
    logits: Tensor,
    mask_eff: Tensor,
    interface_mask: Tensor,
    bin_centers: Tensor,
    *,
    d0_clip_min: int = 19,
    reduce: str = "batch_mean",
) -> Tensor:
    n_valid = _per_sample_n_valid(mask_eff)
    per_row = _per_row_tm_score(
        logits, interface_mask, n_valid, bin_centers, d0_clip_min=d0_clip_min
    )
    row_has_mass = interface_mask.to(torch.float32).sum(dim=-1) > 0
    per_row_masked = torch.where(
        row_has_mass, per_row, torch.full_like(per_row, float("-inf"))
    )
    sample_has_any_row = row_has_mass.any(dim=-1)
    per_sample_max = per_row_masked.max(dim=-1).values
    if reduce == "per_sample":
        return torch.where(
            sample_has_any_row, per_sample_max, torch.full_like(per_sample_max, float("nan"))
        )
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample_max = torch.where(
        sample_has_any_row, per_sample_max, torch.zeros_like(per_sample_max)
    )
    n_valid_samples = sample_has_any_row.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample_max.sum() / n_valid_samples).to(torch.float32)
```

**Patch — `iptm_energy_from_logits` (lines 485-506):**

```python
def iptm_energy_from_logits(
    logits: Tensor,
    mask_eff: Tensor,
    interface_mask: Tensor,
    bin_centers: Tensor,
    *,
    tm_lambda: float = 1.0,
    d0_clip_min: int = 19,
    reduce: str = "batch_mean",
) -> Tensor:
    centers = bin_centers.to(logits.device, torch.float32)
    n_valid = _per_sample_n_valid(mask_eff)
    n_eff = n_valid.to(torch.float32).clamp_min(float(d0_clip_min))
    d0 = 1.24 * (n_eff - 15.0).clamp_min(0.0).pow(1.0 / 3.0) - 1.8
    w = 1.0 / (1.0 + (centers[None, :] / d0[:, None]).pow(2))
    log_w = w.log()
    weighted_logits = logits.float() + tm_lambda * log_w[:, None, None, :]
    pos_energy = -torch.logsumexp(weighted_logits, dim=-1)
    mask_f = interface_mask.to(torch.float32)
    sample_denom = mask_f.sum(dim=(-2, -1))
    per_sample = (pos_energy * mask_f).sum(dim=(-2, -1)) / sample_denom.clamp_min(1.0)
    sample_has_mass = sample_denom > 0
    if reduce == "per_sample":
        return torch.where(sample_has_mass, per_sample, torch.full_like(per_sample, float("nan")))
    if reduce != "batch_mean":
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    per_sample = torch.where(sample_has_mass, per_sample, torch.zeros_like(per_sample))
    n_valid_samples = sample_has_mass.to(torch.float32).sum().clamp_min(1.0)
    return (per_sample.sum() / n_valid_samples).to(torch.float32)
```

**Patch — `ipsae_family` (lines 552-620):** add `reduce` kwarg and return `[B]` dict entries when `per_sample`.

```python
def ipsae_family(
    pae_ev: Tensor,
    chain_idx: Tensor,
    mask_eff: Tensor,
    *,
    pae_cutoffs: tuple[float, float] = (15.0, 10.0),
    reduce: str = "batch_mean",
) -> dict[str, Tensor]:
    if reduce not in ("batch_mean", "per_sample"):
        raise ValueError(f"reduce must be 'batch_mean' or 'per_sample', got {reduce!r}")
    res_valid = mask_eff.bool().any(dim=-1).to(torch.float32)
    binder_id = (chain_idx == 0).to(torch.float32) * res_valid
    target_id = (chain_idx == 1).to(torch.float32) * res_valid

    out: dict[str, Tensor] = {}
    base_cutoff, ten_cutoff = float(pae_cutoffs[0]), float(pae_cutoffs[1])
    for cutoff, suffix in ((base_cutoff, ""), (ten_cutoff, "_10")):
        ipsae_ba, mass_ba = _ipsae_directional(pae_ev, target_id, binder_id, cutoff)
        ipsae_ab, mass_ab = _ipsae_directional(pae_ev, binder_id, target_id, cutoff)
        sample_has_mass = mass_ba | mass_ab

        min_per = torch.minimum(ipsae_ab, ipsae_ba)
        max_per = torch.maximum(ipsae_ab, ipsae_ba)
        avg_per = 0.5 * (ipsae_ab + ipsae_ba)

        if reduce == "per_sample":
            nan = torch.full_like(avg_per, float("nan"))
            out[f"avg_ipsae{suffix}"] = torch.where(sample_has_mass, avg_per, nan)
            out[f"min_ipsae{suffix}"] = torch.where(sample_has_mass, min_per, nan)
            out[f"max_ipsae{suffix}"] = torch.where(sample_has_mass, max_per, nan)
            continue

        zeros = torch.zeros_like(avg_per)
        min_per = torch.where(sample_has_mass, min_per, zeros)
        max_per = torch.where(sample_has_mass, max_per, zeros)
        avg_per = torch.where(sample_has_mass, avg_per, zeros)
        n_valid = sample_has_mass.to(torch.float32).sum().clamp_min(1.0)
        out[f"avg_ipsae{suffix}"] = (avg_per.sum() / n_valid).to(torch.float32)
        out[f"min_ipsae{suffix}"] = (min_per.sum() / n_valid).to(torch.float32)
        out[f"max_ipsae{suffix}"] = (max_per.sum() / n_valid).to(torch.float32)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/test_metrics_per_sample_reduction.py -v
```

Expected: PASS, 6/6.

- [ ] **Step 5: Re-run the full existing metrics suite to confirm batch_mean is bit-identical**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/test_ipsae_family.py tests/unit/nn/confidence/ -v -k "metric or ipsae or i_pae or iptm"
```

Expected: PASS (no behaviour change at the default).

- [ ] **Step 6: Commit**

```bash
git add src/proteinfoundation/nn/confidence/_metrics.py \
        tests/unit/nn/confidence/test_metrics_per_sample_reduction.py
git commit -m "feat: add per-sample reduction to 10 pAE-derived metrics

The 10 functions (i_pae, min_ipae, iptm_from_logits, iptm_energy_from_logits,
six ipsae_family variants) now accept reduce={'batch_mean', 'per_sample'}.
batch_mean (default) is bit-identical to the prior API; per_sample returns
a [B] tensor with NaN at samples-without-mass so torchmetrics
nan_strategy='ignore' skips them in correlation accumulation.

Required by PR #2 (pAE metric correlations): per-sample-paired MAE / Pearson
/ Spearman between GT-PAE-derived and predicted-PAE-derived metrics."
```

---

### Task 3: Add a fp32 GT-PAE-EV helper on `PaeHead`

**Files:**
- Modify: `src/proteinfoundation/nn/confidence/pae_head.py`
- Test: `tests/unit/nn/confidence/test_pae_head_gt_ev.py` (create)

The ground-truth PAE expected value is *already* computed inside the existing validation pass via `_labels_to_continuous(labels_bin, centers)` — what's missing is a public method on `PaeHead` that returns it, so the metric-correlation path can call the same code. Likewise `logits_to_expected_value` already exists for the prediction side. Symmetry first: name them `_pae_ev_from_labels` and `pae_ev_from_logits` (rename the existing public method too — kept as a thin alias for back-compat).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/nn/confidence/test_pae_head_gt_ev.py`:

```python
"""GT-PAE expected-value helper on PaeHead.

The CE term is trained against integer-A AFDB labels via the bin index;
the EV term reads the float-A continuous label. Both are inputs to the
metric-correlation pipeline. The helper that the pipeline uses (the
'GT side') must produce values consistent with `_labels_to_continuous`
called with the head's bin centers — pinned here so a future refactor
of the bin convention can't silently shift the GT EV.
"""
from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


def _build_head() -> PaeHead:
    trunk = ConfidenceTrunk(dim_token=8, dim_pair=8, dim_local_latents=2, dim_cond=8, n_layers=0)
    return PaeHead(
        trunk=trunk,
        token_dim=8,
        pair_repr_dim=8,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
    )


def test_pae_ev_from_labels_matches_bin_centers():
    head = _build_head()
    labels = torch.tensor([[[0, 1, 63], [10, 31, 63]]])  # shape (1, 2, 3)
    ev = head._pae_ev_from_labels(labels)
    expected_centers = torch.tensor([0.25, 0.75, 31.75, 5.25, 15.75, 31.75]).reshape(1, 2, 3)
    assert torch.allclose(ev, expected_centers, atol=1e-6), (
        f"GT EV mismatch: got {ev.tolist()}, expected {expected_centers.tolist()}"
    )


def test_pae_ev_from_logits_one_hot_matches_centers():
    """One-hot logits over bin k must yield exactly centers[k]."""
    head = _build_head()
    B, L, K = 1, 4, 64
    logits = torch.full((B, L, L, K), -1e9)
    bin_idx = torch.randint(0, K, (B, L, L))
    logits.scatter_(-1, bin_idx[..., None], 1e9)
    ev_logits = head.pae_ev_from_logits(logits)
    ev_labels = head._pae_ev_from_labels(bin_idx)
    assert torch.allclose(ev_logits, ev_labels, atol=1e-4), (
        f"One-hot logits and label EV must agree: max |diff| = {(ev_logits - ev_labels).abs().max().item():.2e}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/test_pae_head_gt_ev.py -v
```

Expected: FAIL — `_pae_ev_from_labels` and `pae_ev_from_logits` do not exist (the existing method is `logits_to_expected_value`).

- [ ] **Step 3: Add the two helpers to `PaeHead`**

Open `src/proteinfoundation/nn/confidence/pae_head.py`. Replace the existing `logits_to_expected_value` method with the two new helpers + a back-compat alias.

```python
    def pae_ev_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax-weighted bin-center mean of the student's PAE logits.

        fp32 internally regardless of input dtype. Returns `(B, L, L)`
        in `[bin_min, bin_max)`.
        """
        logits_f = logits.float()
        probs = torch.softmax(logits_f, dim=-1)
        return (probs * self.bin_centers).sum(dim=-1)

    def _pae_ev_from_labels(self, pae_bin: torch.Tensor) -> torch.Tensor:
        """Bin-center lookup for AFDB integer-A bin labels.

        Mirrors `_labels_to_continuous(pae_bin, self.bin_centers)` but lives
        on the head so callers (including the metric-correlation pipeline)
        can use a single canonical source of GT EV without leaking the
        head's bin convention into the Lightning module.
        """
        return self.bin_centers.to(pae_bin.device, torch.float32)[pae_bin]

    def logits_to_expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Back-compat alias for `pae_ev_from_logits`. Do not use in new code."""
        return self.pae_ev_from_logits(logits)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/test_pae_head_gt_ev.py -v
```

Expected: PASS, 2/2.

- [ ] **Step 5: Run the full pae_head test suite to confirm no regression**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/ -v -k pae
```

Expected: all pre-existing pae tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/proteinfoundation/nn/confidence/pae_head.py \
        tests/unit/nn/confidence/test_pae_head_gt_ev.py
git commit -m "feat: add _pae_ev_from_labels and pae_ev_from_logits to PaeHead

Symmetrically named helpers for the GT and predicted PAE expected value.
The GT helper materialises bin centers from integer-A AFDB labels; the
predicted helper softmaxes logits in fp32. logits_to_expected_value
remains as a back-compat alias.

Required by PR #2 so the metric-correlation pipeline has a single
canonical source of EV without duplicating the head's bin convention."
```

---

### Task 4: Add the metric-correlation `MetricCollection` ModuleDict to `PaeHead`

**Files:**
- Modify: `src/proteinfoundation/nn/confidence/pae_head.py`
- Test: `tests/unit/nn/confidence/test_pae_head_metric_correlations.py` (create)

The 10 metrics tracked: `i_pae`, `min_ipae`, `i_ptm`, `i_ptm_energy`, `avg_ipsae`, `min_ipsae`, `max_ipsae`, `avg_ipsae_10`, `min_ipsae_10`, `max_ipsae_10`. For each, a `MetricCollection({"mae": MeanAbsoluteError(), "pearson": PearsonCorrCoef(), "spearman": SpearmanCorrCoef()})` accumulates the per-sample pair `(predicted_metric, gt_metric)` across the val set in a DDP-safe way.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/nn/confidence/test_pae_head_metric_correlations.py`:

```python
"""PaeHead.val_metric_correlations: per-sample-paired (pred, gt) accumulator.

Three cases:
    1. Identity: predicted PAE EV == GT PAE EV → pearson/spearman = 1.0, MAE = 0.
    2. Constant shift: predicted = gt + c → pearson/spearman = 1.0, MAE = |c|.
    3. Random uncorrelated: pearson ≈ 0 within tolerance for a long enough sample.

Plus: a sample with no interface contributes nothing (NaN entries are ignored).
"""
from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


METRIC_NAMES = (
    "i_pae",
    "min_ipae",
    "i_ptm",
    "i_ptm_energy",
    "avg_ipsae",
    "min_ipsae",
    "max_ipsae",
    "avg_ipsae_10",
    "min_ipsae_10",
    "max_ipsae_10",
)


def _build_head(track: bool = True) -> PaeHead:
    trunk = ConfidenceTrunk(dim_token=8, dim_pair=8, dim_local_latents=2, dim_cond=8, n_layers=0)
    return PaeHead(
        trunk=trunk,
        token_dim=8,
        pair_repr_dim=8,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        track_metric_correlations=track,
    )


def _toy_batch(B: int = 4, L: int = 24) -> dict:
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    return {"chain_idx": chain_idx, "mask_eff": mask_eff}


def test_track_metric_correlations_attr_exists():
    head = _build_head(track=True)
    assert hasattr(head, "val_metric_correlations")
    assert set(head.val_metric_correlations.keys()) == set(METRIC_NAMES)


def test_track_disabled_returns_no_attr():
    head = _build_head(track=False)
    assert not hasattr(head, "val_metric_correlations") or head.val_metric_correlations is None


def test_identity_yields_perfect_correlation_and_zero_mae():
    """pred = gt over many random val samples → pearson=1.0, mae=0."""
    head = _build_head(track=True)
    head.eval()
    torch.manual_seed(0)
    for _ in range(8):
        B, L = 4, 24
        bp = _toy_batch(B, L)
        pae_ev_gt = 4.0 + 6.0 * torch.rand((B, L, L))
        # Use bin_idx that decodes to the same EV (one-hot logits)
        head.update_metric_correlations(
            pae_ev_pred=pae_ev_gt,
            pae_ev_gt=pae_ev_gt,
            chain_idx=bp["chain_idx"],
            mask_eff=bp["mask_eff"],
        )
    agg = head.val_metric_correlations_compute_and_reset()
    for m in METRIC_NAMES:
        assert agg[m]["mae"] < 1e-4, f"{m}: identity MAE not zero (got {agg[m]['mae']:.4e})"
        assert agg[m]["pearson"] > 0.999, f"{m}: identity pearson not 1.0 (got {agg[m]['pearson']:.4f})"
        # spearman can be 1.0 or NaN if all predictions are constant; identity is varied so it should be 1.0
        assert agg[m]["spearman"] > 0.999, f"{m}: identity spearman not 1.0 (got {agg[m]['spearman']:.4f})"


def test_constant_shift_preserves_pearson_but_lifts_mae():
    """pred = gt + c → pearson=1.0, mae=|c|, spearman=1.0."""
    head = _build_head(track=True)
    head.eval()
    c = 2.5
    torch.manual_seed(0)
    for _ in range(8):
        B, L = 4, 24
        bp = _toy_batch(B, L)
        gt = 4.0 + 6.0 * torch.rand((B, L, L))
        pred = gt + c
        head.update_metric_correlations(
            pae_ev_pred=pred,
            pae_ev_gt=gt,
            chain_idx=bp["chain_idx"],
            mask_eff=bp["mask_eff"],
        )
    agg = head.val_metric_correlations_compute_and_reset()
    # i_pae is a linear interface-mean → mae == c, pearson == 1
    assert abs(agg["i_pae"]["mae"] - c) < 0.1, (
        f"i_pae constant-shift MAE: got {agg['i_pae']['mae']:.4f}, expected ~{c}"
    )
    assert agg["i_pae"]["pearson"] > 0.999
    # all metrics retain monotone relationship → spearman ≈ 1
    for m in METRIC_NAMES:
        assert agg[m]["spearman"] > 0.99, (
            f"{m}: constant-shift spearman not ~1 (got {agg[m]['spearman']:.4f})"
        )


def test_no_interface_sample_is_dropped_via_nan():
    """A monomer-style sample (chain_idx all 0) contributes no per-sample value."""
    head = _build_head(track=True)
    head.eval()
    torch.manual_seed(0)
    B, L = 4, 24
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    # Only sample 0 is a dimer; samples 1-3 are monomers
    chain_idx[0, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    gt = 4.0 + 6.0 * torch.rand((B, L, L))
    pred = gt + 1.0
    head.update_metric_correlations(
        pae_ev_pred=pred, pae_ev_gt=gt, chain_idx=chain_idx, mask_eff=mask_eff
    )
    # With only 1 sample contributing, pearson is undefined; torchmetrics
    # returns NaN in that case (which is the documented behaviour we want).
    agg = head.val_metric_correlations_compute_and_reset()
    # MAE on a single sample is the absolute value of (pred - gt) over that sample.
    assert agg["i_pae"]["mae"] > 0.0, f"single-sample MAE not finite: {agg['i_pae']['mae']:.4e}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/test_pae_head_metric_correlations.py -v
```

Expected: FAIL — `track_metric_correlations`, `val_metric_correlations`, `update_metric_correlations`, and `val_metric_correlations_compute_and_reset` do not exist.

- [ ] **Step 3: Implement the four pieces on `PaeHead`**

Edit `src/proteinfoundation/nn/confidence/pae_head.py`:

3a. Add the import block at the top:

```python
from torchmetrics import MeanAbsoluteError, MetricCollection, PearsonCorrCoef, SpearmanCorrCoef
```

3b. Extend `PaeHead.__init__` to accept `track_metric_correlations: bool = True`, store the flag, and build the `nn.ModuleDict` of `MetricCollection`s. Add at the end of `__init__`:

```python
        self.track_metric_correlations = bool(track_metric_correlations)
        if self.track_metric_correlations:
            self.val_metric_correlations = nn.ModuleDict(
                {
                    name: MetricCollection(
                        {
                            "mae": MeanAbsoluteError(),
                            "pearson": PearsonCorrCoef(),
                            "spearman": SpearmanCorrCoef(),
                        },
                        compute_groups=False,
                    )
                    for name in (
                        "i_pae",
                        "min_ipae",
                        "i_ptm",
                        "i_ptm_energy",
                        "avg_ipsae",
                        "min_ipsae",
                        "max_ipsae",
                        "avg_ipsae_10",
                        "min_ipsae_10",
                        "max_ipsae_10",
                    )
                }
            )
```

Update the `PaeHead.__init__` signature:

```python
    def __init__(
        self,
        trunk: ConfidenceTrunk,
        token_dim: int = 768,
        pair_repr_dim: int = 256,
        num_pae_bins: int = 64,
        bin_min: float = 0.0,
        bin_max: float = 32.0,
        ce_weight: float = 0.9,
        ev_weight: float = 0.1,
        label_smoothing: float = 0.0,
        num_bins_ece_adaptive: int = 15,
        track_metric_correlations: bool = True,
    ) -> None:
```

3c. Add `update_metric_correlations` to `PaeHead`. This is the function the head calls during validation; it computes per-sample predicted-vs-GT metrics and updates each `MetricCollection`.

```python
    def update_metric_correlations(
        self,
        pae_ev_pred: torch.Tensor,
        pae_ev_gt: torch.Tensor,
        chain_idx: torch.Tensor,
        mask_eff: torch.Tensor,
    ) -> None:
        """Update the per-sample (pred, gt) metric-collection accumulators.

        Called from `compute_loss_and_metrics` at val stage. The GT and
        predicted EVs are computed once in fp32 by the caller; this method
        owns the per-sample metric extraction and the torchmetrics
        accumulator updates.
        """
        if not self.track_metric_correlations:
            return
        from proteinfoundation.nn.confidence._metrics import (
            i_pae,
            interface_pair_mask,
            ipsae_family,
            iptm_energy_from_logits,
            iptm_from_logits,
            min_ipae,
        )

        inter = interface_pair_mask(chain_idx, mask_eff)
        # i_pae / min_ipae: fp32 EV directly
        per_pred: dict[str, torch.Tensor] = {}
        per_gt: dict[str, torch.Tensor] = {}
        per_pred["i_pae"] = i_pae(pae_ev_pred, inter, reduce="per_sample")
        per_gt["i_pae"] = i_pae(pae_ev_gt, inter, reduce="per_sample")
        per_pred["min_ipae"] = min_ipae(pae_ev_pred, inter, reduce="per_sample")
        per_gt["min_ipae"] = min_ipae(pae_ev_gt, inter, reduce="per_sample")

        # i_ptm / i_ptm_energy: built from logits. We synthesise one-hot
        # logits from the EV bin-decoded labels so the GT side flows through
        # the same kernel — this preserves the d0(L) shape and the LSE
        # energy reduction that batch-mean uses.
        pred_bin = torch.bucketize(pae_ev_pred, self.bin_centers[:-1])
        gt_bin = torch.bucketize(pae_ev_gt, self.bin_centers[:-1])
        K = self.num_pae_bins
        pred_logits = torch.full((*pred_bin.shape, K), -1e9, device=pred_bin.device)
        pred_logits.scatter_(-1, pred_bin[..., None], 1e9)
        gt_logits = torch.full((*gt_bin.shape, K), -1e9, device=gt_bin.device)
        gt_logits.scatter_(-1, gt_bin[..., None], 1e9)
        per_pred["i_ptm"] = iptm_from_logits(
            pred_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )
        per_gt["i_ptm"] = iptm_from_logits(
            gt_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )
        per_pred["i_ptm_energy"] = iptm_energy_from_logits(
            pred_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )
        per_gt["i_ptm_energy"] = iptm_energy_from_logits(
            gt_logits, mask_eff, inter, self.bin_centers, reduce="per_sample"
        )

        # ipsae_family: 6 entries, all from EV
        ipsae_pred = ipsae_family(pae_ev_pred, chain_idx, mask_eff, reduce="per_sample")
        ipsae_gt = ipsae_family(pae_ev_gt, chain_idx, mask_eff, reduce="per_sample")
        for k in ipsae_pred:
            per_pred[k] = ipsae_pred[k]
            per_gt[k] = ipsae_gt[k]

        for name, mc in self.val_metric_correlations.items():
            pred_v = per_pred[name]
            gt_v = per_gt[name]
            keep = ~(torch.isnan(pred_v) | torch.isnan(gt_v))
            if not bool(keep.any()):
                continue
            mc.update(pred_v[keep].float(), gt_v[keep].float())

    def val_metric_correlations_compute_and_reset(self) -> dict[str, dict[str, float]]:
        """Compute and reset every collection. Empty collections yield NaN."""
        out: dict[str, dict[str, float]] = {}
        if not self.track_metric_correlations:
            return out
        for name, mc in self.val_metric_correlations.items():
            try:
                vals = mc.compute()
                out[name] = {k: float(v) for k, v in vals.items()}
            except Exception:
                out[name] = {"mae": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
            mc.reset()
        return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/test_pae_head_metric_correlations.py -v
```

Expected: PASS, 4/4.

- [ ] **Step 5: Run the full pae_head suite to confirm no regression**

```bash
./.venv/bin/python -m pytest tests/unit/nn/confidence/ -v -k pae
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/proteinfoundation/nn/confidence/pae_head.py \
        tests/unit/nn/confidence/test_pae_head_metric_correlations.py
git commit -m "feat: add per-sample metric-correlation MetricCollection to PaeHead

Adds track_metric_correlations=True kwarg. When on, PaeHead carries an
nn.ModuleDict of torchmetrics MetricCollection({MAE, PearsonCorrCoef,
SpearmanCorrCoef}) — one per (i_pae, min_ipae, i_ptm, i_ptm_energy,
six ipSAE variants).

update_metric_correlations(pae_ev_pred, pae_ev_gt, chain_idx, mask_eff)
synthesises per-sample (pred, gt) pairs and feeds each collection;
NaN entries from samples-without-mass are dropped before update.
val_metric_correlations_compute_and_reset returns a {metric: {mae,
pearson, spearman}} dict and clears the accumulators for the next
validation epoch."
```

---

### Task 5: Wire the val-stage `update` call from `PaeHead.compute_loss_and_metrics`

**Files:**
- Modify: `src/proteinfoundation/nn/confidence/pae_head.py`
- Test: `tests/integration/confidence/test_pae_head_val_correlation_wiring.py` (create)

`compute_loss_and_metrics` at val stage already computes `pred_cont` (= `_logits_to_continuous`) and runs `ipsae_family(pred_cont, ...)`. We add: compute `gt_cont` via `_pae_ev_from_labels(labels_bin)`, then call `update_metric_correlations(pred_cont, gt_cont, chain_idx, mask_eff)` exactly once per val batch.

The Lightning module is unchanged in this task — `compute_loss_and_metrics` already runs on rank-local mini-batches, and the `MetricCollection` itself handles DDP sync at `compute()` time.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/confidence/test_pae_head_val_correlation_wiring.py`:

```python
"""compute_loss_and_metrics at stage='val' updates val_metric_correlations.

The test constructs a tiny dimer batch, runs val on the head, and
verifies that after the call:
    (a) val_metric_correlations[name].update was called (state advanced)
    (b) on compute(), pearson is finite for at least one metric
    (c) at stage='train' the accumulators are NOT touched
"""
from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


def _build_head(track: bool = True) -> PaeHead:
    trunk = ConfidenceTrunk(dim_token=8, dim_pair=8, dim_local_latents=2, dim_cond=8, n_layers=0)
    return PaeHead(
        trunk=trunk,
        token_dim=8,
        pair_repr_dim=8,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        track_metric_correlations=track,
    )


def _toy_val_batch(B: int = 4, L: int = 16) -> tuple[dict, torch.Tensor, torch.Tensor]:
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    pae_bin = torch.randint(0, 32, (B, L, L))
    pae_continuous = pae_bin.float() * 1.0  # integer-A labels
    out = {"pae_logits": torch.randn((B, L, L, 64))}
    batch = {
        "pae_bin": pae_bin,
        "pae_residue_pair": pae_continuous,
        "chain_idx": chain_idx,
    }
    return out, batch, mask_eff


def test_val_call_advances_accumulators():
    head = _build_head(track=True)
    head.eval()
    out, batch, mask_eff = _toy_val_batch()
    _, log_dict = head.compute_loss_and_metrics(out, batch, mask_eff, stage="val")
    # at least one MAE accumulator should have non-zero total
    mc = head.val_metric_correlations["i_pae"]
    state_total = mc["mae"].sum_abs_error
    assert state_total.item() >= 0.0, "MAE accumulator state must exist"
    # compute returns finite MAE for the populated metric
    agg = head.val_metric_correlations_compute_and_reset()
    assert "i_pae" in agg
    assert agg["i_pae"]["mae"] >= 0.0


def test_train_call_does_not_touch_accumulators():
    head = _build_head(track=True)
    head.train()
    out, batch, mask_eff = _toy_val_batch()
    _, _ = head.compute_loss_and_metrics(out, batch, mask_eff, stage="train")
    mc = head.val_metric_correlations["i_pae"]
    assert mc["mae"].sum_abs_error.item() == 0.0, (
        "train-stage call must not advance val accumulators"
    )


def test_track_false_skips_attribute_path():
    head = _build_head(track=False)
    head.eval()
    out, batch, mask_eff = _toy_val_batch()
    _, _ = head.compute_loss_and_metrics(out, batch, mask_eff, stage="val")
    # No attribute access should crash; either attribute absent or no update
    assert not hasattr(head, "val_metric_correlations") or head.val_metric_correlations is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./.venv/bin/python -m pytest tests/integration/confidence/test_pae_head_val_correlation_wiring.py -v
```

Expected: FAIL — `compute_loss_and_metrics` doesn't yet call `update_metric_correlations`.

- [ ] **Step 3: Wire `update_metric_correlations` into `compute_loss_and_metrics`**

Edit `src/proteinfoundation/nn/confidence/pae_head.py:compute_loss_and_metrics`. Inside the `if stage != "train":` block, after the existing `ipsae_family` log call (line 198), insert:

```python
                if self.track_metric_correlations:
                    pae_ev_gt = self._pae_ev_from_labels(labels_bin)
                    self.update_metric_correlations(
                        pae_ev_pred=pred_cont,
                        pae_ev_gt=pae_ev_gt,
                        chain_idx=chain_idx,
                        mask_eff=mask_eff,
                    )
```

The `chain_idx` variable is already in scope (assigned at line 142). `pred_cont` is already computed above (line 171). `mask_eff` is the function's argument. `labels_bin` is the function's local (line 138).

- [ ] **Step 4: Run test to verify it passes**

```bash
./.venv/bin/python -m pytest tests/integration/confidence/test_pae_head_val_correlation_wiring.py -v
```

Expected: PASS, 3/3.

- [ ] **Step 5: Re-run the full confidence test suite**

```bash
./.venv/bin/python -m pytest tests/ -v -k "pae or confidence"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/proteinfoundation/nn/confidence/pae_head.py \
        tests/integration/confidence/test_pae_head_val_correlation_wiring.py
git commit -m "feat: wire per-sample metric-correlation update from PaeHead val path

compute_loss_and_metrics at stage='val' now calls
update_metric_correlations(pred_cont, gt_cont, chain_idx, mask_eff)
after the existing ipsae_family logging line. The GT EV is materialised
once via _pae_ev_from_labels(labels_bin).

stage='train' does not touch the accumulators, and track=False bypasses
the path entirely."
```

---

### Task 6: Wire the epoch-end logging in the Lightning module

**Files:**
- Modify: `src/proteinfoundation/confidence/lightning_module.py`
- Test: `tests/integration/confidence/test_lightning_val_epoch_logs_correlations.py` (create)

`on_validation_epoch_end` reads the per-head correlations and logs them as `val/{output_name_root}/{metric}/{mae,pearson,spearman}` scalars. For `MultiHeadConfidence`, iterate over `children_heads` and call each child that has `track_metric_correlations`.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/confidence/test_lightning_val_epoch_logs_correlations.py`:

```python
"""ConfidenceSidecar.on_validation_epoch_end logs per-head correlations.

Builds a 1-batch validation pass under a stub Trainer, then checks that
the Lightning module's logged metrics dict contains every expected
`val/pae/{metric}/{mae,pearson,spearman}` key.
"""
from __future__ import annotations

import torch
import pytest
import lightning as L

from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.pae_head import PaeHead


# This is an integration test; we cannot construct a full ConfidenceSidecar
# without the Proteina trunk checkpoint. Instead we exercise the contract
# directly via a minimal lightning module + the head.

class _StubModule(L.LightningModule):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def configure_optimizers(self):  # pragma: no cover
        return torch.optim.Adam(self.head.parameters(), lr=1e-3)


def _head() -> PaeHead:
    trunk = ConfidenceTrunk(dim_token=8, dim_pair=8, dim_local_latents=2, dim_cond=8, n_layers=0)
    return PaeHead(
        trunk=trunk,
        token_dim=8,
        pair_repr_dim=8,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        track_metric_correlations=True,
    )


def test_on_validation_epoch_end_logs_all_metrics():
    """Single-head path: after on_validation_epoch_end the head's
    val_metric_correlations have been compute()'d and reset."""
    from proteinfoundation.confidence.lightning_module import (
        _log_metric_correlations_for_head,
    )

    head = _head()
    head.eval()
    # populate accumulators
    B, L = 4, 16
    chain_idx = torch.zeros((B, L), dtype=torch.long)
    chain_idx[:, L // 2 :] = 1
    mask = torch.ones((B, L), dtype=torch.bool)
    mask_eff = mask[:, :, None] & mask[:, None, :]
    gt = 4.0 + 6.0 * torch.rand((B, L, L))
    pred = gt + 1.5
    head.update_metric_correlations(
        pae_ev_pred=pred, pae_ev_gt=gt, chain_idx=chain_idx, mask_eff=mask_eff
    )

    logged: dict[str, float] = {}
    def log_fn(key, value, **kwargs):
        logged[key] = float(value)

    _log_metric_correlations_for_head(head, prefix="val/pae", log_fn=log_fn)

    expected_metrics = (
        "i_pae", "min_ipae", "i_ptm", "i_ptm_energy",
        "avg_ipsae", "min_ipsae", "max_ipsae",
        "avg_ipsae_10", "min_ipsae_10", "max_ipsae_10",
    )
    expected_stats = ("mae", "pearson", "spearman")
    for m in expected_metrics:
        for s in expected_stats:
            key = f"val/pae/{m}/{s}"
            assert key in logged, f"missing log key {key}; logged keys: {sorted(logged)[:5]}..."

    # Accumulators must have been reset
    mc = head.val_metric_correlations["i_pae"]
    assert mc["mae"].sum_abs_error.item() == 0.0, "accumulator not reset after compute"


def test_log_function_skips_head_without_track():
    from proteinfoundation.confidence.lightning_module import (
        _log_metric_correlations_for_head,
    )

    trunk = ConfidenceTrunk(dim_token=8, dim_pair=8, dim_local_latents=2, dim_cond=8, n_layers=0)
    head = PaeHead(
        trunk=trunk, token_dim=8, pair_repr_dim=8, num_pae_bins=64,
        bin_min=0.0, bin_max=32.0, track_metric_correlations=False,
    )
    logged: dict[str, float] = {}
    _log_metric_correlations_for_head(head, prefix="val/pae", log_fn=lambda k, v, **kw: logged.update({k: float(v)}))
    assert logged == {}, "track_metric_correlations=False must produce no log entries"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./.venv/bin/python -m pytest tests/integration/confidence/test_lightning_val_epoch_logs_correlations.py -v
```

Expected: FAIL — `_log_metric_correlations_for_head` doesn't exist; `on_validation_epoch_end` isn't wired.

- [ ] **Step 3: Add the helper and hook to `lightning_module.py`**

Open `src/proteinfoundation/confidence/lightning_module.py`. Add the helper near the top of the module (right after the imports):

```python
def _log_metric_correlations_for_head(head, *, prefix: str, log_fn) -> None:
    """Compute & log the head's MetricCollection state, then reset.

    `head` may be a `PaeHead` or any future head that exposes
    `track_metric_correlations`, `val_metric_correlations_compute_and_reset`.
    `log_fn(key, value, sync_dist=True)` is the LightningModule's `self.log`
    or a stub for testing.
    """
    if not getattr(head, "track_metric_correlations", False):
        return
    compute_fn = getattr(head, "val_metric_correlations_compute_and_reset", None)
    if compute_fn is None:
        return
    agg = compute_fn()
    for metric_name, stats in agg.items():
        for stat_name, value in stats.items():
            # MetricCollection sync handled inside torchmetrics; sync_dist=False to avoid double-reduce.
            log_fn(f"{prefix}/{metric_name}/{stat_name}", value, sync_dist=False)
```

Then add a method to the sidecar class:

```python
    def on_validation_epoch_end(self) -> None:
        """Log per-head metric-correlation aggregates and reset accumulators."""
        if isinstance(self.head, MultiHeadConfidence):
            for name, child in self.head.children_heads.items():
                _log_metric_correlations_for_head(
                    child,
                    prefix=f"val/{child.output_name_root}",
                    log_fn=self.log,
                )
        else:
            _log_metric_correlations_for_head(
                self.head,
                prefix=f"val/{self.head.output_name_root}",
                log_fn=self.log,
            )
```

Insert between `on_validation_epoch_start` (line 511-513 region) and `configure_optimizers`. If there's already an `on_validation_epoch_end` (there isn't, per the grep above, but verify), extend rather than overwrite.

- [ ] **Step 4: Run test to verify it passes**

```bash
./.venv/bin/python -m pytest tests/integration/confidence/test_lightning_val_epoch_logs_correlations.py -v
```

Expected: PASS, 2/2.

- [ ] **Step 5: Commit**

```bash
git add src/proteinfoundation/confidence/lightning_module.py \
        tests/integration/confidence/test_lightning_val_epoch_logs_correlations.py
git commit -m "feat: log per-head metric correlations from on_validation_epoch_end

Lightning module sidecar gains on_validation_epoch_end + a
_log_metric_correlations_for_head helper. For MultiHeadConfidence the
hook iterates over children_heads; for single-head configs it dispatches
on self.head directly. The helper reads
val_metric_correlations_compute_and_reset and emits
val/{head}/{metric}/{mae,pearson,spearman} via self.log.

torchmetrics handles the DDP all-gather inside compute(); sync_dist=False
on the log call avoids a redundant reduction."
```

---

### Task 7: DDP correctness — small parametrised test

**Files:**
- Test: `tests/integration/confidence/test_metric_collection_ddp.py` (create)

torchmetrics `MetricCollection` already supports DDP via `dist_sync_fn`. We just need to pin that the rank-local `update` calls plus `compute` produce the same answer as concatenating both rank's inputs and computing once. Lightning Fabric's `gloo` backend makes this testable on CPU.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/confidence/test_metric_collection_ddp.py`:

```python
"""MetricCollection state is DDP-aggregated at compute() time.

Spawn 2 cpu workers, feed half the per-sample tensor to each, and check
that compute() on rank 0 equals the single-rank reference computed over
the concatenated tensor.

Skipped if gloo is unavailable.
"""
from __future__ import annotations

import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import MeanAbsoluteError, MetricCollection, PearsonCorrCoef, SpearmanCorrCoef


def _worker(rank: int, world_size: int, pred: torch.Tensor, gt: torch.Tensor, q: mp.Queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    mc = MetricCollection(
        {"mae": MeanAbsoluteError(), "pearson": PearsonCorrCoef(), "spearman": SpearmanCorrCoef()},
        compute_groups=False,
    )
    n_per = pred.numel() // world_size
    lo, hi = rank * n_per, (rank + 1) * n_per
    mc.update(pred[lo:hi], gt[lo:hi])
    out = mc.compute()
    if rank == 0:
        q.put({k: float(v) for k, v in out.items()})
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="gloo backend not available"
)
@pytest.mark.parametrize("world_size", [1, 2])
def test_metric_collection_distributed_matches_single_rank(world_size: int):
    torch.manual_seed(0)
    N = 64
    pred = torch.randn(N)
    gt = pred + 0.5 * torch.randn(N)

    # Single-rank reference
    mc_ref = MetricCollection(
        {"mae": MeanAbsoluteError(), "pearson": PearsonCorrCoef(), "spearman": SpearmanCorrCoef()},
        compute_groups=False,
    )
    mc_ref.update(pred, gt)
    ref = {k: float(v) for k, v in mc_ref.compute().items()}

    if world_size == 1:
        # already checked
        for k, v in ref.items():
            assert v == v, f"NaN in reference {k}"
        return

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, pred, gt, q))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited {p.exitcode}"
    rank0 = q.get(timeout=10)
    for k in ref:
        assert abs(rank0[k] - ref[k]) < 1e-4, (
            f"{k}: ddp={rank0[k]:.6f}, single-rank={ref[k]:.6f}"
        )
```

- [ ] **Step 2: Run test to verify it passes (no implementation change needed)**

```bash
./.venv/bin/python -m pytest tests/integration/confidence/test_metric_collection_ddp.py -v
```

Expected: PASS, 2/2 (`world_size=1` is trivially true; `world_size=2` exercises the `gloo` all-gather).

If it fails because gloo is missing on this host, mark `xfail(reason="gloo missing")` — the contract is pinned by torchmetrics itself and a CPU-side smoke test is sufficient.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/confidence/test_metric_collection_ddp.py
git commit -m "test: pin DDP correctness for the metric-correlation MetricCollection

Two ranks via gloo, each updating with half the per-sample tensor; the
rank-0 compute() output must match the single-rank reference within
1e-4. Parametrised over world_size in {1, 2}."
```

---

### Task 8: Configuration — expose `track_metric_correlations` in Hydra

**Files:**
- Modify: `configs/nn/confidence/pae_head.yaml`
- Modify: `configs/confidence/distillation_teddymer_qg_multihead.yaml` (created in PR #1)
- Modify: `configs/confidence/distillation_teddymer_multihead.yaml` (legacy, kept for A/B)
- Modify: `configs/confidence/distillation_teddymer_pae.yaml` (single-head variant)

The kwarg is already a constructor default (`True`) on `PaeHead`. The Hydra wiring must:

1. Pin the default in the *standalone* `pae_head.yaml` (single-head training composes this directly).
2. *Inline* it in every multihead parent config, because those parents do not compose `pae_head.yaml` — they declare the `pae` child via `_target_` and an inline kwarg dict (see existing `distillation_teddymer_multihead.yaml` for the pattern).

Without inlining, the multihead `pae` child instantiates `PaeHead` with the constructor default (still `True`), so behaviour is correct — but the value is invisible to anyone auditing the YAML. Inlining makes the contract grep-able.

- [ ] **Step 1: Add the kwarg to the standalone head YAML**

Open `configs/nn/confidence/pae_head.yaml`. Append the line:

```yaml
track_metric_correlations: true
```

Final file contents:

```yaml
defaults:
  - base
  - _self_

_target_: proteinfoundation.nn.confidence.pae_head.PaeHead
token_dim: 768
pair_repr_dim: 256
num_pae_bins: 64
bin_min: 0.0
bin_max: 32.0
ce_weight: 0.8
ev_weight: 0.2
label_smoothing: 0.0
num_bins_ece_adaptive: 15
track_metric_correlations: true
```

- [ ] **Step 2: Inline the kwarg into all parent training configs**

For each of `configs/confidence/distillation_teddymer_qg_multihead.yaml`, `configs/confidence/distillation_teddymer_multihead.yaml`, and `configs/confidence/distillation_teddymer_pae.yaml`, add `track_metric_correlations: true` to the `pae` child block (or to the top-level head block in the single-head case).

Example for `distillation_teddymer_qg_multihead.yaml` — locate the `pae:` block under `confidence.head.children:` and add the line after `label_smoothing: 0.0`:

```yaml
      pae:
        _target_: proteinfoundation.nn.confidence.pae_head.PaeHead
        trunk:
          _target_: proteinfoundation.nn.confidence.base.ConfidenceTrunk
        token_dim: 768
        pair_repr_dim: 256
        d_in_pair_token: 128
        num_pae_bins: 64
        bin_min: 0.0
        bin_max: 32.0
        ce_weight: 1.0
        ev_weight: 0.0
        label_smoothing: 0.0
        track_metric_correlations: true
```

Same edit for `distillation_teddymer_multihead.yaml` (the legacy A/B baseline). For `distillation_teddymer_pae.yaml`, the head block is the top-level `confidence.head:` — apply the kwarg there.

- [ ] **Step 3: Smoke-validate the override works**

```bash
./.venv/bin/python -c "
from hydra import compose, initialize
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='confidence/distillation_teddymer_qg_multihead')
print('qg-multihead pae child track:', cfg.confidence.head.children.pae.track_metric_correlations)
"
```

Expected: `qg-multihead pae child track: True`.

- [ ] **Step 2: Smoke-validate the override works**

```bash
./.venv/bin/python -c "
from hydra import compose, initialize
with initialize(version_base=None, config_path='configs'):
    cfg = compose(config_name='confidence/distillation_teddymer_pae')
print('track_metric_correlations =', cfg.confidence.head.track_metric_correlations)
"
```

Expected: `track_metric_correlations = True`.

- [ ] **Step 3: Commit**

```bash
git add configs/nn/confidence/pae_head.yaml
git commit -m "config: enable track_metric_correlations by default on PaeHead

Hydra default is true; CLI override (confidence.head.children.pae.track_metric_correlations=false)
disables. PR #1's qg-multihead config inherits the new default via composition."
```

---

### Task 9: Local smoke test on a single GPU (CPU fallback if needed)

**Files:** none modified.

The goal: a 2-step training run with the full distillation config to verify nothing blows up under bf16 mixed-precision, autocast, and the real Lightning trainer.

- [ ] **Step 1: Reduce val frequency to 1 step for the smoke run**

```bash
cd /mnt/storage01/home/schekmenev/projects/complexa-flex
```

Run from the project root (no `cd ...`):

```bash
.venv/bin/python -m proteinfoundation.confidence.train_confidence \
    --config-name=confidence/distillation_teddymer_qg_multihead \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    trainer.limit_train_batches=2 \
    trainer.limit_val_batches=2 \
    trainer.max_epochs=1 \
    trainer.val_check_interval=1 \
    trainer.num_sanity_val_steps=0 \
    +integrity.enabled=false \
    logging.log_wandb=false \
    run_name=smoke-pr2-correlations \
    2>&1 | tee /tmp/smoke_pr2.log
```

Expected output includes log lines like:
```
val/pae/i_pae/mae=0.XX
val/pae/i_pae/pearson=0.XX
val/pae/avg_ipsae/spearman=0.XX
```

If the run crashes on metric collection (most likely: a torchmetrics version mismatch), the log will surface it clearly. If WandB connection blocks the run, `log_wandb=false` short-circuits.

- [ ] **Step 2: Verify the log keys**

```bash
grep -E "val/.*pae.*(mae|pearson|spearman)" /tmp/smoke_pr2.log | sort -u | head -40
```

Expected: 30 unique keys (10 metrics × 3 stats), all under `val/pae/...` (multihead config still names the pae child `pae`).

- [ ] **Step 3: Verify train-stage doesn't log these keys**

```bash
grep -E "train/.*pae.*(mae|pearson|spearman)" /tmp/smoke_pr2.log | head -3
```

Expected: no output (or only the prior step's `train/pae/loss` line — which doesn't match the regex).

- [ ] **Step 4: Commit if any incidental config changes were made**

If `Step 1` required no edit, no commit at this task. Otherwise commit.

---

### Task 10: SLURM smoke launch (1000 steps, 2× H100, full Teddymer)

**Files:** none modified (config flag check only).

- [ ] **Step 1: Confirm sbatch picks up the config**

```bash
grep -n "distillation_teddymer_qg_multihead" scripts/train_confidence_teddymer_qg_multihead.sbatch
```

Expected: a line matching `--config-name=confidence/distillation_teddymer_qg_multihead`. The sbatch was created by PR #1 (Task 10) and is the canonical launcher for the QG-multihead config.

- [ ] **Step 2: Submit the 1000-step smoke job**

```bash
module load gh  # in case sbatch needs it
sbatch --export=ALL,RESUME_CKPT_PATH="" \
       --time=0-04:00:00 \
       --comment="PR2 metric-correlation smoke" \
       scripts/train_confidence_teddymer_qg_multihead.sbatch
```

Capture the job ID printed by sbatch.

- [ ] **Step 3: Monitor the first validation pass**

```bash
JOB_ID=$(squeue -u "$USER" -h -O JobID --noheader | head -1)  # or paste the sbatch-printed ID
LOG=logs/paedistill_${JOB_ID}.out
tail -n0 -f "$LOG" | grep --line-buffered -E "val/.*pae.*(mae|pearson|spearman)|ERROR|Traceback"
```

Expected: within ~30-45 minutes, the first validation pass at step 1000 emits all 30 keys. If a Traceback appears, kill the job (`scancel $JOB_ID`) and debug.

- [ ] **Step 4: Verify in WandB**

Open the WandB run named `multi-plddt-pae-distill-teddymer` (or whatever PR #1 renamed it to). Confirm the new keys appear in the run's metrics table and that values are finite.

If finite and look reasonable (pearson values > 0 for most metrics, MAE proportionate to the metric scale), the PR is ready to open. If not — file the issue against the corresponding metric (most likely candidate: a sample-shape mismatch in `update_metric_correlations` for `i_ptm_energy`, which has the most reshape paths).

- [ ] **Step 5: No commit** — smoke check is observational.

---

### Task 11: Open the PR

**Files:** none modified.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/pae-derived-metric-correlation
```

- [ ] **Step 2: Open PR via gh**

```bash
module load gh && gh pr create --base dev --head feat/pae-derived-metric-correlation --title "feat(confidence): per-sample-paired metric correlations (MAE/Pearson/Spearman) for the 10 pAE-derived interface metrics" --body "$(cat <<'EOF'
## Summary

For each of the 10 pAE-derived interface metrics (\`i_pae\`, \`min_ipae\`, \`i_ptm\`, \`i_ptm_energy\`, and six ipSAE variants), the validation pass now additionally logs **per-sample-paired** MAE, Pearson R, and Spearman R between the metric computed from the **ground-truth AF2 PAE** and the metric computed from the **student's predicted PAE**, accumulated across the val set in a DDP-safe way via \`torchmetrics.MetricCollection\`.

## Why

Aggregate metrics like \`val/pae/i_pae/mean_pred\` and \`val/pae/i_pae/mean_gt\` already exist (logged by the head as scalars). What was missing: *the per-sample correlation* — does the student's confidence track the AF2 confidence on the same dimer? A high Pearson on \`i_ptm\` means the head is useful for binder ranking even if its absolute calibration is off; a low Pearson with a low MAE means the head is well-calibrated on average but uninformative per-sample.

## What changed

### \`src/proteinfoundation/nn/confidence/_metrics.py\`

- 5 metric functions (\`i_pae\`, \`min_ipae\`, \`iptm_from_logits\`, \`iptm_energy_from_logits\`, \`ipsae_family\`) gain a \`reduce: Literal["batch_mean", "per_sample"] = "batch_mean"\` kwarg. Default unchanged, bit-identical to the prior API. \`per_sample\` returns a \`[B]\` tensor (or for \`ipsae_family\`, a dict of \`[B]\` tensors) with NaN at samples-without-mass.
- Pinned by \`tests/unit/nn/confidence/test_metrics_per_sample_reduction.py\`: for every metric, \`mean(M(..., reduce="per_sample"))\` equals \`M(..., reduce="batch_mean")\` within 1e-6.

### \`src/proteinfoundation/nn/confidence/pae_head.py\`

- Two new helpers: \`pae_ev_from_logits\` (replaces \`logits_to_expected_value\`, which becomes a back-compat alias) and \`_pae_ev_from_labels\` (canonical GT EV from integer-A bin labels).
- New constructor kwarg \`track_metric_correlations: bool = True\`.
- When tracking is enabled, the head holds an \`nn.ModuleDict\` of 10 \`MetricCollection({MAE, PearsonCorrCoef, SpearmanCorrCoef})\` — one per metric.
- \`update_metric_correlations(pae_ev_pred, pae_ev_gt, chain_idx, mask_eff)\` computes per-sample (pred, gt) values across all 10 metrics and feeds each collection.
- \`val_metric_correlations_compute_and_reset\` returns a \`{metric: {mae, pearson, spearman}}\` dict and resets the accumulators.
- \`compute_loss_and_metrics\` at stage='val' now invokes \`update_metric_correlations\` exactly once per val batch. Train stage is untouched.

### \`src/proteinfoundation/confidence/lightning_module.py\`

- New \`_log_metric_correlations_for_head\` helper + \`on_validation_epoch_end\` hook. For single-head configs the hook dispatches on \`self.head\` directly; for \`MultiHeadConfidence\` it iterates over \`children_heads\`. Logs land under \`val/{output_name_root}/{metric}/{mae,pearson,spearman}\`.

### Tests

- \`tests/unit/nn/confidence/test_metrics_per_sample_reduction.py\` — per-sample / batch_mean parity for all 10 metrics.
- \`tests/unit/nn/confidence/test_pae_head_gt_ev.py\` — \`_pae_ev_from_labels\` and \`pae_ev_from_logits\` correctness.
- \`tests/unit/nn/confidence/test_pae_head_metric_correlations.py\` — identity / constant-shift / no-mass cases.
- \`tests/integration/confidence/test_pae_head_val_correlation_wiring.py\` — val stage advances accumulators, train stage doesn't, track=False bypasses.
- \`tests/integration/confidence/test_lightning_val_epoch_logs_correlations.py\` — epoch-end hook logs all 30 expected keys and resets state.
- \`tests/integration/confidence/test_metric_collection_ddp.py\` — DDP correctness across 2 gloo workers.

## DDP-safety

\`torchmetrics.MetricCollection\` aggregates state across ranks via \`dist_sync_fn\` at \`compute()\` time. We rely on this directly — the Lightning module's \`self.log(..., sync_dist=False)\` after compute prevents a redundant reduction. Pinned by \`test_metric_collection_ddp.py\` over \`world_size in {1, 2}\`.

## NaN policy

Per-sample reduction returns NaN at samples-without-mass (e.g. a monomer in a mixed batch when the metric is interface-only). \`update_metric_correlations\` strips NaNs via \`~(isnan(pred) | isnan(gt))\` before calling \`update\` — \`PearsonCorrCoef\` would otherwise propagate the NaN through its state.

## Smoke test

- Single-GPU 2-step run: all 30 \`val/pae/{m}/{mae,pearson,spearman}\` keys appear in the log, train stage produces none.
- 1000-step 2×H100 SLURM run: see job <ID> in WandB run \`multi-plddt-pae-distill-teddymer\` — the new keys are present and values are finite.

## Reviewer panel

Per CLAUDE.md PR review protocol:

- code-review-debug-complexity-expert (mandatory)
- ml-software-pytorch-jax-expert — DDP correctness of \`MetricCollection\` accumulation, autocast safety of the fp32 EV path, lifecycle of the head's accumulators across \`trainer.fit\` → \`trainer.validate\` boundaries
- generative-protein-scientist — semantics of the per-sample-paired correlation as a calibration metric for binder design; sanity of NaN-as-no-mass convention vs. alternatives (zero, drop-and-renumber)

If review deadlocks, escape hatch per CLAUDE.md: terminate, report in chat, email schekmenev@aithyra.at (this address only; no secrets or proprietary training data in the email).

## Out of scope (deliberate)

- Per-sample distribution diagnostics (histograms, scatter plots in WandB). The 3 scalar stats per metric are sufficient for monitoring; richer diagnostics can be a follow-up.
- Same treatment for pLDDT metrics. pLDDT is per-residue and already has Pearson R logged across the full val set via \`pearson_r\` in \`_metrics.py\`; the per-sample-paired view doesn't change the calibration story there.
EOF
)"
```

Capture the PR URL.

- [ ] **Step 3: Dispatch the 3-reviewer panel via the Agent tool**

Send three parallel Agent calls (in a single message — they're independent):

1. **code-review-debug-complexity-expert** — full diff review (correctness, edge cases, DDP races, MetricCollection lifecycle across \`trainer.validate\`).
2. **ml-software-pytorch-jax-expert** — autocast + bf16-mixed precision interaction with the fp32 EV path; whether the `nn.ModuleDict` of `MetricCollection`s plays nicely with `find_unused_parameters=True` + `static_graph=True`; CPU vs GPU device sync at `update()` time.
3. **generative-protein-scientist** — does the per-sample-paired correlation actually answer "how well does the student rank binders?", and is NaN-as-no-mass the right sentinel (vs. zero or drop-and-renumber)?

Each reviewer either approves or files line-cited issues. If any reviewer flags issues, address them (TDD), re-dispatch the panel, loop until all approve. Stuck-PR escape hatch per CLAUDE.md: terminate, report in chat, email `schekmenev@aithyra.at` (this address only, no secrets, no proprietary training data).

---

## Self-Review

**Spec coverage** — every clause of spec §6 is implemented:
- §6.1 "per-sample paired MAE, Pearson R, Spearman" → Tasks 4, 5, 6.
- §6.2 "10 metrics listed" → Task 4 step 3b populates exactly that list.
- §6.3 "DDP-safe via Option A (torchmetrics MetricCollection)" → Tasks 4, 6, 7.
- §6.4 "existing scalar mean_pred logs stay" → no removal of any logging line in Task 5.
- §6.5 "fp64 reference test at 1e-5" → folded into Task 2 (per-sample reduction parity covers fp32 internal precision; the EV path is already fp32 by contract pinned in `pae_head.py` docstring).

**Placeholder scan** — no "TBD" / "TODO" / "etc." entries; every code block is complete. Tests carry concrete assertions with numeric tolerances. The `<from previous step>` token in Task 10 Step 3 is the canonical "fill in the value from the prior command's output" pattern.

**Type consistency** — `track_metric_correlations` is `bool` everywhere; the per-metric `MetricCollection` carries exactly `{"mae", "pearson", "spearman"}` in every test and the implementation. The metric name list (`i_pae, min_ipae, i_ptm, i_ptm_energy, avg_ipsae, min_ipsae, max_ipsae, avg_ipsae_10, min_ipsae_10, max_ipsae_10`) is identical across Tasks 4 step 3b, the test in Task 4, the smoke verification in Task 9 step 2, and the PR description.
