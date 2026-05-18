"""Contract tests for the PR-4 confidence losses.

`masked_plddt_cross_entropy` reduces by `sum(loss * mask) / mask.sum().clamp_min(1)` —
NOT plain `mean`. `masked_smooth_l1_on_expected_value` casts logits to fp32
before the softmax-weighted bin-center mean, so the expected-value loss is
numerically stable even under bf16 inputs. `combined_plddt_loss` is a strict
linear combination of the two with no hidden reduction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from proteinfoundation.confidence.losses import (
    combined_plddt_loss,
    masked_plddt_cross_entropy,
    masked_smooth_l1_on_expected_value,
)


def _bin_centers(num_bins: int = 50, bin_min: float = 0.0, bin_max: float = 100.0) -> torch.Tensor:
    bin_width = (bin_max - bin_min) / num_bins
    return torch.tensor(
        [bin_min + bin_width * (i + 0.5) for i in range(num_bins)],
        dtype=torch.float32,
    )


def test_masked_ce_matches_manual_reduction() -> None:
    torch.manual_seed(0)
    b, n, k = 2, 4, 5
    logits = torch.randn(b, n, k)
    labels = torch.randint(0, k, (b, n))
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    out = masked_plddt_cross_entropy(logits, labels, mask)

    per_residue = F.cross_entropy(
        logits.reshape(-1, k), labels.reshape(-1), reduction="none"
    ).reshape(b, n)
    manual = (per_residue * mask).sum() / mask.sum().clamp_min(1.0)
    assert torch.allclose(out, manual, atol=1e-6)
    plain_mean = per_residue.mean()
    assert not torch.allclose(out, plain_mean, atol=1e-6)


def test_masked_ce_empty_mask_finite() -> None:
    logits = torch.randn(1, 3, 4)
    labels = torch.zeros(1, 3, dtype=torch.int64)
    mask = torch.zeros(1, 3)
    out = masked_plddt_cross_entropy(logits, labels, mask)
    assert torch.isfinite(out)
    assert out.item() == 0.0


def test_masked_ce_label_smoothing_knob() -> None:
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 6)
    labels = torch.randint(0, 6, (2, 4))
    mask = torch.ones(2, 4)
    out0 = masked_plddt_cross_entropy(logits, labels, mask, label_smoothing=0.0)
    out1 = masked_plddt_cross_entropy(logits, labels, mask, label_smoothing=0.1)
    assert not torch.allclose(out0, out1, atol=1e-5)


def test_smooth_l1_on_expected_value_finite() -> None:
    torch.manual_seed(2)
    b, n, k = 2, 8, 50
    logits_bf = torch.randn(b, n, k, dtype=torch.bfloat16)
    targets = torch.rand(b, n) * 100.0
    mask = torch.ones(b, n)
    centers = _bin_centers(k)
    out = masked_smooth_l1_on_expected_value(logits_bf, targets, mask, centers)
    assert torch.isfinite(out)
    assert out.dtype == torch.float32


def test_combined_loss_decomposition() -> None:
    torch.manual_seed(3)
    b, n, k = 2, 6, 50
    logits = torch.randn(b, n, k)
    labels = torch.randint(0, k, (b, n))
    targets = torch.rand(b, n) * 100.0
    mask = torch.tensor([[1.0] * n, [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
    centers = _bin_centers(k)

    ce = masked_plddt_cross_entropy(logits, labels, mask, label_smoothing=0.0)
    sl1 = masked_smooth_l1_on_expected_value(logits, targets, mask, centers)

    total, parts = combined_plddt_loss(
        logits,
        labels,
        targets,
        mask,
        centers,
        ce_weight=0.9,
        smooth_l1_weight=0.1,
        label_smoothing=0.0,
    )
    expected = 0.9 * ce + 0.1 * sl1
    assert torch.allclose(total, expected, atol=1e-6)
    assert torch.allclose(parts["loss_ce"], ce, atol=1e-6)
    assert torch.allclose(parts["loss_smooth_l1"], sl1, atol=1e-6)
