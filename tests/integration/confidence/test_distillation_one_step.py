"""One-step distillation tests: loss finite, only head gets grads, trunk frozen.

Uses a fake `Proteina` stub (`nn.Module` with `.nn.cond_factory`,
`.nn.expose_intermediates`, and a forward returning a `nn_out` dict whose
`trunk_intermediates` key holds shape-correct random tensors). Real
`Proteina.load_from_checkpoint` is GPU-only and exercised in PR-5 smoke.
"""

from __future__ import annotations

import torch
from torch import nn

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16
NUM_BINS = 50


class _FakeCondFactory(nn.Module):
    def __init__(self, modalities: tuple[str, ...], dim_cond: int) -> None:
        super().__init__()
        self.modalities = modalities
        self.linear = nn.Linear(len(modalities), dim_cond, bias=False)

    def forward(self, batch: dict) -> torch.Tensor:
        t_stack = torch.stack([batch["t"][m] for m in self.modalities], dim=-1)
        mask = batch["mask"]
        b, n = mask.shape
        cond = self.linear(t_stack)
        cond = cond[:, None, :].expand(b, n, -1).contiguous()
        return cond * mask[..., None].to(cond.dtype)


class _FakeProteinaNN(nn.Module):
    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.cond_factory = _FakeCondFactory(("bb_ca", "local_latents"), dim_cond)
        self.expose_intermediates = True
        self.token_dim = token_dim
        self.pair_repr_dim = pair_repr_dim
        self.embed = nn.Linear(1, token_dim, bias=False)
        self.embed_pair = nn.Linear(1, pair_repr_dim, bias=False)

    def forward(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n = mask.shape
        ones = torch.ones(b, n, 1, device=mask.device)
        s = self.embed(ones) * mask[..., None].to(ones.dtype)
        ones_pair = torch.ones(b, n, n, 1, device=mask.device)
        z = self.embed_pair(ones_pair)
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None].to(z.dtype)
        z = z * pair_mask
        return {
            "trunk_intermediates": {
                "s": s,
                "z": z,
                "mask": mask,
                "orig_mask": mask,
                "n_orig": int(n),
            }
        }


class _FakeProteina(nn.Module):
    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(dim_cond, token_dim, pair_repr_dim)


def _make_head() -> PLDDTHead:
    trunk = ConfidenceTrunk(
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        n_blocks=2,
        n_heads=4,
        dim_cond=DIM_COND,
        use_tri_mult=True,
        use_tri_attn=False,
        use_qkln=True,
        dropout=0.0,
        update_pair_repr_every_n=1,
    )
    return PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_BINS,
        bin_min=0.0,
        bin_max=100.0,
    )


def _make_module() -> ConfidenceDistillationModule:
    head = _make_head()
    fake_trunk = _FakeProteina(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM)
    return ConfidenceDistillationModule.from_components(
        head=head,
        proteina=fake_trunk,
        cond_modalities=("bb_ca", "local_latents"),
        trunk_eval_t=0.99,
    )


def _make_batch(b: int = 2, n: int = 9) -> dict:
    torch.manual_seed(0)
    mask = torch.ones(b, n, dtype=torch.bool)
    plddt_mask = torch.ones(b, n, dtype=torch.bool)
    plddt_mask[0, n // 2 :] = False
    return {
        "mask": mask,
        "plddt_residue": torch.rand(b, n) * 100.0,
        "plddt_bin": torch.randint(0, NUM_BINS, (b, n)),
        "plddt_mask": plddt_mask,
    }


def test_one_step_loss_finite() -> None:
    mod = _make_module()
    batch = _make_batch()
    loss = mod.training_step(batch, batch_idx=0)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_only_head_params_get_grad() -> None:
    mod = _make_module()
    batch = _make_batch()
    loss = mod.training_step(batch, batch_idx=0)
    loss.backward()
    for p in mod.proteina.parameters():
        assert p.grad is None, "frozen proteina trunk must not collect gradients"
    head_trainable = [p for p in mod.head.parameters() if p.requires_grad]
    assert len(head_trainable) > 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head_trainable)


def test_trunk_frozen_grad_state() -> None:
    mod = _make_module()
    for p in mod.proteina.parameters():
        assert p.requires_grad is False
    assert mod.proteina.training is False
