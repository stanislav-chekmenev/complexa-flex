"""Integration: `ConfidenceHeadScorer.score_native` on a fake trunk.

Wires a `MultiHeadConfidence` (plddt + pae) onto a fake `Proteina`
(the concat-features test's pattern) whose intermediates report
`n_ext == n_orig` (native single frame, `n_concat=0`) and runs
`score_native` end-to-end on a synthetic joint complex.

Asserts:
- output schema / shapes / dtypes,
- the batch-builder sets NO concat keys (`x_target` / `seq_target` /
  `target_mask`), keeping the trunk on the native path,
- a 2-valued `chain_idx` yields a finite ipae.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from proteinfoundation.confidence.inference_scorer import ConfidenceHeadScorer
from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.base import ConfidenceTrunk
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead


TOKEN_DIM = 32
PAIR_REPR_DIM = 16
DIM_COND = 16
NUM_PLDDT_BINS = 50
NUM_PAE_BINS = 64
LATENT_DIM = 8
TRUNK_EVAL_T = 0.99


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
    """Native single frame: `n_ext == n_orig` (no concat)."""

    def __init__(self, dim_cond: int, token_dim: int, pair_repr_dim: int) -> None:
        super().__init__()
        self.cond_factory = _FakeCondFactory(("bb_ca", "local_latents"), dim_cond)
        self.expose_intermediates = True
        self.embed = nn.Linear(1, token_dim, bias=False)
        self.embed_pair = nn.Linear(1, pair_repr_dim, bias=False)

    def forward(self, batch: dict) -> dict:
        assert "x_target" not in batch, "native path must not carry x_target"
        mask = batch["mask"]
        b, n = mask.shape
        ones = torch.ones(b, n, 1)
        s = self.embed(ones) * mask[..., None].to(ones.dtype)
        ones_pair = torch.ones(b, n, n, 1)
        z = self.embed_pair(ones_pair)
        pair_mask = (mask[:, None, :] & mask[:, :, None])[..., None].to(z.dtype)
        z = z * pair_mask
        local_latents = torch.zeros(b, n, LATENT_DIM, dtype=torch.float32)
        local_latents[:] = batch["x_t"]["local_latents"] * mask[..., None].to(
            local_latents.dtype
        )
        ca_coords = batch["coords_nm"][:, :, 1, :]
        return {
            "trunk_intermediates": {
                "s": s,
                "z": z,
                "mask": mask,
                "orig_mask": mask,
                "n_orig": int(n),
                "local_latents": local_latents,
                "ca_coords": ca_coords,
            }
        }


class _FakeFM:
    def __init__(self) -> None:
        self.data_modes = ("bb_ca", "local_latents")

    def corrupt_batch(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n = mask.shape
        x_1 = batch["x_1"]
        x_0 = {
            "bb_ca": torch.randn(b, n, 3),
            "local_latents": torch.randn(b, n, LATENT_DIM),
        }
        t = {"bb_ca": torch.rand(b), "local_latents": torch.rand(b)}
        x_t = {
            m: t[m][:, None, None] * x_1[m] + (1.0 - t[m][:, None, None]) * x_0[m]
            for m in self.data_modes
        }
        batch["x_0"] = x_0
        batch["x_1"] = x_1
        batch["x_t"] = x_t
        batch["t"] = t
        return batch

    def interpolate(self, x_0, x_1, t, mask=None):
        return {
            m: t[m][:, None, None] * x_1[m] + (1.0 - t[m][:, None, None]) * x_0[m]
            for m in x_1
        }


class _FakeAutoEncoder:
    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        self.latent_dim = latent_dim

    def encode(self, batch: dict) -> dict:
        mask = batch["mask"]
        b, n = mask.shape
        return {"z_latent": torch.zeros(b, n, self.latent_dim)}


class _FakeProteina(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nn = _FakeProteinaNN(DIM_COND, TOKEN_DIM, PAIR_REPR_DIM)
        self.fm = _FakeFM()
        self.autoencoder = _FakeAutoEncoder()
        self.cfg_exp = SimpleNamespace(
            product_flowmatcher=("bb_ca", "local_latents"),
        )


def _make_trunk() -> ConfidenceTrunk:
    return ConfidenceTrunk(
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
        latent_dim=LATENT_DIM,
    )


def _make_multi_head() -> MultiHeadConfidence:
    trunk = _make_trunk()
    plddt = PLDDTHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_plddt_bins=NUM_PLDDT_BINS,
    )
    pae = PaeHead(
        trunk=trunk,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        num_pae_bins=NUM_PAE_BINS,
        bin_min=0.0,
        bin_max=32.0,
        track_metric_correlations=False,
    )
    return MultiHeadConfidence(
        trunk=trunk,
        children={"plddt": plddt, "pae": pae},
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
    )


def _make_scorer() -> ConfidenceHeadScorer:
    module = ConfidenceDistillationModule.from_components(
        head=_make_multi_head(),
        proteina=_FakeProteina(),
        cond_modalities=("bb_ca", "local_latents"),
        trunk_eval_t=TRUNK_EVAL_T,
    )
    module.eval()
    return ConfidenceHeadScorer.from_module(module, trunk_eval_t=TRUNK_EVAL_T)


def _make_complex_prots(b: int = 2, l_per_chain: int = 5) -> dict:
    torch.manual_seed(0)
    l = 2 * l_per_chain
    coors = torch.randn(b, l, 37, 3) * 3.0
    residue_type = torch.randint(0, 20, (b, l))
    chain_index = torch.zeros(b, l, dtype=torch.long)
    chain_index[:, l_per_chain:] = 1
    mask = torch.ones(b, l, dtype=torch.bool)
    return {
        "coors": coors,
        "residue_type": residue_type,
        "chain_index": chain_index,
        "mask": mask,
    }


def test_build_confidence_batch_sets_no_concat_keys() -> None:
    scorer = _make_scorer()
    complex_prots = _make_complex_prots()
    batch = scorer._build_native_batch(complex_prots)

    for k in ("x_target", "seq_target", "seq_target_mask", "target_mask", "target_chains"):
        assert k not in batch, f"native batch must not set concat key {k!r}"
    assert batch["chain_idx"].unique().numel() == 2

    # Encoder / trunk feature factories require this full key set (regression
    # guard: score_native failed against the real checkpoint until all were
    # present with the right shapes).
    b, l = complex_prots["mask"].shape
    for k in ("coords", "coords_nm", "coors", "residue_type", "mask", "chain_idx", "chains", "coord_mask", "mask_dict"):
        assert k in batch, f"native batch missing encoder key {k!r}"
    assert batch["coords"].shape == (b, l, 37, 3)
    assert batch["coord_mask"].shape == (b, l, 37)
    # process_batch derives the residue mask as mask_dict['coords'][..., 0, 0],
    # so this must be [B, L, 37, 3] and its atom-0/coord-0 slice the residue mask.
    assert batch["mask_dict"]["coords"].shape == (b, l, 37, 3)
    assert torch.equal(batch["mask_dict"]["coords"][..., 0, 0].bool(), complex_prots["mask"].bool())
    # chains carries the 2-valued chain id (training convention, not a 0/1 mask).
    assert torch.equal(batch["chains"], batch["chain_idx"])


def test_score_native_schema_and_finite_ipae() -> None:
    scorer = _make_scorer()
    complex_prots = _make_complex_prots()
    out = scorer.score_native(complex_prots)

    assert set(out.keys()) == {"ipae", "complex_plddt", "provisional_success"}
    b = complex_prots["coors"].shape[0]
    assert out["ipae"].shape == (b,)
    assert out["complex_plddt"].shape == (b,)
    assert out["provisional_success"].shape == (b,)
    assert out["ipae"].dtype == torch.float32
    assert out["complex_plddt"].dtype == torch.float32
    assert out["provisional_success"].dtype == torch.bool
    assert torch.isfinite(out["ipae"]).all()
    assert (out["complex_plddt"] >= 0.0).all() and (out["complex_plddt"] <= 1.0).all()
