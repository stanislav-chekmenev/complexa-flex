"""Off-by-default regression guard for the trunk `expose_intermediates` hook.

Targets Risk 6 of the PR-1 plan
(`docs/superpowers/plans/2026-05-16-pr1-trunk-internals-hook.md`): the
introduction of `expose_intermediates` must not perturb the legacy
`forward(input)` output. A single fixed-seed forward pass is snapshot at
`tests/fixtures/trunk_baseline_<sha>.pt` and compared byte-for-byte on every
run. `<sha>` is a short content hash over the trunk kwargs + seeded input, so
the fixture is auto-keyed if anything about the synthetic setup changes (the
guard remains active only when the test setup is unchanged).

Fixture regeneration mechanism
------------------------------
The fixture is generated once on `merge_quality_graft` HEAD before the trunk
hook lands, then committed to the branch. To regenerate:

    REGEN_FIXTURES=1 \\
        /path/to/python -m pytest \\
        tests/regression/test_flow_matching_loss_unchanged.py -v

If the fixture is absent and `REGEN_FIXTURES=1`, the test writes the snapshot
and `pytest.skip`s with a "fixture regenerated" message. Subsequent runs
without the env var must pass without skip — that is the actual regression
guard. If a legitimate trunk change is made later that intentionally alters
the FM output, delete the fixture, regenerate on the new HEAD, and update the
commit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from proteinfoundation.nn import local_latents_transformer as v1_module


TOKEN_DIM = 64
PAIR_REPR_DIM = 32
LATENT_DIM = 8
DIM_COND = 32
NLAYERS = 2
NHEADS = 4
SEED = 17
BATCH = 2
N = 12

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _trunk_kwargs() -> dict:
    return dict(
        nlayers=NLAYERS,
        token_dim=TOKEN_DIM,
        pair_repr_dim=PAIR_REPR_DIM,
        nheads=NHEADS,
        parallel_mha_transition=False,
        strict_feats=False,
        feats_seq=["xt_bb_ca", "xt_local_latents"],
        feats_cond_seq=["time_emb_bb_ca", "time_emb_local_latents"],
        feats_pair_repr=["rel_seq_sep"],
        feats_pair_cond=["time_emb_bb_ca"],
        dim_cond=DIM_COND,
        idx_emb_dim=DIM_COND,
        t_emb_dim=DIM_COND,
        seq_sep_dim=63,
        xt_pair_dist_dim=30,
        xt_pair_dist_min=0.1,
        xt_pair_dist_max=3.0,
        x_sc_pair_dist_dim=30,
        x_sc_pair_dist_min=0.1,
        x_sc_pair_dist_max=3.0,
        update_pair_repr=False,
        update_pair_repr_every_n=3,
        use_tri_mult=False,
        use_tri_attn=False,
        use_qkln=True,
        latent_dim=LATENT_DIM,
        output_parameterization={"bb_ca": "v", "local_latents": "v"},
        concat_features={
            "enable_motif": False,
            "enable_target": False,
            "enable_ligand": False,
            "motif_pair_features": False,
            "target_pair_features": False,
            "ligand_pair_features": False,
        },
    )


def _setup_hash() -> str:
    payload = json.dumps(
        {"kwargs": _trunk_kwargs(), "seed": SEED, "batch": BATCH, "n": N},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _build_and_run(expose: bool) -> dict:
    torch.manual_seed(SEED)
    mdl = v1_module.LocalLatentsTransformer(**_trunk_kwargs()).eval()
    mdl.expose_intermediates = expose

    g = torch.Generator().manual_seed(SEED + 1)
    inp = {
        "mask": torch.ones(BATCH, N, dtype=torch.bool),
        "x_t": {
            "bb_ca": torch.randn(BATCH, N, 3, generator=g),
            "local_latents": torch.randn(BATCH, N, LATENT_DIM, generator=g),
        },
        "t": {
            "bb_ca": torch.full((BATCH,), 0.5),
            "local_latents": torch.full((BATCH,), 0.5),
        },
    }
    with torch.no_grad():
        return mdl(inp)


def _fixture_path() -> Path:
    return FIXTURE_DIR / f"trunk_baseline_{_setup_hash()}.pt"


def _flatten_for_compare(out: dict) -> dict[str, torch.Tensor]:
    bb_ca_key, bb_ca_val = next(iter(out["bb_ca"].items()))
    ll_key, ll_val = next(iter(out["local_latents"].items()))
    return {
        "bb_ca_key": bb_ca_key,
        "bb_ca": bb_ca_val,
        "local_latents_key": ll_key,
        "local_latents": ll_val,
    }


def test_default_off_forward_bit_exact_against_baseline():
    out = _build_and_run(expose=False)
    flat = _flatten_for_compare(out)

    path = _fixture_path()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        if os.environ.get("REGEN_FIXTURES") == "1":
            torch.save(
                {
                    "bb_ca_key": flat["bb_ca_key"],
                    "bb_ca": flat["bb_ca"],
                    "local_latents_key": flat["local_latents_key"],
                    "local_latents": flat["local_latents"],
                },
                path,
            )
            pytest.skip(f"fixture regenerated at {path}")
        pytest.skip(
            f"baseline fixture {path} missing; run with REGEN_FIXTURES=1 once "
            "on merge_quality_graft HEAD then commit the fixture"
        )

    baseline = torch.load(path, weights_only=False)
    assert flat["bb_ca_key"] == baseline["bb_ca_key"]
    assert flat["local_latents_key"] == baseline["local_latents_key"]
    assert torch.equal(flat["bb_ca"], baseline["bb_ca"]), "bb_ca drifted from baseline"
    assert torch.equal(flat["local_latents"], baseline["local_latents"]), "local_latents drifted from baseline"


def test_expose_intermediates_does_not_perturb_fm_outputs():
    out_off = _build_and_run(expose=False)
    out_on = _build_and_run(expose=True)

    flat_off = _flatten_for_compare(out_off)
    flat_on = _flatten_for_compare(out_on)

    assert flat_off["bb_ca_key"] == flat_on["bb_ca_key"]
    assert flat_off["local_latents_key"] == flat_on["local_latents_key"]
    assert torch.equal(flat_off["bb_ca"], flat_on["bb_ca"])
    assert torch.equal(flat_off["local_latents"], flat_on["local_latents"])
