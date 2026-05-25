# PR #1 — Quality-graft confidence head port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the trainable backbone of `MultiHeadConfidence` (currently `ConfidenceTrunk` — pair-update layers + attention transitions) with quality-graft's recipe: one `AdaptorModule` (776→384 single, 256→128 pair, Cα-Cα distogram, 1 pair-biased attention block) followed by 4 Boltz-1 `PairformerLayer`s with triangular attention. Vendor the minimum Boltz-1 slice into `community_models/boltz/`. Expose `ca_coords` in `LocalLatentsTransformer.trunk_intermediates`. Retarget `PlddtHead` / `PaeHead` readouts at the new `(s=384, z=128)` dimensions. Ship a Teddymer multihead config + sbatch that mirror the existing one but swap the backbone.

**Architecture:**

1. **Vendor first, then port:** copy the Boltz-1 manifest verbatim from quality-graft's `src/boltz/model/` into `community_models/boltz/` (manifest fixed by the Phase 0 investigation report). Verify the layer instantiates standalone via a unit smoke test before touching anything downstream.
2. **Then trunk hook:** add `ca_coords` to `LocalLatentsTransformer.trunk_intermediates`. Verify the regression contract (existing flow-matching loss bit-identity) survives.
3. **Then adaptor:** port `AdaptorModule` to `src/proteinfoundation/nn/confidence/adaptor.py`. Drop the `hybrid` source-mode and the decoder-fusion path (complexa has no decoder analogue). Pin every projection's LayerNorm-bias-leak guard.
4. **Then dim swap:** add `d_in_token` / `d_in_pair_token` kwargs to `PLDDTHead` / `PaeHead` with backwards-compatible defaults (768 / 256). Existing single-head configs unaffected.
5. **Then backbone swap:** rewrite `MultiHeadConfidence` so it owns an `AdaptorModule` + `QgPairformerStack(n_layers=4)` instead of a `ConfidenceTrunk`. `ConfidenceTrunk` stays in-tree for single-head configs.
6. **Then config + sbatch:** add `distillation_teddymer_qg_multihead.yaml` + matching sbatch. Old `distillation_teddymer_multihead.yaml` stays in-tree for A/B.
7. **Smoke train, then panel review.**

The order is strict because every later step depends on the earlier one's contract being green. Each step ends in a commit.

**Tech Stack:** PyTorch 2.10 + CUDA 13, Lightning ≥2.5,<2.6, Hydra 1.3, torchmetrics. Pinned `.venv/` (no `uv run`). bf16-mixed precision. DDP with `find_unused_parameters=true` + `static_graph=true`. h100nvl × 2 nodes × 2 GPUs for the convergence smoke run.

**Reference spec:** [docs/superpowers/specs/2026-05-25-qg-head-port-and-pae-metric-correlations-design.md](../specs/2026-05-25-qg-head-port-and-pae-metric-correlations-design.md) §5.

**Reference Phase 0 report (output of plan 1):** `docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md` §2.3 fixes the Boltz-1 file manifest. **Do not start this plan until the Phase 0 report is committed.**

---

### Task 1: Create feature branch off `dev`

**Files:** none (git plumbing).

- [ ] **Step 1: Verify clean working tree on `dev`**

Run:
```bash
git status
git branch --show-current
```

Expected: `dev` checked out, clean working tree.

If dirty, stash or commit before proceeding. Do NOT use `git reset --hard`.

- [ ] **Step 2: Pull latest `dev`**

Run:
```bash
git pull --ff-only origin dev
```

Expected: fast-forward update or "Already up to date".

- [ ] **Step 3: Verify Phase 0 report is committed on `dev`**

Run:
```bash
git log --oneline -1 -- docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md
```

Expected: one commit on `dev` for the Phase 0 report. If empty, the Phase 0 plan hasn't finished — stop and complete plan 1 first.

- [ ] **Step 4: Create + check out `feat/qg-confidence-head`**

Run:
```bash
git checkout -b feat/qg-confidence-head
git branch --show-current
```

Expected: `feat/qg-confidence-head`.

---

### Task 2: Vendor Boltz-1 minimum slice into `community_models/boltz/`

**Files:**
- Create: `community_models/boltz/__init__.py` (empty)
- Create: `community_models/boltz/model/__init__.py` (empty)
- Create: `community_models/boltz/model/layers/__init__.py` (empty)
- Create: `community_models/boltz/model/modules/__init__.py` (empty)
- Create: `community_models/boltz/README.md` (provenance + license)
- Create: each file from the Phase 0 report's §2.3 manifest, e.g. (subject to manifest correction):
  - `community_models/boltz/model/layers/attention.py`
  - `community_models/boltz/model/layers/triangular_mult.py`
  - `community_models/boltz/model/layers/triangular_attention/__init__.py`
  - `community_models/boltz/model/layers/triangular_attention/primitives.py`
  - `community_models/boltz/model/layers/triangular_attention/attention.py`
  - `community_models/boltz/model/layers/transition.py`
  - `community_models/boltz/model/layers/dropout.py`
  - `community_models/boltz/model/modules/pairformer.py`

- [ ] **Step 1: Re-read the Phase 0 §2.3 manifest**

Run:
```bash
grep -nA 60 "Boltz-1 dependency manifest" docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md
```

The manifest is the authoritative list. Treat the bullet list above as a default; replace with the manifest list if it differs.

- [ ] **Step 2: Create the package skeleton**

Run:
```bash
mkdir -p community_models/boltz/model/layers/triangular_attention
mkdir -p community_models/boltz/model/modules
touch community_models/boltz/__init__.py
touch community_models/boltz/model/__init__.py
touch community_models/boltz/model/layers/__init__.py
touch community_models/boltz/model/layers/triangular_attention/__init__.py
touch community_models/boltz/model/modules/__init__.py
```

Expected: empty `__init__.py` files in place.

- [ ] **Step 3: Copy every file from the manifest verbatim**

For each file `<rel_path>` in the manifest, run:
```bash
cp /mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/<rel_path> community_models/boltz/<rel_path>
```

Do NOT edit the file contents at this step. Verbatim copy only.

- [ ] **Step 4: Rewrite `from boltz...` imports → `from community_models.boltz...`**

For each copied file, search for `from boltz` and `import boltz` lines:
```bash
grep -rn "from boltz\|import boltz" community_models/boltz/
```

For each hit, edit the import to point at `community_models.boltz` instead. Example edit:
```python
# Before
from boltz.model.layers.attention import AttentionPairBias
# After
from community_models.boltz.model.layers.attention import AttentionPairBias
```

This is the ONLY allowed semantic edit at this step. No other code changes.

- [ ] **Step 5: Add `community_models/boltz/README.md` with provenance**

Write the file content:
```markdown
# Vendored Boltz-1 minimum slice

This directory contains a minimal vendored slice of [Boltz-1](https://github.com/jwohlwend/boltz) — specifically the pairformer-layer machinery (attention with pair bias, triangle attention, triangle multiplication, transition) — used by `proteinfoundation.nn.confidence.MultiHeadConfidence`'s trainable backbone.

**Upstream provenance.** Files copied verbatim (with the single edit of rewriting `from boltz...` imports to `from community_models.boltz...`) from quality-graft's `src/boltz/model/` (commit `<COMMIT_HASH_FROM_PHASE0>`). Quality-graft itself sourced these files from the upstream [Boltz-1 repository](https://github.com/jwohlwend/boltz). See the Phase 0 investigation report (`docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md` §2.3) for the file manifest.

**License.** Boltz-1 is released under the MIT License. The MIT license text is preserved in each vendored file's header.

**Modification policy.** No semantic edits to vendored code. Re-vendor from upstream if Boltz-1 fixes a bug. The only edit applied at vendoring time is the import-path rewrite (`from boltz...` → `from community_models.boltz...`).
```

Replace `<COMMIT_HASH_FROM_PHASE0>` with the commit hash recorded in the Phase 0 report; if unavailable, write `unknown — re-fetched 2026-05-25 from quality-graft local checkout`.

- [ ] **Step 6: Verify package imports clean**

Run:
```bash
.venv/bin/python -c "from community_models.boltz.model.modules.pairformer import PairformerLayer; print(PairformerLayer)"
```

Expected: prints the class, no import error.

If an `ImportError` fires, it points at a missing transitive dependency. Read the offending line; if the import is for a Boltz-1 file NOT in the manifest, either (a) add that file to the manifest and re-copy, OR (b) confirm with the Phase 0 report whether the import path can be dropped (e.g. a non-pairformer utility that's never called from `PairformerLayer.forward`).

- [ ] **Step 7: Commit the vendor drop**

```bash
git add community_models/boltz/
git status
git commit -m "$(cat <<'EOF'
vendor: import boltz-1 minimum pairformer slice into community_models/

Copies AttentionPairBias, TriangleAttention*, TriangleMultiplication*,
Transition, dropout helpers, and PairformerLayer verbatim from
quality-graft's src/boltz/model/. The only edit is the import-path
rewrite (from boltz... -> from community_models.boltz...). See
community_models/boltz/README.md for upstream provenance.
EOF
)"
```

Expected: clean commit, no hooks failing.

---

### Task 3: Write + pass smoke test for `PairformerLayer`

**Files:**
- Create: `tests/unit/community_models/__init__.py` (empty)
- Create: `tests/unit/community_models/boltz/__init__.py` (empty)
- Create: `tests/unit/community_models/boltz/test_pairformer_layer_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/community_models/boltz/test_pairformer_layer_smoke.py`:

```python
"""Smoke test for the vendored Boltz-1 PairformerLayer.

Pins that the layer instantiates standalone, that forward shapes and
dtypes are preserved, and that the residue mask is honoured at padded
positions.
"""

from __future__ import annotations

import pytest
import torch

from community_models.boltz.model.modules.pairformer import PairformerLayer


def _make_inputs(B: int = 2, L: int = 16, s_dim: int = 384, z_dim: int = 128, dtype=torch.float32):
    s = torch.randn(B, L, s_dim, dtype=dtype)
    z = torch.randn(B, L, L, z_dim, dtype=dtype)
    # First sample: all valid. Second sample: last 4 residues padded.
    mask = torch.ones(B, L, dtype=dtype)
    mask[1, -4:] = 0.0
    return s, z, mask


def test_pairformer_layer_forward_shapes_and_dtype():
    layer = PairformerLayer(s_dim=384, z_dim=128, num_heads=16)
    s, z, mask = _make_inputs()
    s_out, z_out = layer(s=s, z=z, mask=mask)

    assert s_out.shape == s.shape
    assert z_out.shape == z.shape
    assert s_out.dtype == s.dtype
    assert z_out.dtype == z.dtype


def test_pairformer_layer_respects_mask_at_padded_positions():
    """Padded rows of `s` and padded rows/cols of `z` must be zero after a
    masked forward when the layer's contract is to multiply by `mask`."""
    torch.manual_seed(0)
    layer = PairformerLayer(s_dim=384, z_dim=128, num_heads=16)
    s, z, mask = _make_inputs()
    # Zero out the inputs at padded positions so the layer cannot read
    # them; the contract is that the output is also zero there.
    s = s * mask[..., None]
    z = z * mask[:, :, None, None] * mask[:, None, :, None]
    s_out, z_out = layer(s=s, z=z, mask=mask)

    pad_rows = mask[1] == 0
    assert torch.allclose(s_out[1, pad_rows], torch.zeros_like(s_out[1, pad_rows]), atol=1e-6)
    assert torch.allclose(z_out[1, pad_rows, :], torch.zeros_like(z_out[1, pad_rows, :]), atol=1e-6)
    assert torch.allclose(z_out[1, :, pad_rows], torch.zeros_like(z_out[1, :, pad_rows]), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pairformer_layer_forward_bf16_mixed_cuda():
    layer = PairformerLayer(s_dim=384, z_dim=128, num_heads=16).cuda()
    s, z, mask = _make_inputs()
    s, z, mask = s.cuda(), z.cuda(), mask.cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        s_out, z_out = layer(s=s, z=z, mask=mask)
    # Autocast: outputs may be bf16 or fp32 depending on the op; just
    # assert finite + shape preservation.
    assert torch.isfinite(s_out).all()
    assert torch.isfinite(z_out).all()
    assert s_out.shape == s.shape
    assert z_out.shape == z.shape
```

- [ ] **Step 2: Run the test to verify it fails (or passes — see expected outcome)**

Run:
```bash
.venv/bin/python -m pytest tests/unit/community_models/boltz/test_pairformer_layer_smoke.py -v
```

Expected:
- If the vendor drop was complete, this PASSES on first run (no implementation needed; the vendored layer is the implementation).
- If it FAILS with `ModuleNotFoundError` or `ImportError`, return to Task 2 step 4 (import rewrites) and Task 2 step 6 (closure check). Add the missing file to `community_models/boltz/` and re-run.
- If it FAILS with a `TypeError` on the `PairformerLayer(...)` constructor (different signature than `s_dim`, `z_dim`, `num_heads`), inspect the vendored signature and adjust the test — the test is the spec for our adapter usage; if the upstream signature differs, write a thin wrapper in `community_models/boltz/model/modules/pairformer.py` to expose the names we want, OR adjust the test to use the upstream names. Prefer adjusting the test (no semantic edits to vendored code).

- [ ] **Step 3: Make the test pass (if needed)**

If the test failed at Step 2, fix per the diagnosis there. Re-run until green.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/community_models/
git commit -m "$(cat <<'EOF'
test: smoke test for vendored boltz-1 PairformerLayer

Pins shape, dtype, and mask preservation on CPU; adds a bf16-mixed
autocast smoke when CUDA is available.
EOF
)"
```

---

### Task 4: Extend `LocalLatentsTransformer.trunk_intermediates` with `ca_coords`

**Files:**
- Modify: `src/proteinfoundation/nn/local_latents_transformer.py:325-355` (add `ca_coords_extended` to the `intermediates` dict)

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/confidence/test_trunk_intermediates_ca_coords.py`:

```python
"""Trunk-hook contract: `ca_coords` exposed in `trunk_intermediates`.

The quality-graft adaptor consumes a Cα-Cα distogram, so the trunk
must expose its predicted Cα coordinates alongside `s`, `z`,
`local_latents`. This test pins the contract that `expose_intermediates`
either exposes ALL of them or none — there is no partial state.
"""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer


def _build_transformer(expose_intermediates: bool) -> LocalLatentsTransformer:
    # The simplest valid LocalLatentsTransformer config is brittle to
    # construct by hand; reuse the test scaffolding already in the repo.
    from tests.unit.nn.test_local_latents_transformer_intermediates import (
        build_minimal_transformer,
    )

    return build_minimal_transformer(expose_intermediates=expose_intermediates)


def test_intermediates_expose_ca_coords_when_enabled():
    model = _build_transformer(expose_intermediates=True)
    B, n = 2, 16
    fake_input = _make_fake_input(B=B, n=n)
    out = model(fake_input)

    assert "trunk_intermediates" in out
    inter = out["trunk_intermediates"]
    assert "ca_coords" in inter
    # ca_coords lives in the same extended-length frame as `local_latents`.
    # Same first two dims as `local_latents`; last dim is 3 (xyz).
    assert inter["ca_coords"].shape[0] == inter["local_latents"].shape[0]
    assert inter["ca_coords"].shape[1] == inter["local_latents"].shape[1]
    assert inter["ca_coords"].shape[-1] == 3


def test_intermediates_absent_when_expose_intermediates_false():
    model = _build_transformer(expose_intermediates=False)
    B, n = 2, 16
    fake_input = _make_fake_input(B=B, n=n)
    out = model(fake_input)

    assert "trunk_intermediates" not in out


def _make_fake_input(B: int, n: int) -> dict:
    # Mirror the keys LocalLatentsTransformer.forward requires.
    # The minimal set is reused from the existing intermediates test.
    from tests.unit.nn.test_local_latents_transformer_intermediates import (
        build_minimal_input,
    )

    return build_minimal_input(B=B, n=n)
```

If the helpers `build_minimal_transformer` / `build_minimal_input` do not exist in the existing intermediates test, write inline equivalents — the test must be self-contained.

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest tests/integration/confidence/test_trunk_intermediates_ca_coords.py::test_intermediates_expose_ca_coords_when_enabled -v
```

Expected: FAIL with `KeyError: 'ca_coords'` or `AssertionError: 'ca_coords' not in ...`.

- [ ] **Step 3: Add `ca_coords_extended` to the intermediates dict**

Edit `src/proteinfoundation/nn/local_latents_transformer.py` to snapshot the pre-trim `ca_nm_out` and expose it alongside `local_latents_extended`.

At line 326 (`ca_nm_out = self.ca_linear(seqs) * mask[..., None]`), snapshot the pre-trim copy. At line 331 the file already does this for local_latents (`local_latents_extended = local_latents_out`); mirror the pattern.

Current code at 325–348:
```python
local_latents_out = self.local_latents_linear(seqs) * mask[..., None]  # [b, n_extended, latent_dim]
ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n_extended, 3]

# Snapshot pre-trim n_extended local_latents for the confidence-distill
# sidecar; consumers of `nn_out["local_latents"]` still see the trimmed
# `[b, n_orig, latent_dim]` tensor below.
local_latents_extended = local_latents_out

# Trim back to original sequence length (remove concat features) if we extended
if n_concat > 0:
    local_latents_out = local_latents_out[:, :n_orig, :] * orig_mask[:, :, None]

    ca_nm_out = ca_nm_out[:, :n_orig, :] * orig_mask[:, :, None]

s_out = seqs * mask[..., None]
z_out = pair_rep
intermediates = {
    "s": s_out,
    "z": z_out,
    "mask": mask,
    "orig_mask": orig_mask,
    "n_orig": int(n_orig),
    "local_latents": local_latents_extended,
}
```

Edit to:
```python
local_latents_out = self.local_latents_linear(seqs) * mask[..., None]  # [b, n_extended, latent_dim]
ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n_extended, 3]

# Snapshot pre-trim n_extended local_latents and ca_coords for the
# confidence-distill sidecar (quality-graft-style adaptor consumes
# ca_coords as a Cα-Cα distogram). Consumers of `nn_out["local_latents"]`
# and `nn_out["bb_ca"]` still see the trimmed `[b, n_orig, *]` tensors
# below.
local_latents_extended = local_latents_out
ca_coords_extended = ca_nm_out

# Trim back to original sequence length (remove concat features) if we extended
if n_concat > 0:
    local_latents_out = local_latents_out[:, :n_orig, :] * orig_mask[:, :, None]

    ca_nm_out = ca_nm_out[:, :n_orig, :] * orig_mask[:, :, None]

s_out = seqs * mask[..., None]
z_out = pair_rep
intermediates = {
    "s": s_out,
    "z": z_out,
    "mask": mask,
    "orig_mask": orig_mask,
    "n_orig": int(n_orig),
    "local_latents": local_latents_extended,
    "ca_coords": ca_coords_extended,
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest tests/integration/confidence/test_trunk_intermediates_ca_coords.py -v
```

Expected: PASS on both cases.

- [ ] **Step 5: Verify the regression contract still holds**

Run:
```bash
.venv/bin/python -m pytest tests/regression/test_flow_matching_loss_unchanged.py -v
```

Expected: PASS. The `expose_intermediates=False` default path must be bit-identical to before.

If FAIL, the edit accidentally changed behaviour on the default path — re-inspect the diff and ensure only the `intermediates = {...}` dict construction and the snapshot line were changed.

- [ ] **Step 6: Commit**

```bash
git add src/proteinfoundation/nn/local_latents_transformer.py tests/integration/confidence/test_trunk_intermediates_ca_coords.py
git commit -m "$(cat <<'EOF'
feat: expose ca_coords in LocalLatentsTransformer.trunk_intermediates

Mirrors local_latents_extended snapshot pattern. Required by the
quality-graft-style adaptor (Cα-Cα distogram input). The
expose_intermediates=False default path stays bit-identical.

Pins regression contract via existing test_flow_matching_loss_unchanged.
EOF
)"
```

---

### Task 5: Port `AdaptorModule` to `src/proteinfoundation/nn/confidence/adaptor.py`

**Files:**
- Create: `src/proteinfoundation/nn/confidence/adaptor.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/nn/confidence/test_adaptor_module.py`:

```python
"""Unit tests for the quality-graft-style adaptor.

Pinned properties:
- Output shapes match the Boltz-1 dim target (384 single, 128 pair).
- LayerNorm-bias-leak invariant: drifting LN.bias off zero does NOT
  contaminate valid positions through padded ones.
- Cα distogram: bins match a fp64 numpy reference.
- Single-attention-block residual init: with zero-init on attn.proj_o
  and the s/z MLP final linears, the block is a near-identity transform
  of the linearly-projected (s, z) within 1e-6.
- Translation equivariance under n_attn_layers=1: translating ca_coords
  by a constant leaves the adaptor output bit-identical (distogram uses
  pairwise distances; AttentionPairBias is not coordinate-aware).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from proteinfoundation.nn.confidence.adaptor import AdaptorModule


def _make_inputs(B: int = 2, n: int = 16, dtype=torch.float32):
    trunk_seqs = torch.randn(B, n, 768, dtype=dtype)
    trunk_pair = torch.randn(B, n, n, 256, dtype=dtype)
    local_latents = torch.randn(B, n, 8, dtype=dtype)
    ca_coords = torch.randn(B, n, 3, dtype=dtype)
    mask = torch.ones(B, n, dtype=dtype)
    mask[1, -4:] = 0.0  # padded tail on second sample
    return trunk_seqs, trunk_pair, local_latents, ca_coords, mask


def test_adaptor_output_shapes():
    adaptor = AdaptorModule()
    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()
    s, z = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)
    assert s.shape == (2, 16, 384)
    assert z.shape == (2, 16, 16, 128)


def test_adaptor_layernorm_bias_leak_guard():
    """Set every LN.bias in the adaptor to 0.3 (drift simulation); the
    output at non-padded positions must NOT depend on padded inputs."""
    adaptor = AdaptorModule()
    for m in adaptor.modules():
        if isinstance(m, torch.nn.LayerNorm) and m.bias is not None:
            with torch.no_grad():
                m.bias.fill_(0.3)

    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()
    s1, z1 = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)

    # Mutate the padded positions of every input on sample 1 (which has
    # 4 padded residues at the tail). Non-padded positions must be
    # bit-identical between s1/s2 and z1/z2.
    trunk_seqs2 = trunk_seqs.clone()
    trunk_pair2 = trunk_pair.clone()
    local_latents2 = local_latents.clone()
    ca_coords2 = ca_coords.clone()
    trunk_seqs2[1, -4:] = torch.randn_like(trunk_seqs2[1, -4:]) * 100.0
    trunk_pair2[1, -4:, :] = torch.randn_like(trunk_pair2[1, -4:, :]) * 100.0
    trunk_pair2[1, :, -4:] = torch.randn_like(trunk_pair2[1, :, -4:]) * 100.0
    local_latents2[1, -4:] = torch.randn_like(local_latents2[1, -4:]) * 100.0
    ca_coords2[1, -4:] = torch.randn_like(ca_coords2[1, -4:]) * 100.0

    s2, z2 = adaptor(trunk_seqs2, trunk_pair2, local_latents2, ca_coords2, mask=mask)

    valid = mask[1] == 1
    assert torch.allclose(s1[1, valid], s2[1, valid], atol=1e-6)
    # Pair: assert (valid, valid) sub-block is bit-identical
    valid_idx = torch.where(valid)[0]
    assert torch.allclose(
        z1[1][valid_idx][:, valid_idx],
        z2[1][valid_idx][:, valid_idx],
        atol=1e-6,
    )


def test_adaptor_ca_distogram_matches_fp64_reference():
    """The one-hot Cα distogram must match a fp64 numpy reference
    bin-for-bin."""
    torch.manual_seed(0)
    adaptor = AdaptorModule()
    n_bins = 128
    B, n = 1, 8
    ca_coords = torch.randn(B, n, 3, dtype=torch.float64)

    # fp64 reference
    diff = ca_coords[:, :, None, :] - ca_coords[:, None, :, :]
    dist = torch.linalg.norm(diff, dim=-1).numpy()
    bin_limits = np.linspace(0.1, 3.0, n_bins - 1)
    expected_bins = np.digitize(dist, bin_limits)  # 0..n_bins-1

    # Adaptor's internal binning
    got_one_hot = adaptor._binned_ca_distogram(ca_coords.to(torch.float32))
    got_bins = got_one_hot.argmax(dim=-1).numpy()

    np.testing.assert_array_equal(got_bins, expected_bins)


def test_adaptor_single_attn_block_near_identity_at_init():
    """At init, zero-init on attn.proj_o + s_mlp[1].weight + z_mlp[1].weight
    makes each AdaptorAttentionBlock a near-identity residual transform."""
    torch.manual_seed(0)
    adaptor = AdaptorModule(n_attn_layers=1)
    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()

    # Linear-only path (skip attn blocks) for comparison
    adaptor_linear = AdaptorModule(n_attn_layers=0)
    adaptor_linear.single_proj.load_state_dict(adaptor.single_proj.state_dict())
    adaptor_linear.pair_proj.load_state_dict(adaptor.pair_proj.state_dict())

    s_full, z_full = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)
    s_lin, z_lin = adaptor_linear(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)

    # Attention block's outputs are scaled by zero-init proj_o, so
    # adding them to s/z is a near-zero perturbation. Tight tol.
    assert torch.allclose(s_full, s_lin, atol=1e-5)
    assert torch.allclose(z_full, z_lin, atol=1e-5)


def test_adaptor_translation_equivariance():
    """Translating ca_coords by a constant leaves the output unchanged;
    pairwise distances are translation-invariant and AttentionPairBias
    is not coordinate-aware."""
    torch.manual_seed(0)
    adaptor = AdaptorModule(n_attn_layers=1)
    trunk_seqs, trunk_pair, local_latents, ca_coords, mask = _make_inputs()

    s1, z1 = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords, mask=mask)
    shift = torch.tensor([1.5, -2.0, 0.7])[None, None, :]
    s2, z2 = adaptor(trunk_seqs, trunk_pair, local_latents, ca_coords + shift, mask=mask)

    assert torch.allclose(s1, s2, atol=1e-6)
    assert torch.allclose(z1, z2, atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_adaptor_module.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'proteinfoundation.nn.confidence.adaptor'`.

- [ ] **Step 3: Port `AdaptorModule` to `src/proteinfoundation/nn/confidence/adaptor.py`**

Create the file:

```python
"""Quality-graft-style adaptor.

Adapts complexa trunk intermediates `(trunk_seqs[B,n,768],
trunk_pair[B,n,n,256], local_latents[B,n,8], ca_coords[B,n,3])` into
Boltz-1 confidence-head input dims `(s=384, z=128)`.

Architecture mirrors `quality_graft.models.adaptor.AdaptorModule` with
two simplifications:

- `source_mode` and the decoder-fusion path are dropped; complexa has
  no decoder analogue in the confidence-distill sidecar.
- `AttentionPairBias` is imported from the vendored Boltz-1 slice at
  `community_models.boltz.model.layers.attention`.

LayerNorm-bias-leak invariant. Every projection into a masked sequence
representation is followed by `* mask[..., None]` (for s) or
`* mask[:, :, None, None] * mask[:, None, :, None]` (for z), after the
LayerNorm. Drifted `LN.bias != 0` cannot contaminate valid positions
through padded ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from community_models.boltz.model.layers.attention import AttentionPairBias


class AdaptorAttentionBlock(nn.Module):
    """Pair-biased self-attention block in the Boltz-1 dim space."""

    def __init__(
        self,
        s_dim: int = 384,
        z_dim: int = 128,
        num_heads: int = 16,
    ) -> None:
        super().__init__()

        self.attn = AttentionPairBias(
            c_s=s_dim,
            c_z=z_dim,
            num_heads=num_heads,
            initial_norm=True,
        )
        self.silu = nn.SiLU()

        self.s_mlp = nn.Sequential(
            nn.LayerNorm(s_dim),
            nn.Linear(s_dim, s_dim, bias=False),
        )
        self.z_mlp = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, z_dim, bias=False),
        )

        nn.init.zeros_(self.s_mlp[1].weight)
        nn.init.zeros_(self.z_mlp[1].weight)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = s + self.attn(s=s, z=z, mask=mask)
        s = s + self.silu(self.s_mlp(s))
        z = z + self.silu(self.z_mlp(z))

        s = s * mask[..., None]
        z = z * mask[:, :, None, None] * mask[:, None, :, None]
        return s, z


class AdaptorModule(nn.Module):
    """Adapt complexa trunk intermediates to Boltz-1 confidence-head dims.

    Output: `(s[B,n,target_s_dim], z[B,n,n,target_z_dim])`. Cα coordinates
    enter via a one-hot pairwise distogram added to `z`.
    """

    def __init__(
        self,
        trunk_dim: int = 768,
        pair_dim: int = 256,
        latent_dim: int = 8,
        target_s_dim: int = 384,
        target_z_dim: int = 128,
        n_attn_layers: int = 1,
        num_heads: int = 16,
        ca_pair_dist_min: float = 0.1,
        ca_pair_dist_max: float = 3.0,
    ) -> None:
        super().__init__()
        self.n_attn_layers = n_attn_layers
        self.ca_pair_dist_min = ca_pair_dist_min
        self.ca_pair_dist_max = ca_pair_dist_max

        single_input_dim = trunk_dim + latent_dim  # 776
        self.single_proj = nn.Sequential(
            nn.LayerNorm(single_input_dim),
            nn.Linear(single_input_dim, target_s_dim, bias=False),
        )
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, target_z_dim, bias=False),
        )

        if n_attn_layers > 0:
            self.attn_blocks = nn.ModuleList(
                [
                    AdaptorAttentionBlock(
                        s_dim=target_s_dim,
                        z_dim=target_z_dim,
                        num_heads=num_heads,
                    )
                    for _ in range(n_attn_layers)
                ]
            )

    def _binned_ca_distogram(self, ca_coords: torch.Tensor) -> torch.Tensor:
        pair_dists = torch.norm(
            ca_coords[:, :, None, :] - ca_coords[:, None, :, :],
            dim=-1,
        )
        n_bins = self.pair_proj[1].out_features
        bin_limits = torch.linspace(
            self.ca_pair_dist_min,
            self.ca_pair_dist_max,
            n_bins - 1,
            device=ca_coords.device,
            dtype=ca_coords.dtype,
        )
        bin_indices = torch.bucketize(pair_dists, bin_limits)
        return F.one_hot(bin_indices, num_classes=n_bins).to(dtype=ca_coords.dtype)

    def forward(
        self,
        trunk_seqs: torch.Tensor,
        trunk_pair: torch.Tensor,
        local_latents: torch.Tensor,
        ca_coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(
                trunk_seqs.shape[:2], dtype=trunk_seqs.dtype, device=trunk_seqs.device
            )

        single_in = torch.cat([trunk_seqs, local_latents], dim=-1)  # [B, n, 776]
        s = self.single_proj(single_in)  # [B, n, target_s_dim]
        s = s * mask[..., None]

        z = self.pair_proj(trunk_pair)  # [B, n, n, target_z_dim]
        z = z + self._binned_ca_distogram(ca_coords)
        z = z * mask[:, :, None, None] * mask[:, None, :, None]

        if self.n_attn_layers > 0:
            for block in self.attn_blocks:
                s, z = block(s, z, mask)

        return s, z
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_adaptor_module.py -v
```

Expected: all five tests PASS.

If `test_adaptor_layernorm_bias_leak_guard` FAILS, an LN-bias leak path was missed — re-inspect every `nn.LayerNorm` and confirm a `* mask` follows every projection into `s` / `z`.

If `test_adaptor_translation_equivariance` FAILS, the attention block accidentally received coordinate input — confirm `AttentionPairBias` signature matches the vendored copy and consumes only `(s, z, mask)`.

- [ ] **Step 5: Commit**

```bash
git add src/proteinfoundation/nn/confidence/adaptor.py tests/unit/nn/confidence/test_adaptor_module.py
git commit -m "$(cat <<'EOF'
feat: add quality-graft-style AdaptorModule

Projects complexa trunk intermediates (s_768 + latents_8 -> s_384;
z_256 -> z_128 + Cα distogram) into Boltz-1 confidence-head dims.
One pair-biased self-attention block by default (n_attn_layers=1),
zero-init on residual paths so the block starts as near-identity.

LayerNorm-bias-leak invariant pinned by unit test.
EOF
)"
```

---

### Task 6: Add backwards-compatible `d_in_token` / `d_in_pair_token` kwargs to `PLDDTHead` / `PaeHead`

**Files:**
- Modify: `src/proteinfoundation/nn/confidence/plddt_head.py` (constructor + readout)
- Modify: `src/proteinfoundation/nn/confidence/pae_head.py` (constructor + readout)
- Create: `tests/unit/nn/confidence/test_head_dim_kwargs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/nn/confidence/test_head_dim_kwargs.py`:

```python
"""Pin backwards-compatible d_in_token / d_in_pair_token kwargs on heads.

Existing single-head configs construct PLDDTHead(token_dim=768, ...) and
PaeHead(pair_repr_dim=256, ...). The new kwarg must default to the same
values so those configs keep working unchanged. MultiHeadConfidence
constructs the heads with d_in_token=384 / d_in_pair_token=128.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.plddt_head import PLDDTHead
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.base import ConfidenceTrunk


def test_plddt_head_default_d_in_token_is_token_dim():
    trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    head = PLDDTHead(trunk=trunk, token_dim=768)
    # Default: readout reads from 768-dim s.
    s = torch.randn(2, 16, 768)
    mask = torch.ones(2, 16)
    out = head._predict(s, torch.randn(2, 16, 16, 256), mask)
    assert "plddt_logits" in out


def test_plddt_head_accepts_smaller_d_in_token():
    trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    head = PLDDTHead(trunk=trunk, token_dim=768, d_in_token=384)
    # Readout now reads from 384-dim s (the MultiHead backbone output).
    s = torch.randn(2, 16, 384)
    mask = torch.ones(2, 16)
    out = head._predict(s, torch.randn(2, 16, 16, 128), mask)
    assert "plddt_logits" in out
    assert out["plddt_logits"].shape == (2, 16, head.num_plddt_bins)


def test_pae_head_default_d_in_pair_token_is_pair_repr_dim():
    trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    head = PaeHead(trunk=trunk, pair_repr_dim=256)
    z = torch.randn(2, 16, 16, 256)
    mask = torch.ones(2, 16)
    out = head._predict(torch.randn(2, 16, 768), z, mask)
    assert "pae_logits" in out


def test_pae_head_accepts_smaller_d_in_pair_token():
    trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    head = PaeHead(trunk=trunk, pair_repr_dim=256, d_in_pair_token=128)
    z = torch.randn(2, 16, 16, 128)
    mask = torch.ones(2, 16)
    out = head._predict(torch.randn(2, 16, 384), z, mask)
    assert "pae_logits" in out
    assert out["pae_logits"].shape == (2, 16, 16, head.num_pae_bins)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_head_dim_kwargs.py -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'd_in_token'`.

- [ ] **Step 3: Add the kwarg to `PLDDTHead.__init__`**

Edit `src/proteinfoundation/nn/confidence/plddt_head.py`:

Add `d_in_token: int | None = None` to the constructor signature; in the constructor body, set `self.d_in_token = d_in_token if d_in_token is not None else token_dim` and use `self.d_in_token` instead of `token_dim` when constructing the final `Linear` readout. **Do not change the assert / docstring on `token_dim` —** the trunk's emitted `s` is still `token_dim`-wide; the kwarg only retargets the readout.

- [ ] **Step 4: Add the kwarg to `PaeHead.__init__`**

Edit `src/proteinfoundation/nn/confidence/pae_head.py`:

Add `d_in_pair_token: int | None = None` to the constructor signature; set `self.d_in_pair_token = d_in_pair_token if d_in_pair_token is not None else pair_repr_dim` and use `self.d_in_pair_token` when constructing the readout `Linear`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_head_dim_kwargs.py -v
```

Expected: all four PASS.

- [ ] **Step 6: Verify existing head tests are still green**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/ tests/integration/confidence/ -v
```

Expected: no regressions. All previously passing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/proteinfoundation/nn/confidence/plddt_head.py src/proteinfoundation/nn/confidence/pae_head.py tests/unit/nn/confidence/test_head_dim_kwargs.py
git commit -m "$(cat <<'EOF'
feat: add d_in_token / d_in_pair_token kwargs to PLDDTHead / PaeHead

Defaults preserve the existing token_dim / pair_repr_dim semantics so
single-head configs keep working unchanged. MultiHeadConfidence sets
d_in_token=384 and d_in_pair_token=128 to consume the quality-graft
backbone output.
EOF
)"
```

---

### Task 7: Create `QgPairformerStack` wrapper

**Files:**
- Create: `src/proteinfoundation/nn/confidence/qg_pairformer_stack.py`
- Create: `tests/unit/nn/confidence/test_qg_pairformer_stack.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/nn/confidence/test_qg_pairformer_stack.py`:

```python
"""Pin the 4-layer Boltz-1 pairformer stack wrapper.

Just a thin nn.Module that owns 4 PairformerLayer's and the mask
re-application after each layer. No semantic novelty; the test is mostly
shape preservation and the mask post-condition.
"""

from __future__ import annotations

import torch

from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack


def test_stack_default_4_layers():
    stack = QgPairformerStack()
    assert len(stack.layers) == 4


def test_stack_forward_shapes_and_mask():
    stack = QgPairformerStack(n_layers=4, s_dim=384, z_dim=128, num_heads=16)
    B, n = 2, 16
    s = torch.randn(B, n, 384)
    z = torch.randn(B, n, n, 128)
    mask = torch.ones(B, n)
    mask[1, -4:] = 0.0

    s = s * mask[..., None]
    z = z * mask[:, :, None, None] * mask[:, None, :, None]

    s_out, z_out = stack(s, z, mask)
    assert s_out.shape == s.shape
    assert z_out.shape == z.shape

    # Padded rows / cols zero
    assert torch.allclose(s_out[1, -4:], torch.zeros_like(s_out[1, -4:]), atol=1e-6)
    assert torch.allclose(z_out[1, -4:, :], torch.zeros_like(z_out[1, -4:, :]), atol=1e-6)
    assert torch.allclose(z_out[1, :, -4:], torch.zeros_like(z_out[1, :, -4:]), atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_qg_pairformer_stack.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `QgPairformerStack`**

Create `src/proteinfoundation/nn/confidence/qg_pairformer_stack.py`:

```python
"""Quality-graft-style stack of Boltz-1 PairformerLayer's.

Thin wrapper that owns N stacked `PairformerLayer` instances (from the
vendored Boltz-1 slice) and re-applies the residue mask after each
layer. Used as the trainable backbone of `MultiHeadConfidence` together
with the upstream `AdaptorModule`.

The layers are randomly initialised — no Boltz-1 checkpoint loading
(per design spec §3 non-goals).
"""

from __future__ import annotations

import torch
from torch import nn

from community_models.boltz.model.modules.pairformer import PairformerLayer


class QgPairformerStack(nn.Module):
    def __init__(
        self,
        n_layers: int = 4,
        s_dim: int = 384,
        z_dim: int = 128,
        num_heads: int = 16,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PairformerLayer(s_dim=s_dim, z_dim=z_dim, num_heads=num_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            s, z = layer(s=s, z=z, mask=mask)
            s = s * mask[..., None]
            z = z * mask[:, :, None, None] * mask[:, None, :, None]
        return s, z
```

If the vendored `PairformerLayer` constructor signature is `(c_s, c_z, ...)` rather than `(s_dim, z_dim, ...)`, adjust the call site here — the wrapper exposes the names we want even when the vendored layer uses different ones.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_qg_pairformer_stack.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/proteinfoundation/nn/confidence/qg_pairformer_stack.py tests/unit/nn/confidence/test_qg_pairformer_stack.py
git commit -m "$(cat <<'EOF'
feat: add QgPairformerStack — 4-layer boltz-1 pairformer wrapper

Thin nn.Module owning N stacked PairformerLayer's plus residue-mask
re-application after each layer. Backbone for the new
MultiHeadConfidence head; random init (no boltz-1 checkpoint).
EOF
)"
```

---

### Task 8: Swap `MultiHeadConfidence` backbone

**Files:**
- Modify: `src/proteinfoundation/nn/confidence/multi_head.py` (replace trunk with adaptor + stack)
- Create: `tests/unit/nn/confidence/test_multi_head_qg_backbone.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/nn/confidence/test_multi_head_qg_backbone.py`:

```python
"""Pin the quality-graft-style MultiHeadConfidence backbone.

Backbone: AdaptorModule -> QgPairformerStack(n_layers=4). Heads:
PLDDTHead(d_in_token=384), PaeHead(d_in_pair_token=128).

Outputs: {"plddt": {"plddt_logits": [B,L,n_plddt_bins]},
          "pae":   {"pae_logits":   [B,L,L,n_pae_bins]}}.
"""

from __future__ import annotations

import pytest
import torch
import warnings

from proteinfoundation.nn.confidence.adaptor import AdaptorModule
from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.nn.confidence.plddt_head import PLDDTHead
from proteinfoundation.nn.confidence.pae_head import PaeHead
from proteinfoundation.nn.confidence.base import ConfidenceTrunk


def _build_head() -> MultiHeadConfidence:
    # Children's `trunk` argument is required by BaseConfidenceHead.__init__
    # but unused in the qg backbone path. A small ConfidenceTrunk is
    # acceptable as a placeholder — MultiHeadConfidence doesn't dispatch
    # to it.
    placeholder_trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    plddt = PLDDTHead(
        trunk=placeholder_trunk,
        token_dim=768,
        d_in_token=384,
        num_plddt_bins=50,
        bin_min=0.0,
        bin_max=100.0,
        ce_weight=1.0,
        ev_weight=0.0,
        label_smoothing=0.0,
    )
    pae = PaeHead(
        trunk=placeholder_trunk,
        pair_repr_dim=256,
        d_in_pair_token=128,
        num_pae_bins=64,
        bin_min=0.0,
        bin_max=32.0,
        ce_weight=1.0,
        ev_weight=0.0,
        label_smoothing=0.0,
    )

    adaptor = AdaptorModule()
    backbone = QgPairformerStack(n_layers=4)

    return MultiHeadConfidence(
        adaptor=adaptor,
        backbone=backbone,
        children={"plddt": plddt, "pae": pae},
        loss_weights={"plddt": 1.0, "pae": 1.0},
    )


def _fake_intermediates(B: int = 2, L: int = 16) -> dict:
    return {
        "trunk_seqs": torch.randn(B, L, 768),
        "trunk_pair": torch.randn(B, L, L, 256),
        "local_latents": torch.randn(B, L, 8),
        "ca_coords": torch.randn(B, L, 3),
        "mask": torch.ones(B, L),
    }


def test_multi_head_forward_shapes():
    head = _build_head()
    inputs = _fake_intermediates(B=2, L=16)
    out = head(**inputs)
    assert set(out.keys()) == {"plddt", "pae"}
    assert out["plddt"]["plddt_logits"].shape == (2, 16, 50)
    assert out["pae"]["pae_logits"].shape == (2, 16, 16, 64)


def test_multi_head_zero_weight_emits_warning():
    placeholder_trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    plddt = PLDDTHead(trunk=placeholder_trunk, token_dim=768, d_in_token=384)
    pae = PaeHead(trunk=placeholder_trunk, pair_repr_dim=256, d_in_pair_token=128)
    with pytest.warns(RuntimeWarning, match=r"loss_weights\[\['pae'\]\] == 0\.0"):
        MultiHeadConfidence(
            adaptor=AdaptorModule(),
            backbone=QgPairformerStack(n_layers=4),
            children={"plddt": plddt, "pae": pae},
            loss_weights={"plddt": 1.0, "pae": 0.0},
        )


def test_multi_head_empty_output_name_root_raises():
    placeholder_trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    plddt = PLDDTHead(trunk=placeholder_trunk, token_dim=768, d_in_token=384)
    pae = PaeHead(trunk=placeholder_trunk, pair_repr_dim=256, d_in_pair_token=128)
    pae.output_name_root = ""
    with pytest.raises(ValueError, match="empty output_name_root"):
        MultiHeadConfidence(
            adaptor=AdaptorModule(),
            backbone=QgPairformerStack(n_layers=4),
            children={"plddt": plddt, "pae": pae},
        )


def test_multi_head_mismatched_trunk_eval_t_raises():
    placeholder_trunk = ConfidenceTrunk(token_dim=768, pair_repr_dim=256, n_blocks=1)
    plddt = PLDDTHead(trunk=placeholder_trunk, token_dim=768, d_in_token=384)
    pae = PaeHead(trunk=placeholder_trunk, pair_repr_dim=256, d_in_pair_token=128)
    pae.expected_trunk_eval_t = 0.5  # mismatch
    with pytest.raises(ValueError, match="expected_trunk_eval_t"):
        MultiHeadConfidence(
            adaptor=AdaptorModule(),
            backbone=QgPairformerStack(n_layers=4),
            children={"plddt": plddt, "pae": pae},
        )


def test_multi_head_runs_one_optim_step_with_static_graph():
    """Smoke: backward + step succeeds with static_graph=True. Single-process
    ddp_setup is not invoked; we only check loss.backward() doesn't error."""
    head = _build_head()
    inputs = _fake_intermediates(B=2, L=16)
    out = head(**inputs)
    # Simulated loss: sum of softmax-entropy of plddt + pae logits.
    loss = (
        torch.nn.functional.softmax(out["plddt"]["plddt_logits"], dim=-1).log().mean()
        + torch.nn.functional.softmax(out["pae"]["pae_logits"], dim=-1).log().mean()
    )
    loss.backward()
    # At least one parameter has a non-None grad.
    assert any(p.grad is not None for p in head.parameters())
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_multi_head_qg_backbone.py -v
```

Expected: FAIL — most likely `TypeError` on the new `MultiHeadConfidence(adaptor=, backbone=, ...)` signature.

- [ ] **Step 3: Rewrite `MultiHeadConfidence`**

Edit `src/proteinfoundation/nn/confidence/multi_head.py`. The new class owns an `AdaptorModule` and a `QgPairformerStack` instead of a `ConfidenceTrunk`. `forward` runs adaptor → stack once, dispatches `(s_out, z_out, mask)` to each child's `_predict`. The `expected_trunk_eval_t` parity assertion, `output_name_root` non-empty assertion, and `loss_weights == 0.0` `RuntimeWarning` stay exactly as before.

Replace the existing class body:

```python
"""Multi-head confidence wrapper (quality-graft-style backbone).

`MultiHeadConfidence` is itself a registered `BaseConfidenceHead`. It owns
an `AdaptorModule` (projects complexa trunk intermediates into Boltz-1
dims `(s=384, z=128)`) and a `QgPairformerStack` (4 Boltz-1 PairformerLayer's
with triangular attention). It holds an `nn.ModuleDict` of child heads,
runs the backbone exactly once per forward, and dispatches the same
refined `(s, z, mask)` to every child's `_predict`. Joint training is
opt-in via this wrapper; the per-head registry (`PLDDTHead`, `PaeHead`,
...) remains the primary path for single-head distillation through
`ConfidenceTrunk`.

Two layers of loss weighting:
- **Within-head** (`ce_weight` / `ev_weight` / `label_smoothing`) lives on
  each child's constructor.
- **Across-head** `loss_weights[name]` multiplies each child's total
  before summing into the wrapper aggregate. `loss_weights[name] == 0.0`
  is a hard short-circuit: the child's loss path is skipped entirely.

`expected_trunk_eval_t` parity is asserted at construction.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

from proteinfoundation.nn.confidence.adaptor import AdaptorModule
from proteinfoundation.nn.confidence.base import BaseConfidenceHead, StageLiteral
from proteinfoundation.nn.confidence.qg_pairformer_stack import QgPairformerStack
from proteinfoundation.nn.confidence.registry import register_confidence_head


@register_confidence_head("multi_head")
class MultiHeadConfidence(nn.Module):
    """Quality-graft-style multi-head wrapper.

    Note: deliberately not a subclass of `BaseConfidenceHead`. The base
    class' `__init__` requires a `ConfidenceTrunk`, which we no longer use.
    The Lightning module checks for `compute_multi_loss_and_metrics`
    (duck-typing), not isinstance.
    """

    output_keys: tuple[str, ...] = ()
    output_name_root: str = "multi"
    expected_trunk_eval_t: float = 0.99

    def __init__(
        self,
        adaptor: AdaptorModule,
        backbone: QgPairformerStack,
        children: dict[str, "BaseConfidenceHead"],
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        for name, child in children.items():
            if not child.output_name_root:
                raise ValueError(
                    f"MultiHeadConfidence: child head {name!r} has empty "
                    f"output_name_root; every child must declare a non-empty "
                    f"output_name_root for log-key prefixing."
                )
            if child.expected_trunk_eval_t != self.expected_trunk_eval_t:
                raise ValueError(
                    f"MultiHeadConfidence: child head {name!r} has "
                    f"expected_trunk_eval_t={child.expected_trunk_eval_t} which "
                    f"!= wrapper's {self.expected_trunk_eval_t}. Use a separate "
                    f"sidecar for heads needing a different t."
                )

        self.adaptor = adaptor
        self.backbone = backbone
        self.children_heads = nn.ModuleDict(children)
        self.output_keys = tuple(
            key for child in children.values() for key in child.output_keys
        )
        self.loss_weights = (
            {k: float(v) for k, v in loss_weights.items()}
            if loss_weights is not None
            else {name: 1.0 for name in children}
        )

        zero_weighted = [n for n, w in self.loss_weights.items() if w == 0.0]
        if zero_weighted:
            warnings.warn(
                f"MultiHeadConfidence: loss_weights[{zero_weighted}] == 0.0 disables "
                f"those heads' gradient paths. Under DDP with "
                f"find_unused_parameters=false this will raise at backward (each "
                f"child's _predict still runs in forward, so its parameters enter "
                f"the autograd graph). To run with a zero-weighted child:\n"
                f"  (a) remove the child from `children:` in the Hydra config, OR\n"
                f"  (b) set the trainer strategy to ddp_find_unused_parameters_true, "
                f"or wrap with DDP(static_graph=True) when the per-batch graph is fixed.",
                RuntimeWarning,
                stacklevel=2,
            )

    def forward(
        self,
        trunk_seqs: torch.Tensor,
        trunk_pair: torch.Tensor,
        local_latents: torch.Tensor,
        ca_coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        s, z = self.adaptor(
            trunk_seqs=trunk_seqs,
            trunk_pair=trunk_pair,
            local_latents=local_latents,
            ca_coords=ca_coords,
            mask=mask,
        )
        s, z = self.backbone(s, z, mask)
        return {
            name: head._predict(s, z, mask)
            for name, head in self.children_heads.items()
        }

    def compute_multi_loss_and_metrics(
        self,
        multi_out: dict[str, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
        masks_by_head: dict[str, torch.Tensor],
        *,
        stage: StageLiteral = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from proteinfoundation.nn.confidence._losses import MultiHeadLoss

        loss_fn = MultiHeadLoss(self.loss_weights)
        return loss_fn(
            multi_out=multi_out,
            heads=self.children_heads,
            batch=batch,
            masks_by_head=masks_by_head,
            stage=stage,
        )
```

This is a **breaking change to the public `MultiHeadConfidence` API**: the old `trunk=` kwarg is replaced with `adaptor=` + `backbone=`. The matching config edit is in Task 9.

- [ ] **Step 4: Adjust the Lightning module's dispatch path**

The sidecar Lightning module currently passes `(s, z, mask, cond, local_latents)` to `head(...)`. The new signature is `(trunk_seqs, trunk_pair, local_latents, ca_coords, mask)`. Read `src/proteinfoundation/confidence/lightning_module.py` to find the call site (likely in `training_step` / `_step_shared`).

Grep:
```bash
grep -n "self.head(" src/proteinfoundation/confidence/lightning_module.py
```

For the multi-head dispatch path, the call site must select the appropriate trunk-intermediate keys depending on whether the head is the qg-style `MultiHeadConfidence` (consumes `trunk_seqs`, `trunk_pair`, `local_latents`, `ca_coords`) or a legacy head (consumes `s`, `z`, `local_latents`).

Implementation: introduce a small dispatch helper that inspects `self.head` for an `.adaptor` attribute (or use `isinstance(self.head, MultiHeadConfidence)` after importing). For the qg-style head, build the kwargs dict from `trunk_intermediates`; for legacy heads, keep the existing kwargs. Both code paths run inside the same `training_step` body.

```python
# Inside _step_shared (or wherever the head is invoked):
inter = nn_out["trunk_intermediates"]
if isinstance(self.head, MultiHeadConfidence):
    head_out = self.head(
        trunk_seqs=inter["s"],
        trunk_pair=inter["z"],
        local_latents=inter["local_latents"],
        ca_coords=inter["ca_coords"],
        mask=inter["mask"],
    )
else:
    head_out = self.head(
        s=inter["s"],
        z=inter["z"],
        mask=inter["mask"],
        cond=cond,
        local_latents=inter["local_latents"],
    )
```

Where `MultiHeadConfidence` is imported at the top of `lightning_module.py`. This duck-typed branch keeps legacy single-head configs working unchanged.

- [ ] **Step 5: Run the tests**

Run:
```bash
.venv/bin/python -m pytest tests/unit/nn/confidence/test_multi_head_qg_backbone.py -v
.venv/bin/python -m pytest tests/unit/nn/confidence/ tests/integration/confidence/ -v
```

Expected: all PASS. No regressions on the previously-green tests.

If any legacy multi-head test FAILS (the old `MultiHeadConfidence(trunk=...)` signature), update it to use the new signature OR delete it if it pinned the now-removed `trunk-once invariant` (which is replaced by the adaptor + backbone-once invariant).

- [ ] **Step 6: Commit**

```bash
git add src/proteinfoundation/nn/confidence/multi_head.py src/proteinfoundation/confidence/lightning_module.py tests/unit/nn/confidence/test_multi_head_qg_backbone.py
git commit -m "$(cat <<'EOF'
feat: swap MultiHeadConfidence backbone to adaptor + boltz-1 stack

Replaces ConfidenceTrunk (pair-update + transformer transitions) with
AdaptorModule (776->384 single, 256->128 pair, Cα distogram) followed
by QgPairformerStack(n_layers=4). MultiHeadConfidence's signature
changes from trunk= to adaptor=+backbone=. Single-head configs are
unaffected — they keep using ConfidenceTrunk.

Lightning module duck-types on MultiHeadConfidence to dispatch the new
(trunk_seqs, trunk_pair, local_latents, ca_coords, mask) kwargs vs the
legacy (s, z, mask, cond, local_latents) kwargs.
EOF
)"
```

---

### Task 9: Add `distillation_teddymer_qg_multihead.yaml`

**Files:**
- Create: `configs/confidence/distillation_teddymer_qg_multihead.yaml`

- [ ] **Step 1: Read the existing multihead config**

Re-read `configs/confidence/distillation_teddymer_multihead.yaml` (already in context) to confirm the field set we need to preserve.

- [ ] **Step 2: Create the new config**

Write `configs/confidence/distillation_teddymer_qg_multihead.yaml`:

```yaml
# @package _global_
#
# Joint pLDDT + directional-PAE distillation on the Teddymer dimer view
# with the quality-graft-style backbone (adaptor + 4 boltz-1 pairformer
# layers). See docs/superpowers/specs/2026-05-25-qg-head-port-and-pae-
# metric-correlations-design.md §5 for the full design.

defaults:
  - /dataset/unified/teddymer_with_plddt_and_pae@data
  - /training/confidence_distill@training
  - /logging/wandb@logging
  - _self_

logging:
  wandb_tags: ["plddt", "pae", "teddymer", "multi", "qg"]

confidence:
  head:
    _target_: proteinfoundation.nn.confidence.multi_head.MultiHeadConfidence
    adaptor:
      _target_: proteinfoundation.nn.confidence.adaptor.AdaptorModule
      trunk_dim: 768
      pair_dim: 256
      latent_dim: 8
      target_s_dim: 384
      target_z_dim: 128
      n_attn_layers: 1
      num_heads: 16
      ca_pair_dist_min: 0.1
      ca_pair_dist_max: 3.0
    backbone:
      _target_: proteinfoundation.nn.confidence.qg_pairformer_stack.QgPairformerStack
      n_layers: 4
      s_dim: 384
      z_dim: 128
      num_heads: 16
    children:
      plddt:
        _target_: proteinfoundation.nn.confidence.plddt_head.PLDDTHead
        trunk:
          _target_: proteinfoundation.nn.confidence.base.ConfidenceTrunk
        token_dim: 768
        d_in_token: 384
        pair_repr_dim: 256
        num_plddt_bins: 50
        bin_min: 0.0
        bin_max: 100.0
        ce_weight: 1.0
        ev_weight: 0.0
        label_smoothing: 0.0
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
    loss_weights:
      plddt: 1.0
      pae: 1.0

trainer:
  _target_: lightning.pytorch.Trainer
  accelerator: gpu
  devices: 1
  num_nodes: 1
  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    find_unused_parameters: true
    static_graph: true
  precision: ${training.precision}
  max_epochs: ${training.max_epochs}
  gradient_clip_val: ${training.gradient_clip_val}
  log_every_n_steps: 15
  check_val_every_n_epoch: 1
  val_check_interval: 1000
  callbacks:
    - _target_: lightning.pytorch.callbacks.ModelCheckpoint
      dirpath: ${oc.env:RUN_DIR,runs/local}/checkpoints
      monitor: "val/multi/total"
      mode: "min"
      save_top_k: 3
      filename: "qg-multi-{epoch:03d}-{val/multi/total:.4f}"
      auto_insert_metric_name: false

data:
  datamodule:
    cluster_column: null
    cluster_seed: 42

seed: 42
run_name: "qg-multi-plddt-pae-distill-teddymer"
resume_id: null
resume_ckpt_path: null

integrity:
  enabled: true
  view_root: ${oc.env:TEDDYMER_VIEW_ROOT,/netscratch/schekmenev/teddymer_v1_blob}
  snapshot_path: /mnt/storage01/home/schekmenev/data/teddymer_v1/data.md5
```

Note: the children heads still receive a `trunk:` field because `BaseConfidenceHead.__init__` requires it. The trunk is a placeholder — `MultiHeadConfidence` ignores it and uses its own adaptor + backbone.

- [ ] **Step 3: Verify the config composes via Hydra**

Run:
```bash
.venv/bin/python -m proteinfoundation.confidence.train_confidence --config-name=confidence/distillation_teddymer_qg_multihead --cfg job --resolve 2>&1 | head -100
```

Expected: prints a composed config showing `confidence.head._target_: proteinfoundation.nn.confidence.multi_head.MultiHeadConfidence` and the adaptor/backbone/children sub-trees.

If composition errors, fix the YAML (missing `_target_`, indent, package mismatch) and re-run.

- [ ] **Step 4: Commit**

```bash
git add configs/confidence/distillation_teddymer_qg_multihead.yaml
git commit -m "$(cat <<'EOF'
config: add distillation_teddymer_qg_multihead.yaml

Multi-head pLDDT+PAE Teddymer distill config using the new
adaptor+boltz-1 backbone. The existing distillation_teddymer_multihead
config stays in-tree for A/B comparison.
EOF
)"
```

---

### Task 10: Add `scripts/train_confidence_teddymer_qg_multihead.sbatch`

**Files:**
- Create: `scripts/train_confidence_teddymer_qg_multihead.sbatch`

- [ ] **Step 1: Copy and edit the existing sbatch**

```bash
cp scripts/train_confidence_teddymer_pae.sbatch scripts/train_confidence_teddymer_qg_multihead.sbatch
```

- [ ] **Step 2: Edit the new sbatch**

Apply these edits to `scripts/train_confidence_teddymer_qg_multihead.sbatch`:

1. Change `--job-name=complexa_paedistill` → `--job-name=complexa_qgmulti`.
2. Change `--output=logs/paedistill_%j.out` → `--output=logs/qgmulti_%j.out`.
3. Change `--error=logs/paedistill_%j.err` → `--error=logs/qgmulti_%j.err`.
4. Change `--config-name=confidence/distillation_teddymer_pae` → `--config-name=confidence/distillation_teddymer_qg_multihead`.

Keep:
- The h100nvl partition, ntasks-per-node=2, nodes=1 defaults (override at sbatch-time for multi-node).
- The venv tarball staging (`/netscratch/schekmenev/venvs/complexa-flex/venv.tar.gz` with labs-NFS fallback).
- `PYTORCH_ALLOC_CONF=expandable_segments:True` (allocator-side mitigation; pair-update layers in the trunk forward still use reentrant ckpt — see CLAUDE.md).
- `NCCL_IB_DISABLE=1`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=0`, `NCCL_ASYNC_ERROR_HANDLING=1`.
- `unset WANDB_API_KEY`.
- `RESUME_CKPT_PATH` env-var → Hydra override wiring.
- `bf16-mixed` precision.

- [ ] **Step 3: Lint the sbatch syntactically**

Run:
```bash
bash -n scripts/train_confidence_teddymer_qg_multihead.sbatch
```

Expected: no output (clean parse).

- [ ] **Step 4: Commit**

```bash
git add scripts/train_confidence_teddymer_qg_multihead.sbatch
git commit -m "$(cat <<'EOF'
script: add sbatch for qg-style multihead teddymer distill

Mirror of train_confidence_teddymer_pae.sbatch with the new config name.
Same venv-tarball staging, PYTORCH_ALLOC_CONF + NCCL env, RESUME_CKPT_PATH
contract.
EOF
)"
```

---

### Task 11: Add the one-step smoke test for the qg multihead

**Files:**
- Create: `tests/smoke/__init__.py` (if not present)
- Create: `tests/smoke/test_qg_multihead_one_step.py`

- [ ] **Step 1: Write the smoke test**

```python
"""One-step smoke train of the qg-style multihead Teddymer distill.

Constructs the Lightning module + datamodule from the Hydra config,
runs trainer.fit for a single train step + one val pass on a 4-sample
micro-batch, asserts the headline log keys are present and finite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence.registry import build_confidence_head_from_cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not (REPO_ROOT / "ckpts" / "complexa.ckpt").exists(),
    reason="complexa.ckpt not staged for smoke",
)
def test_qg_multihead_one_step_smoke(tmp_path):
    os.environ["RUN_DIR"] = str(tmp_path / "run")
    os.environ["WANDB_MODE"] = "disabled"
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS_DIR)):
        cfg = compose(
            config_name="confidence/distillation_teddymer_qg_multihead",
            overrides=[
                "trainer.devices=1",
                "trainer.num_nodes=1",
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=1",
                "trainer.val_check_interval=1",
                "data.datamodule.batch_size=2",
                "integrity.enabled=false",
            ],
        )

    head = build_confidence_head_from_cfg(cfg.confidence.head)
    module = ConfidenceDistillationModule(
        head=head,
        trunk_ckpt_path=str(REPO_ROOT / "ckpts" / "complexa.ckpt"),
        autoencoder_ckpt_path=str(REPO_ROOT / "ckpts" / "complexa_ae.ckpt"),
        trunk_eval_t=cfg.training.trunk_eval_t,
        lr=cfg.training.opt.lr,
        weight_decay=cfg.training.opt.weight_decay,
        betas=tuple(cfg.training.opt.betas),
        warmup_steps=cfg.training.opt.warmup_steps,
        min_lr=cfg.training.opt.min_lr,
    )
    # The full datamodule construction reaches /netscratch; skip when
    # that path doesn't exist (local dev).
    if not Path("/netscratch/schekmenev/teddymer_v1_blob").exists():
        pytest.skip("Teddymer staged blob unavailable")

    import hydra
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=False)
    trainer.fit(module, datamodule=datamodule)

    # Pull logged metrics from the trainer.
    logged = trainer.callback_metrics
    for key in (
        "val/multi/total",
        "val/plddt/total",
        "val/pae/total",
    ):
        assert key in logged, f"Missing log key {key!r}; got {sorted(logged.keys())}"
        assert torch.isfinite(logged[key]).all()
```

- [ ] **Step 2: Run the smoke test**

Run:
```bash
.venv/bin/python -m pytest tests/smoke/test_qg_multihead_one_step.py -v -s
```

Expected: PASS in ≤ 10 minutes on a single h100. If the Teddymer staged blob is unavailable on the host running the test, the test is skipped (acceptable for local dev).

If FAIL with an OOM, the smoke is configured too wide — drop `data.datamodule.batch_size` to 1.

If FAIL with a NaN in `val/*/total`, the adaptor or stack is mis-initialised. Re-inspect zero-init paths in `AdaptorAttentionBlock` and `PairformerLayer`.

- [ ] **Step 3: Commit**

```bash
git add tests/smoke/test_qg_multihead_one_step.py
git commit -m "$(cat <<'EOF'
test: one-step smoke for qg-style multihead teddymer distill

Builds the head + lightning module from the new config, runs one train
step + one val pass on a 2-sample batch, asserts headline log keys are
present and finite. Skipped when CUDA, complexa.ckpt, or the Teddymer
staged blob is unavailable.
EOF
)"
```

---

### Task 12: Smoke-launch the qg multihead on h100 × 4 GPU for ≥ 1000 steps

**Files:** none (sbatch launch + WandB monitoring).

- [ ] **Step 1: Push the branch to origin**

```bash
git push -u origin feat/qg-confidence-head
```

- [ ] **Step 2: Launch the smoke train**

From a login node:
```bash
sbatch --nodes=1 --gres=gpu:h100nvl:4 --ntasks-per-node=4 \
  scripts/train_confidence_teddymer_qg_multihead.sbatch
```

The sbatch's `NUM_DEVICES=2` default in the script will be overridden by the Hydra-side `trainer.devices`; explicitly pass `trainer.devices=4` as an extra override OR edit the smoke sbatch to set `NUM_DEVICES=4` at the top.

For a tight 1000-step probe, pass `trainer.max_steps=1000` and `trainer.max_epochs=1` (whichever fires first wins).

- [ ] **Step 3: Monitor the WandB run**

Find the run in the `confidence-distillation` project. Check at step 1000:
- `val/plddt/pearson` — expect *higher trajectory* than the legacy multihead baseline (which started at 0.78).
- `val/pae/pearson` — also higher than baseline.
- No DDP errors ("marked as ready twice" / "parameters not used in producing the loss").
- No allocator OOMs.

- [ ] **Step 4: Decide go / no-go**

If `val/*/pearson` at step 1000 is *not* meaningfully higher than the baseline at the same step, the head rewrite has not solved the problem on its own. Options:
  a) Continue the run to 5000 steps and re-evaluate (the from-scratch boltz-1 init may take longer to find the right basin than quality-graft's pretrained init).
  b) Open a follow-up PR to load pretrained Boltz-1 pairformer weights (deferred per spec §9 out-of-scope).
  c) Re-open the Phase 0 leak audit with the new evidence.

Surface the decision to the user in chat. Do NOT auto-decide.

If `val/*/pearson` IS meaningfully higher (e.g. starts at ≤ 0.4 and is climbing past 0.6 by step 1000), the rewrite is on the expected trajectory and PR #1 is ready for review.

---

### Task 13: Open PR #1 with the 4-reviewer panel

**Files:** none (PR machinery).

- [ ] **Step 1: Verify the branch is up to date with `dev`**

Run:
```bash
git fetch origin
git log --oneline origin/dev..HEAD
git log --oneline HEAD..origin/dev
```

Expected: first command shows the PR commits; second shows zero (the branch has all of `dev`).

If the second command is non-empty, rebase:
```bash
git rebase origin/dev
```

Resolve any conflicts (likely none — the only file `dev` may have touched is `multi_head.py` / `pae_head.py` / `plddt_head.py`).

- [ ] **Step 2: Open the PR via `gh`**

Run:
```bash
module load gh && gh pr create --base dev --head feat/qg-confidence-head --title "feat: quality-graft confidence head port (adaptor + boltz-1 pairformer)" --body "$(cat <<'EOF'
## Summary
- Vendor minimum Boltz-1 pairformer slice into `community_models/boltz/` (AttentionPairBias, TriangleAttention*, TriangleMultiplication*, Transition, PairformerLayer). Verbatim copy with the single edit of import-path rewrites.
- Port `quality-graft`'s `AdaptorModule` to `src/proteinfoundation/nn/confidence/adaptor.py` (776→384 single, 256→128 pair, one-hot Cα-Cα distogram over [0.1, 3.0] nm with 128 bins, 1 pair-biased attention block at near-identity init).
- Add `QgPairformerStack(n_layers=4)` wrapper around 4 vendored `PairformerLayer`s with residue-mask re-application.
- Swap `MultiHeadConfidence` backbone from `ConfidenceTrunk` (the previous pair-update + transformer stack) to `AdaptorModule` + `QgPairformerStack`. Signature changes from `trunk=` to `adaptor=`+`backbone=`.
- Expose `ca_coords` in `LocalLatentsTransformer.trunk_intermediates` (pre-trim, mirrors `local_latents_extended`).
- Add backwards-compatible `d_in_token` / `d_in_pair_token` kwargs to `PLDDTHead` / `PaeHead`; default to the existing `token_dim` / `pair_repr_dim` so single-head configs are unaffected.
- Add `configs/confidence/distillation_teddymer_qg_multihead.yaml` + matching sbatch. The legacy `distillation_teddymer_multihead.yaml` stays in-tree for A/B.

## Test plan
- [ ] `tests/unit/community_models/boltz/test_pairformer_layer_smoke.py` — PASS
- [ ] `tests/unit/nn/confidence/test_adaptor_module.py` (5 properties, including LayerNorm-bias-leak guard + Cα distogram fp64 reference) — PASS
- [ ] `tests/unit/nn/confidence/test_qg_pairformer_stack.py` — PASS
- [ ] `tests/unit/nn/confidence/test_head_dim_kwargs.py` — PASS
- [ ] `tests/unit/nn/confidence/test_multi_head_qg_backbone.py` (forward shapes, RuntimeWarning on weight=0, expected_trunk_eval_t mismatch, output_name_root assertion, one optim step) — PASS
- [ ] `tests/integration/confidence/test_trunk_intermediates_ca_coords.py` — PASS
- [ ] `tests/regression/test_flow_matching_loss_unchanged.py` — still green (no regression on `expose_intermediates=False`)
- [ ] `tests/smoke/test_qg_multihead_one_step.py` — PASS
- [ ] 1000-step smoke train on 1 h100 node × 4 GPU runs without OOM, NCCL deadlock, DDP "marked as ready twice" / "parameters not used" errors

## Reviewer panel (per CLAUDE.md)
- @code-review-debug-complexity-expert (mandatory on every PR)
- @ml-protein-architect (confidence-distill subsystem layout + Hydra configs)
- @ml-software-pytorch-jax-expert (vendored triangular-attention modules, DDP + checkpoint interaction, dim swaps + memory budget)
- @generative-protein-scientist (architectural recipe is a generative-modeling design call)

All four must approve.

## Reference
- Design spec: `docs/superpowers/specs/2026-05-25-qg-head-port-and-pae-metric-correlations-design.md` §5
- Phase 0 investigation: `docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md`
EOF
)"
```

Expected: PR URL returned.

- [ ] **Step 2 alternative: If `gh` is not available, surface to user**

If `module load gh` fails or the `gh` command returns an error not related to the PR content, surface in chat: "Cannot open PR via gh — failed with <error>. PR title: '<title>'. PR body in clipboard / scratchpad. Please open manually."

- [ ] **Step 3: Dispatch the 4-reviewer panel in parallel**

Send a single message with four parallel `Agent` tool calls — one per reviewer. Each subagent prompt is self-contained, references the PR URL, and asks for either APPROVE or list-of-issues-with-file:line-and-severity.

Example reviewer prompt skeleton (one per agent type):
```
You are <agent role>. Review PR <URL> on the complexa-flex repo.

CONTEXT
-------
This PR replaces the trainable backbone of `MultiHeadConfidence` with
quality-graft's adaptor + boltz-1 pairformer recipe. See:
- docs/superpowers/specs/2026-05-25-qg-head-port-and-pae-metric-correlations-design.md
- docs/superpowers/plans/2026-05-25-pr1-qg-head-port.md
- docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md

YOUR REVIEW SCOPE
-----------------
<scope text per agent — e.g. ml-protein-architect focuses on Hydra config layout,
ml-software-pytorch-jax-expert on DDP / checkpoint / dim correctness, etc.>

OUTPUT
------
Either:
  APPROVE: <one-line justification>
Or:
  ISSUES:
    - file:line — severity (BLOCK / NIT) — description
    - ...
```

- [ ] **Step 4: Loop on issues**

For each panel round:
- Collect issues from all four reviewers.
- Group by severity; BLOCK issues must be addressed.
- Form a fix plan (use `software-planning-architect` subagent if the fix touches design, not just code).
- Execute the fix as TDD (write or update tests first; then implement).
- Push the fix to the same branch.
- Re-dispatch the same 4-reviewer panel.

Loop until all four APPROVE.

- [ ] **Step 5: Stuck-PR escape hatch**

If the panel does not converge after 3 rounds OR two reviewers disagree on direction, terminate the loop. Per CLAUDE.md:
1. Report the deadlock in chat.
2. Send a summary email to `schekmenev@aithyra.at` (no other addresses). Subject: "PR #1 qg-head-port deadlocked". Body: links + reviewer summary + the conflicting recommendations. **No secrets, no credentials, no proprietary training data in the email.**
3. Wait for the user to unblock.

- [ ] **Step 6: Merge to `dev`**

When all four reviewers APPROVE, the user is the one to merge — do NOT auto-merge. Surface the green panel status in chat and wait for the user to click Merge.

---

## Verification

PR #1 is mergeable when:

- All seven test files in Task 11's test plan pass on the pinned `.venv/`.
- `tests/regression/test_flow_matching_loss_unchanged.py` is still green.
- The 1000-step smoke train on 1 h100 node × 4 GPU completes without OOM, NCCL deadlock, or DDP errors.
- `val/plddt/pearson` trajectory at step 1000 is meaningfully different from the legacy multihead baseline (interpretation: spec §8 lists "smoke train without OOM/NCCL/DDP error" as the hard pass criterion; convergence quality is the *evaluation* criterion gating decision to keep the new config — not the merge criterion).
- All four reviewers APPROVE.
- The user clicks Merge.

When the PR is merged into `dev`, this plan is closed. The next plan (`2026-05-25-pr2-pae-metric-correlations.md`) opens, sequenced *after* this merge to avoid a `pae_head.py` conflict.
