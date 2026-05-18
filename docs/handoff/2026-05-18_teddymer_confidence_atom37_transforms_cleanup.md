# Session handoff — Teddymer confidence yamls: prune over-inclusive `atom37_transforms`

**Date:** 2026-05-18
**Predecessor:** PR #9 (merged) + PR #10 (open) — see [[teddymer_pae_distillation_shipped]] in memory.
**Status:** PR #10 is open and review-approved. The fix below is a clean-up commit to land **on `prepare_teddy` after PR #10 merges** (or as an additional commit on the PR-#10 branch before merge — author's call).

## TL;DR for the next agent

The Teddymer confidence-distillation yamls (`distillation_teddymer_pae.yaml` and, by composition, `distillation_teddymer_multihead.yaml`) inherit `atom37_transforms` from `ted_dimers.yaml` that the confidence sidecar does NOT need. Specifically, the confidence trunk does not consume the frame fields (`rotations_gt`, `translations_gt`) produced by `OpenFoldFrameTransform`. The SwissProt-pLDDT path (`afdb_monomers_with_plddt.yaml`) ships with **zero** geometry transforms beyond the label transform — that's the right convention for confidence training, and the Teddymer path should mirror it.

Land a small cleanup commit that:
1. Removes `OpenFoldFrameTransform`, `GlobalRotationTransform`, and `ChainBreakPerResidueTransform` from `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml`.
2. Decides whether to keep `CoordsToNanometers` and `CenteringTransform` (see "Open questions" below).
3. Verifies the integration test (`test_teddymer_interface_cca_spot_check.py`) still passes — the cutoff `INTERFACE_CA_CUTOFF_A = 10.0` Å was calibrated against the *current* (centered, possibly nm-scaled) coords; changing the transform stack may shift it.

## What is on disk (verified)

- **Branch:** `teddymer_pr_b_pae_head_and_multi_head` (PR #10, open against `prepare_teddy`). Last commit `67fb423`.
- **PR-A merged into `prepare_teddy`** manually by the user.
- **Tests passing:** 186 / 186 repo-wide as of session close.
- **Yaml in question:** `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml`. The over-inclusive `atom37_transforms` list lives inside the `datamodule:` block.

Current `atom37_transforms` shipped (paste from the yaml):

```yaml
atom37_transforms:
  - _target_: proteinfoundation.datasets.transforms.CoordsToNanometers
  - _target_: proteinfoundation.datasets.transforms.GlobalRotationTransform
  - _target_: proteinfoundation.datasets.transforms.OpenFoldFrameTransform
  - _target_: proteinfoundation.datasets.transforms.ChainBreakPerResidueTransform
  - _target_: proteinfoundation.datasets.transforms.CenteringTransform
    center_mode: full
    data_mode: all-atom
  - _target_: proteinfoundation.datasets.transforms.AddPLDDTFromParentAFDB
    bin_width: 2.0
    max_bins: 50
    scale_max: 100.0
  - _target_: proteinfoundation.datasets.transforms.AddPAEFromParentAFDB
    bin_width: 0.5
    max_bins: 64
    scale_max: 31.75
```

## Evidence that the geometry transforms are unnecessary for the confidence path

Verified during the end-of-session Q&A:

1. **SwissProt-pLDDT path uses zero geometry transforms.** `configs/dataset/unified/afdb_monomers_with_plddt.yaml` only appends `AddPLDDTFromBFactor` on top of `afdb_monomers`, whose `atom37_transforms` block is empty. The pLDDT distillation has been training successfully without any frame, rotation, or nm-scaling transforms.
2. **Frame fields are not consumed downstream.** `grep -rn "rotations_gt\|translations_gt" src/proteinfoundation/confidence/ src/proteinfoundation/nn/confidence/` returns zero hits. Those fields are only consumed by `proteinfoundation.train` (the main generative-training entry point), not by `train_confidence`.
3. **The trunk's input is `(s, z, mask, cond)`.** `proteina.nn` constructs `s` and `z` internally from raw coords inside the frozen complexa pipeline; the confidence sidecar reads `trunk_intermediates` and passes them to the head. The head consumes `(s, z, mask, cond)` only.
4. **Origin of the over-inclusion.** During PR-B Slice 1 review (round 1, ml-protein-architect major #2), the original PR-A yaml was flagged for missing `atom37_transforms` entirely (smoke could not run). The fix-pass implementer reached for `ted_dimers.yaml`'s list as a template because it preserves token count; copying that subset was over-inclusive. `ted_dimers.yaml` is a *generative-training* dataset config — the frame fields belong there.

## What is NOT on disk yet (the gap)

- The cleaned-up yaml.
- A confirmation that the integration test still passes after the cleanup (and a short note on whether the `INTERFACE_CA_CUTOFF_A = 10.0` Å needs re-tuning if the coord transforms change).
- Optionally: a one-line yaml comment near the (now-shorter) `atom37_transforms` list saying "Confidence training does not need the generative pipeline's frame / rotation transforms; mirror `afdb_monomers_with_plddt.yaml`'s minimal posture."

## Architecture reference

- **The SwissProt counterpart** (the convention to mirror): [configs/dataset/unified/afdb_monomers_with_plddt.yaml](../../configs/dataset/unified/afdb_monomers_with_plddt.yaml). Note the empty `atom37_transforms` outside the single label transform.
- **The generative dimer config** (where the geometry transforms belong, not here): [configs/dataset/unified/ted_dimers.yaml](../../configs/dataset/unified/ted_dimers.yaml). Frame/rotation transforms are needed because the generative head's loss consumes them.
- **The trunk's actual input contract:** [src/proteinfoundation/nn/confidence/base.py](../../src/proteinfoundation/nn/confidence/base.py) — `ConfidenceTrunk.forward(s, z, mask, cond) -> (s, z)`. No frame consumption.
- **The Lightning module's forward path:** [src/proteinfoundation/confidence/lightning_module.py](../../src/proteinfoundation/confidence/lightning_module.py) `_forward` — invokes the frozen `proteina.nn(batch)` to get `trunk_intermediates`, then dispatches to the head. The frame fields are inert past this point.
- **The integration test calibrated against current coord conventions:** [tests/integration/test_teddymer_interface_cca_spot_check.py:60](../../tests/integration/test_teddymer_interface_cca_spot_check.py#L60) — `INTERFACE_CA_CUTOFF_A = 10.0`. If `CoordsToNanometers` is removed and the integration test reads coords in Å directly, the cutoff is already correct. If `CoordsToNanometers` stays and coords are in nm, the cutoff implicitly already accounts for that (the test was authored against the current stack). Re-verify after edits.

## PR plan — single cleanup commit

### Scope

One commit on `teddymer_pr_b_pae_head_and_multi_head` (or a new branch off `prepare_teddy` if PR #10 has merged by then). No test additions — the existing integration test is the contract.

### Test plan (TDD)

1. **Red phase**: NOT applicable — this is a deletion, not a feature add. The contract is "the integration test continues to pass; no new fields appear in the batch dict; the trunk forward still completes". Verify by running the existing tests BEFORE and AFTER the edit and confirming both are green.
2. **Green phase**: edit the yaml; run `python -m pytest tests/unit/datasets/ tests/integration/test_teddymer_interface_cca_spot_check.py tests/integration/test_pae_distillation_smoke.py tests/integration/test_multi_head_distillation_smoke.py -p no:warnings`. All must remain green.

### Implementation deliverables

- Edit `configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` to remove (at minimum) `OpenFoldFrameTransform`, `GlobalRotationTransform`, `ChainBreakPerResidueTransform`. See "Open questions" for whether `CoordsToNanometers` and `CenteringTransform` should also go.
- Confirm via Hydra compose that the config still instantiates cleanly:
  ```bash
  python -c "
  from hydra import compose, initialize_config_dir
  from pathlib import Path
  cfg_dir = Path('configs/dataset/unified').resolve()
  with initialize_config_dir(config_dir=str(cfg_dir), version_base='1.3'):
      cfg = compose(config_name='teddymer_with_plddt_and_pae')
      print('OK; transforms:', [t._target_.split('.')[-1] for t in cfg.datamodule.atom37_transforms])
  "
  ```
- One short yaml comment noting the rule (mirror SwissProt confidence-dataset posture).

### Reviewer panel

Mandatory (per CLAUDE.md):
- `code-review-debug-complexity-expert` — verify nothing downstream depends on the dropped fields.

Domain-add:
- `ml-protein-architect` — confirm the Hydra composition stays consistent with `afdb_monomers_with_plddt.yaml` and that the integration-test cutoff doesn't need re-tuning.

That's a 2-reviewer panel for this small cleanup. Larger panel is overkill for a deletion.

## Open questions for the user before starting

1. **Drop `CoordsToNanometers` and `CenteringTransform` too?** The SwissProt path doesn't use them, but the Teddymer integration test was calibrated against centered, possibly nm-scaled coords. Two options:
   - **(a) Drop all 5 geometry transforms** (full SwissProt-parity). The integration test's `INTERFACE_CA_CUTOFF_A = 10.0` may need re-checking — if Å-scale coords with no centering still place the inter-chain Cαs within ±2 of the metadata `interface_length`, ship it. If not, re-tune the cutoff (likely a small adjustment; the test samples 20 random dimers).
   - **(b) Drop only the three "definitely unused" ones** (`OpenFoldFrameTransform`, `GlobalRotationTransform`, `ChainBreakPerResidueTransform`). Keeps the centering + nm-scaling that the test was calibrated against. Safer; smaller diff. Recommended for the cleanup commit; option (a) can be a follow-up.
2. **Land on `teddymer_pr_b_pae_head_and_multi_head` (amend PR #10 before merge) or as a separate post-merge commit on `prepare_teddy`?** Author's call. Amending PR #10 keeps the history tight; a post-merge commit is simpler git mechanics.

## Reference paths (most-used)

- Yaml to edit: `/mnt/storage01/home/schekmenev/projects/complexa-flex/configs/dataset/unified/teddymer_with_plddt_and_pae.yaml`
- SwissProt counterpart (the convention): `/mnt/storage01/home/schekmenev/projects/complexa-flex/configs/dataset/unified/afdb_monomers_with_plddt.yaml`
- ted_dimers (where the geometry transforms belong, NOT here): `/mnt/storage01/home/schekmenev/projects/complexa-flex/configs/dataset/unified/ted_dimers.yaml`
- Confidence subsystem: `/mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/{confidence,nn/confidence}/`
- Integration test calibrated against current stack: `/mnt/storage01/home/schekmenev/projects/complexa-flex/tests/integration/test_teddymer_interface_cca_spot_check.py`
- PR #10 (PaeHead + MultiHeadConfidence): https://github.com/stanislav-chekmenev/complexa-flex/pull/10
- Teddymer view: `/mnt/storage01/home/schekmenev/data/teddymer_v1/`

## Project conventions reminder

- Branch off `prepare_teddy` (or amend `teddymer_pr_b_pae_head_and_multi_head` if PR #10 hasn't merged yet); PR into `dev` only after the whole Teddymer work wraps.
- TDD via subagents. `code-review-debug-complexity-expert` mandatory on every PR.
- No Claude attribution in commits or PRs.
- `gh` requires `module load gh` in the same Bash invocation.
- Pinned env: uv, Python 3.12, PyTorch 2.10 + CUDA 13, Hydra 1.3, Lightning ≥2.5 <2.6.
- Runtime artefacts (`ckpts/`, `wandb/`, sample/eval outputs) stay gitignored.
- After the cleanup ships, **archive both PR-A and PR-B plans** under `docs/archive/2026-05-18-teddymer-pae-distillation/` per `[[archiving-conventions]]` and `git rm` from `docs/superpowers/plans/`.
