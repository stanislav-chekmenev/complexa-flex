# Phase 0 — Quality-graft head port investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a single read-only investigation report that (a) inventories quality-graft's confidence-head + adaptor + Boltz-1 dependency stack with file:line citations, and (b) audits *the dataset construction and training-corpus differences between complexa and La-Proteina* — the leading hypothesis for the suspicious 0.78 starting Pearson R.

**User-supplied priors on the leak audit (binding):**

1. **AUDIT-4 (trunk_eval_t saturation) is EXCLUDED** — user has ruled this out. Do not investigate.
2. **AUDIT-1 (local_latents pLDDT-correlation via trunk training) is LOW PROBABILITY** — quality-graft uses essentially the same pipeline, so a trunk-internal leak should manifest there too. Investigate with low confidence; do not over-invest.
3. **NEW AUDIT-0 is the LEADING HYPOTHESIS:** the 0.78 floor comes from *complexa-specific* dataset construction and extra training data complexa was exposed to relative to La-Proteina. The complexa trunk was fine-tuned on data sources beyond La-Proteina's monomer corpus (potentially including PDB multimers, AFDB multimers, or other pLDDT-labelled sets); those extra sources may have given the trunk a head start on pLDDT-correlated structural features that La-Proteina's trunk lacks. Concretely: when the head is initialised, its `s` input from the complexa trunk already carries label-correlated geometry that quality-graft's La-Proteina-trunk-driven head does not see.

**Architecture:** One subagent dispatch (`code-review-debug-complexity-expert`, read-only, has Read/Grep/Glob/Bash/WebSearch/WebFetch). Subagent produces a single markdown report at the spec path. Main thread reviews the report and either opens a quick fix-bug PR (if the audit finds a separable leak) or proceeds straight to PR #1 (the qg-style head port).

**Tech Stack:** Bash (grep, find, file reading), no code writes, no environment touches.

**Reference spec:** [docs/superpowers/specs/2026-05-25-qg-head-port-and-pae-metric-correlations-design.md](../specs/2026-05-25-qg-head-port-and-pae-metric-correlations-design.md) §4.

---

### Task 1: Dispatch the investigation subagent

**Files:**
- Create (via subagent): `docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md`

- [ ] **Step 1: Verify the spec exists and is on `dev`**

Run:
```bash
git log --oneline -1 -- docs/superpowers/specs/2026-05-25-qg-head-port-and-pae-metric-correlations-design.md
```

Expected: one commit referencing the spec landing on `dev`. If absent, the spec was lost — stop and re-commit before dispatching the subagent.

- [ ] **Step 2: Verify the quality-graft repo is readable from this host**

Run:
```bash
ls /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/
ls /mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/model/layers/
ls /mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/model/modules/
```

Expected (from earlier exploration):

`quality_graft/models/` contains `adaptor.py`, `confidence_head.py`, `quality_graft.py`, `student_head.py`, `la_proteina_wrapper.py`, `__init__.py`.

`boltz/model/layers/` contains `attention.py`, `dropout.py`, `initialize.py`, `outer_product_mean.py`, `pair_averaging.py`, `transition.py`, `triangular_attention/`, `triangular_mult.py`.

`boltz/model/modules/` contains `confidence.py`, `confidence_utils.py`, `encoders.py`, `transformers.py`, `trunk.py`, `utils.py`.

If any of these paths is missing, the subagent cannot do its job — stop and surface the discrepancy.

- [ ] **Step 3: Dispatch the subagent**

Use the `Agent` tool with `subagent_type: "code-review-debug-complexity-expert"`. The full prompt below is self-contained — the subagent has no conversation history.

Prompt to the subagent (copy verbatim, replacing `<<PASTE_SPEC_PATH>>` with the absolute path):

```
You are a code-archeology and debugging subagent. Your job is to produce a single read-only investigation report that informs an upcoming architecture rewrite of the complexa-flex confidence-head subsystem.

PRIMARY DELIVERABLE
-------------------
Write a markdown report to:
  /mnt/storage01/home/schekmenev/projects/complexa-flex/docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md

Structure (every section is mandatory; do not collapse):
  1. Executive summary (≤ 200 words)
  2. Quality-graft architecture inventory
     2.1 AdaptorModule
     2.2 ConfidenceHead body
     2.3 Boltz-1 dependency manifest (list of files to copy)
     2.4 Loss / metrics recipe
     2.5 Training-loop knobs (LR, warmup, EMA, grad clip, AMP)
     2.6 Data pipeline differences vs complexa
  3. Complexa 0.78-floor leak audit
     3.1 Audit checklist with verdict + evidence per item
     3.2 Probability ranking of root cause
     3.3 Recommended diagnostic experiment (if any) to confirm before PR #1
  4. Open questions for main thread

Every factual claim MUST cite file:line. No prose-only assertions.

INPUTS YOU MUST READ
--------------------
Reference spec (your context for why this investigation exists):
  <<PASTE_SPEC_PATH>>
  In particular §4 (Phase 0).

THREAD A — Quality-graft inventory:
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/adaptor.py
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/confidence_head.py
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/student_head.py
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/quality_graft.py
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/models/la_proteina_wrapper.py
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/training/
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/data/
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/quality_graft/losses/
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/model/layers/
  /mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/model/modules/
  /mnt/storage01/home/schekmenev/projects/quality-graft/configs/

For Section 2.3 (Boltz-1 dependency manifest), use this procedure:
  - Read confidence_head.py and trace every `from boltz.model...` import.
  - Read each imported file and recursively trace its `from boltz...` imports.
  - The transitive closure is the manifest. Output as a bullet list of absolute paths under /mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/.
  - Annotate each file with the symbols imported (e.g. "attention.py: AttentionPairBias, AttentionPairBiasInitial").

For Section 2.4: confirm or correct from quality-graft's loss module: pLDDT loss recipe (CE / SmoothL1 split, weights, label smoothing), PAE loss recipe (if quality-graft distills PAE too), validation metrics computed and how they're aggregated.

For Section 2.5: open quality-graft's Lightning module and extract:
  - optimizer class + LR + weight decay + betas
  - warmup steps + LR schedule
  - EMA (if present)
  - gradient clip value
  - AMP precision (bf16-mixed, fp16, fp32?)
  - val_check_interval
  - DDP strategy (find_unused_parameters, static_graph, gradient_as_bucket_view)

For Section 2.6: compare quality-graft's dataset → label flow against complexa's. Specifically:
  - How does quality-graft compute pLDDT labels? Bin scale and width?
  - Any label transforms (e.g. AddPLDDTFromBFactor analogue)?
  - Does the trunk see ground-truth pLDDT anywhere on the input path?
  - What is quality-graft's `trunk_eval_t` (the fixed t at which it evaluates the trunk)?

THREAD B — Complexa 0.78-floor leak audit:
  /mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/confidence/base.py
  /mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/confidence/multi_head.py
  /mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/confidence/plddt_head.py
  /mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/confidence/pae_head.py
  /mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/nn/local_latents_transformer.py
  /mnt/storage01/home/schekmenev/projects/complexa-flex/src/proteinfoundation/confidence/lightning_module.py
  /mnt/storage01/home/schekmenev/projects/complexa-flex/configs/confidence/distillation_teddymer_multihead.yaml
  /mnt/storage01/home/schekmenev/projects/complexa-flex/configs/dataset/unified/teddymer_with_plddt_and_pae.yaml
  /mnt/storage01/home/schekmenev/projects/complexa-flex/scripts/train_confidence_teddymer_pae.sbatch

Audit checklist (Section 3.1) — each item must have a one-line verdict (LEAK / NO LEAK / UNCERTAIN) followed by evidence (file:line). PRIORITY ORDER reflects the user's binding priors: AUDIT-0 is the leading hypothesis; AUDIT-1 is LOW probability; AUDIT-4 is EXCLUDED.

  AUDIT-0 (LEADING HYPOTHESIS — investigate first, allocate ~60% of report depth):
  Does complexa's trunk-fine-tuning corpus include data the La-Proteina trunk did NOT see, and does that extra data carry pLDDT-correlated structural patterns that give the complexa trunk a "head start" on the label?

    Investigation steps:
      a) Fetch the La-Proteina paper (Geffner et al. 2025, arxiv 2507.09466) and the Proteina/Complexa paper(s) the project descends from. Use WebSearch for "La-Proteina atomistic flow matching" (arxiv id 2507.09466) and "Proteina protein backbone flow matching" / "Complexa atomistic flow matching" — the exact Complexa paper id is in the project README or the original Proteina release.
      b) From each paper, extract:
         - Training corpus (which datasets, how filtered, monomer vs multimer, AFDB vs PDB vs custom).
         - Sequence-cluster splits and filtering criteria.
         - Any data augmentation, distillation, or pretraining on AF-derived structures.
         - Resolution / pLDDT cutoffs applied during data curation.
      c) Diff: what does complexa's corpus include that La-Proteina's does NOT? Candidates to look for: AFDB multimers, AlphaFold-Multimer-derived dimers, PDB heterodimers, custom Teddymer training subset, anything pLDDT- or PAE-labelled at training time, augmentations that re-noise around the data manifold.
      d) Cross-check repo: grep `proteinfoundation/datasets/`, `configs/dataset/`, and any `train*.py` for dataset names that match the Complexa-paper corpus list. Confirm which checkpoints in `ckpts/complexa.ckpt` were trained on which.
      e) Reason about the leak: if complexa's trunk saw more pLDDT-labelled (or pLDDT-correlated, e.g. AFDB-derived) data than La-Proteina's, its per-residue features at trunk_eval_t (whatever that is — note we are NOT auditing the value, only its consequences) would already encode label-correlated geometry. Quality-graft's La-Proteina trunk, lacking that exposure, starts the head from a less-correlated representation, hence the 0.2 → 0.99 trajectory.
      f) Verdict: STRONG / WEAK / NO / UNCERTAIN evidence for "extra training data is the cause of the 0.78 floor". State which paper-derived facts are load-bearing.

  AUDIT-1 (LOW PROBABILITY — investigate with caution, ~10% report depth):
  Does `local_latents` carry information correlated with the ground-truth per-residue pLDDT label *for reasons independent of AUDIT-0*?

    Caveat (binding): quality-graft uses essentially the same pipeline shape (trunk → adaptor → boltz layers → head) and also consumes `local_latents`, yet does not show the 0.78 floor. So even if `local_latents` carries some label-correlated signal, that alone cannot be the dominant cause. Treat AUDIT-1 as "complete the audit for the record, but do not put weight on this hypothesis without a positive empirical signal".

    Investigation steps:
      a) Read local_latents_transformer.py:280-355 to confirm `local_latents = local_latents_linear(seqs)` is computed from the trunk's `seqs` after `nlayers` of transformer + pair-update layers.
      b) Cross-check: does La-Proteina's trunk emit a `local_latents`-equivalent? If yes (and quality-graft consumes it), this path is shared — verdict NO LEAK (or at least no *differential* leak).
      c) Verdict: LEAK / NO LEAK / UNCERTAIN. If LEAK, state explicitly why the same input doesn't produce the same floor in quality-graft.

  AUDIT-2: Does `CenteringTransform(full, bb_ca)` (configured in teddymer_with_plddt_and_pae.yaml) expose chain-level pLDDT correlations through Cα-COM recentering?
    Investigation steps:
      a) Grep `CenteringTransform` definition in the codebase. Cite which atom set determines the COM.
      b) Reason: if the COM is computed from one chain's Cα only (or weighted by something that correlates with confidence — e.g. pLDDT-weighted mean), the per-residue feature distribution becomes label-correlated.
      c) Cross-check quality-graft's centering policy if it has one — same caveat as AUDIT-1.
      d) Verdict: LEAK / NO LEAK / UNCERTAIN with evidence.

  AUDIT-3: Are GT pLDDT/PAE labels accidentally entering a feature dict the trunk consumes?
    Investigation steps:
      a) `grep -rn "plddt\|pae" configs/dataset/unified/teddymer_with_plddt_and_pae.yaml` — confirm `AddPLDDTFromBFactor` (or similar) populates `batch["plddt_bin"]` / `batch["plddt_residue"]` and NOTHING ELSE.
      b) Trace every read of `batch["plddt_*"]` / `batch["pae_*"]` in the codebase. The ONLY reader should be `PLDDTHead.compute_loss_and_metrics` / `PaeHead.compute_loss_and_metrics`. Anything else is a leak.
      c) Read `LocalLatentsTransformer.forward` and `ConfidenceTrunk.forward` and confirm neither consumes any pLDDT/PAE key from `batch`.
      d) Verdict: LEAK / NO LEAK / UNCERTAIN with evidence.

  AUDIT-4: EXCLUDED by user binding prior. Do not investigate.

  AUDIT-5: Cross-reference quality-graft (sanity check on AUDIT-0..3 verdicts).
    Investigation steps:
      a) Quality-graft consumes essentially the same set of trunk intermediates (trunk_seqs, trunk_pair, local_latents, ca_coords) at a comparable `t` and observed 0.2 → 0.99 Pearson R progression. The complexa baseline starts at 0.78.
      b) Therefore: any audit verdict that says LEAK along a path shared between the two trunks (architecture, transform stack, head wiring) must explain why quality-graft does not show the floor. If it cannot, the verdict must be downgraded.
      c) The most natural explanation that survives this constraint is AUDIT-0: complexa's trunk saw more / different data than La-Proteina's, so the head's `s` input is differentially label-correlated at init.
      d) Verdict: PROBABLE-CAUSE-IS-DATA-EXPOSURE-DIFF (AUDIT-0) / PROBABLE-COMPLEXA-SPECIFIC-LEAK-IN-HEAD-WIRING (AUDIT-2 or AUDIT-3) / PROBABLE-NO-CORRECTABLE-CAUSE / UNCERTAIN.

For Section 3.2 (probability ranking): rank the surviving audit items (AUDIT-0, AUDIT-1, AUDIT-2, AUDIT-3, AUDIT-5) by your estimated probability of being the root cause. The user's prior is that AUDIT-0 is the leading candidate; if your evidence supports that, say so explicitly. If your evidence contradicts the prior, state what contradicts it and which audit you would re-rank to the top.

For Section 3.3: state the minimum diagnostic experiment to run BEFORE PR #1 lands. If you find no leak and recommend no experiment, say so explicitly.

For Section 4: list any questions you could not answer from the repo state alone that the main thread should resolve before PR #1.

CONSTRAINTS
-----------
- Read-only. No file writes outside the report itself. No `git commit`. No environment touches (`uv` / venv).
- Wall-time budget: 90 minutes. If you run out of time, write what you have and end Section 4 with "TIME-OUT after <minutes>; remaining unfinished items: <list>".
- Every factual claim cites file:line. Bare prose claims will block PR #1.

When the report is complete, output one final message to the main thread:
  "INVESTIGATION COMPLETE. Report at <path>. Top finding: <one-sentence-summary-of-Section-3-verdict>."

DO NOT write any other file. DO NOT commit anything. DO NOT edit existing code.
```

- [ ] **Step 4: Wait for subagent completion**

The Agent tool returns when the subagent finishes. Do not poll, do not sleep.

Expected return: a one-message summary from the subagent confirming the report was written and the top-line finding.

- [ ] **Step 5: Verify the report exists and is non-empty**

Run:
```bash
test -f docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md && wc -l docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md
```

Expected: file exists, line count > 100 (a real report; a stub is < 50 lines).

If line count is suspiciously low, read the report — it may have hit the wall-time budget. If it's incomplete in a way that blocks PR #1 (no Boltz-1 manifest in Section 2.3), re-dispatch with a tightened prompt focusing on the missing section.

---

### Task 2: Review the report and decide next step

**Files:**
- Read: `docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md`

- [ ] **Step 1: Read the report end-to-end**

Use the Read tool. Pay particular attention to:
- Section 2.3 (Boltz-1 dependency manifest) — this becomes PR #1's `community_models/boltz/` file list.
- Section 3.1 audit verdicts.
- Section 3.2 probability ranking.
- Section 3.3 recommended diagnostic experiment (if any).
- Section 4 open questions.

- [ ] **Step 2: Surface Section 4 open questions to the user**

If Section 4 lists any open question that blocks PR #1 (e.g. "the manifest is incomplete because file X imports from a Boltz-1 module not present in the repo — main thread must decide whether to skip or vendor"), surface them in a single concise message to the user. Wait for user resolution.

If Section 4 is empty or all questions are non-blocking, skip to Step 3.

- [ ] **Step 3: Decide the next-step routing**

Three routes; pick exactly one based on Section 3.1 verdicts:

  ROUTE A — **No leak found, proceed to PR #1.** If every audit item is NO LEAK or UNCERTAIN-without-recommended-experiment, the rewrite proceeds. The leak audit's findings (negative) are referenced in PR #1's description.

  ROUTE B — **Leak found, separable from head rewrite, fix-bug PR first.** If any audit item is LEAK and the recommended diagnostic in Section 3.3 is a small fix (e.g. "the dataset config exposes `batch['plddt_residue']` to the trunk via a transform — remove that transform"), open a fix-bug PR before PR #1.

  ROUTE C — **Leak found, recommended experiment is exploratory.** If Section 3.3 recommends "run a smoke experiment to confirm" rather than a code fix, propose the experiment to the user and wait for their decision (run it now? defer? proceed to PR #1 anyway?).

State the chosen route + one-sentence justification in the chat. Do not start PR #1's plan until the user acknowledges the route.

- [ ] **Step 4: Commit the report under the docs/ subtree per the session-lifecycle convention**

The investigation report is a durable artefact. Commit it on `dev`.

```bash
git add docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md
git status
```

Expected status: one new file, no other staged changes.

```bash
git commit -m "$(cat <<'EOF'
docs: phase 0 investigation report for qg head port

Read-only inventory of quality-graft confidence head + boltz-1
dependency manifest, plus a leak audit of complexa's confidence
trunk to explain the 0.78 starting Pearson R on the multihead
Teddymer run.
EOF
)"
```

Expected: clean commit on `dev`, no hooks failing.

---

## Verification

Phase 0 is complete when:

- `docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md` exists, is committed to `dev`, has all four numbered sections populated.
- Section 2.3 contains an explicit list of Boltz-1 file paths to vendor.
- Section 3.1 has a verdict + evidence for AUDIT-0, AUDIT-1, AUDIT-2, AUDIT-3, AUDIT-5. AUDIT-4 is explicitly marked EXCLUDED (no investigation, no verdict).
- AUDIT-0 receives substantially more report depth than AUDIT-1..3, reflecting the user's binding prior.
- The main thread has chosen one of Route A / B / C and surfaced it to the user.

When the user acknowledges the routing decision, this plan is closed. The next plan (`2026-05-25-pr1-qg-head-port.md`) opens, parameterised by Route A/B/C.

**Auto-mode note (binding for this session):** the user has activated auto-mode and authorised self-approval of all three plans. Once Phase 0 produces a report, route the result without waiting for user acknowledgement. The user will redirect if the route is wrong.
