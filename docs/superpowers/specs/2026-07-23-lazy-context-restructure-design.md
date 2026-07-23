# Lazy context restructure — CLAUDE.md + MEMORY.md

**Date:** 2026-07-23
**Branch:** `chore/lazy-context-restructure` (off `dev`)
**Tier:** T1 (docs-only; no training/eval/ckpt/DDP semantics touched)

## Problem

`/recall` consumed ~75% of the context window on session start. CLAUDE.md is ~203 lines / ~8K tokens and is loaded in full **every** session. The dominant bulk is the "Confidence-head distillation subsystem" block — ~10 dense invariant paragraphs — plus the search-gate, community-parity, and teddymer-filter sections. These are **path-specific**: relevant only when touching those subsystems, yet paid for on every turn.

The sibling repo `~/projects/seq2conf` already solves this: CLAUDE.md carries **cross-cutting invariants only** and points, one line each, to `docs/claude/*.md` files loaded lazily ("read the file for the path you are touching — do not load them all"). Its MEMORY.md is a bare index of one-line hooks; per-fact detail lives in individual memory files opened on demand.

## Goal

Adopt the seq2conf lazy pattern. Target: CLAUDE.md from ~8K down to ~3–3.5K always-loaded tokens, with **zero loss** of any current invariant — every rule survives either inline (if cross-cutting) or in a `docs/claude/*.md` (if path-specific).

## Design

### 1. New lazy-loaded `docs/claude/*.md` files (git-tracked)

Split "by subsystem" — one file per path you'd be editing:

| File | Content migrated out of CLAUDE.md |
| --- | --- |
| `docs/claude/confidence.md` | Confidence-head subsystem, **minus** the two-success-gate rule: package layout + layering rule, entry point, datasets (SwissProt / Teddymer), loss/metrics recipe, trunk hook + AdaLN + `expose_intermediates`, pair-repr symmetrisation, WandB logging fragment, rank-0 loguru gate, DDP `find_unused_parameters=true` + `static_graph=true`, CUDA allocator `expandable_segments`, resume-from-ckpt contract, `MultiHeadConfidence` wrapper, LayerNorm-bias-leak invariant. |
| `docs/claude/search.md` | Confidence-head provisional gate vs AF2 reporting gate (pLDDT 0.90 story + the immutable `DEFAULT_PROTEIN_BINDER_THRESHOLDS`), parity range-restriction corollary, confhead-evaluate = scRMSD-only, interface-OOD / native-frame. |
| `docs/claude/parity.md` | "Community-standard metric / reward parity" — verbatim-port rule, fp64 reference test, canonical in-tree references. |
| `docs/claude/data.md` | "Teddymer filter is geometry-only" selection-bias antipattern. |

Each file opens with a one-line scope statement so its `description`-equivalent is scannable.

### 2. Slimmed CLAUDE.md

**Retained inline (cross-cutting, every-session):** project-at-a-glance, uv-run ban, venv/compute-node staging, configs tree, runtime-artefact gitignore, session lifecycle (recall/ul/handoff), subagent roster, TDD workflow, PR-review protocol (tiers/panel/blocking), git conventions, style, reproducibility.

**Added:**
- A **"Context-window budget"** preamble (ported/adapted from seq2conf): orchestrate, don't accumulate — push broad search, large-file / many-file reads, and log triage into subagents that return compact conclusions; the main thread holds a small working set.
- A **"Load-bearing invariants (read lazily)"** section: one-line pointer per `docs/claude/*.md`, with the directive *"Read the file for the path you are touching — do not load them all. Skipping the relevant file before editing that path is a correctness risk."*

**Collapsed:** the confidence-head section shrinks from ~10 paragraphs to ~4 pointer lines.

**Updated guardrail:** the CLAUDE.md-editing / `/ul` rule gains a line — new load-bearing *path-specific* detail goes into the matching `docs/claude/*.md`, with only a one-line pointer added here (mirrors seq2conf).

### 3. MEMORY.md hook tightening

MEMORY.md stays a bare index. Compress the long multi-line hooks (`confhead_parity_range_restriction`, `teddymer_seqid_split_residual_leak`, `partition_netscratch_separation`, `confhead_bon_search_impl_status`) to one crisp line each. **Per-fact memory files are NOT edited** — full detail already lives in them and is loaded on recall.

### 4. Non-goals (YAGNI)

- No change to any per-fact memory file body.
- No change to the pre-existing large docs (`EVALUATION_METRICS.md`, `CONFIGURATION_GUIDE.md`, etc.) — they are already lazy (never auto-loaded).
- No new memory files.
- No code, config, or test changes.

## Verification

1. **No-invariant-lost diff.** Enumerate every rule/invariant currently in CLAUDE.md; confirm each appears either in slimmed CLAUDE.md or in a `docs/claude/*.md`. Nothing silently dropped.
2. **Every `docs/claude/*.md` is pointed to** from the new lazy section (no orphan file).
3. **Line/token count.** `wc -l CLAUDE.md` materially lower; rough token estimate (`wc -c`/4) ≈ 3–3.5K.
4. **MEMORY.md links still resolve** — each `[[slug]]` / `(file.md)` pointer matches an existing memory file.
5. **Markdown links in CLAUDE.md** resolve to real paths (spot-check the new `docs/claude/*.md` links).

## Bookkeeping

`docs/claude/` is tracked (as in seq2conf). Final commit on this branch: CLAUDE.md + the four new `docs/claude/*.md`, added by explicit path. MEMORY.md lives outside the repo (in `~/.claude/projects/.../memory/`) and is never committed. The spec + any handoff follow the standard `/ul` bookkeeping-commit rule.
