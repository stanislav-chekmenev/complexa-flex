# Boltz-1 (vendored)

Vendored subset of [Boltz-1](https://github.com/jwohlwend/boltz) needed by the
quality-graft pLDDT confidence head port.

## Provenance

- Upstream repository: <https://github.com/jwohlwend/boltz>
- License: MIT (Boltz-1) with embedded Apache 2.0 headers on files derived
  from OpenFold / AlQuraishi Laboratory / DeepMind Technologies Limited
  (preserved verbatim in each file).
- Vendored from local checkout at
  `/mnt/storage01/home/schekmenev/projects/quality-graft/src/boltz/` on
  2026-05-25.
- Upstream commit hash: unknown — re-fetched 2026-05-25 from quality-graft
  local checkout.

## Manifest

```
community_models/boltz/
  __init__.py
  README.md
  model/
    __init__.py
    layers/
      __init__.py
      attention.py             # AttentionPairBias
      dropout.py               # get_dropout_mask
      initialize.py            # weight init helpers
      transition.py            # Transition
      triangular_mult.py       # TriangleMultiplication{Outgoing,Incoming}
      triangular_attention/
        __init__.py
        attention.py           # TriangleAttention{Starting,Ending}Node
        primitives.py          # Linear / Attention / LayerNorm primitives
        utils.py               # permute_final_dims, flatten_final_dims, chunking
    modules/
      __init__.py
      pairformer.py            # PairformerModule + PairformerLayer (surgical extract)
```

## Surgical-extract notes

`community_models/boltz/model/modules/pairformer.py` is **not** a verbatim
copy of upstream `boltz/model/modules/trunk.py`. Upstream `trunk.py` bundles
`PairformerModule` + `PairformerLayer` together with `InputEmbedder`,
`MSAModule`, `MSALayer`, `DistogramModule`, and a top-level import of
`AtomAttentionEncoder` that drags in the entire `boltz.model.modules.encoders`
sub-tree. The extract:

1. Keeps `PairformerModule` and `PairformerLayer` verbatim (lines 424-653 in
   upstream `trunk.py`).
2. Drops the unused `AtomAttentionEncoder`, `OuterProductMean`, and
   `PairWeightedAveraging` imports (none of them are referenced by
   `PairformerLayer`).
3. Replaces `from boltz.data import const` + the single usage
   `const.chunk_size_threshold` with an inlined `_CHUNK_SIZE_THRESHOLD = 384`
   (the constant's only value upstream).
4. Guards the `fairscale.nn.checkpoint.checkpoint_activations.checkpoint_wrapper`
   import in `try/except ImportError`, so the module is importable without
   `fairscale`; an `ImportError` is raised at construction time only when
   `activation_checkpointing=True`.

## Import-rewrite policy

The only mechanical edits applied to copied files are `from boltz...` →
`from community_models.boltz...` (and the `import boltz.X` form).
No other semantic changes have been made; all license headers are preserved.

## Where this lives in the project

Used by `proteinfoundation.nn.confidence.qg_plddt_head` (see Phase 0 report
`docs/superpowers/specs/2026-05-25-quality-graft-head-port-investigation.md`).
