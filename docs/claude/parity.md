# Community-standard metric / reward parity

Read this file before any PR that touches the `_metrics.py` / `_losses.py` modules under
`nn/confidence/`, any future reward head (ipTM, iPLDDT, pAE_interaction, ipSAE), or any port of
a field-published confidence/reward number. This rule is a **force-T3 trigger** in the PR-review
protocol.

When porting any community-standard PAE/pLDDT-derived metric or reward — colabdesign's `get_ipsae_loss`, AlphaFold ipTM, RFdiff/BindCraft binder filters, AF3 confidence kernels, Boltz/Chai scores, or anything else the field publishes as a recognised number — port it **verbatim** in the first PR, with a fp64 reference test locking parity within `1e-5`.

- **Verbatim, not "improved".** Keep the upstream quirks: `1e-8` additives vs `clamp_min`, magic clip thresholds (e.g. ipSAE's `L >= 27`, ipTM's `n_eff >= 19`), two-direction reductions, per-row vs per-sample `d_0`. These quirks are part of the contract; deviating from them silently makes the logged values incomparable to the published binder-design literature (Watson et al., Bennett et al., Cao et al., Pacesa et al. / BindCraft, Yin et al.).
- **Reference lives in the test, not the implementation.** Write a fp64 numpy reproduction of the canonical upstream code inside the test file (e.g. [tests/unit/nn/confidence/test_ipsae_family.py](../../tests/unit/nn/confidence/test_ipsae_family.py)'s `_colabdesign_ipsae_reference`). The reference is itself reviewable — generative-protein-scientist must confirm it mirrors the upstream source line-by-line before the parity assertion is meaningful.
- **Generalisations come as separately-named follow-ups.** Want a Cα-distance-gated ipSAE variant or a soft-row ipTM aggregation? Ship it in a follow-up PR under a *new* name (`ipsae_contact_gated`, `iptm_soft`). Never reuse the canonical name for a different formulation, even if the new formulation is more principled.
- **Canonical references in-tree.** [community_models/colabdesign/af/loss.py](../../community_models/colabdesign/af/loss.py) is the colabdesign source of truth; [src/proteinfoundation/rewards/alphafold2_reward_utils.py](../../src/proteinfoundation/rewards/alphafold2_reward_utils.py) wraps the AF2-side primitives. Cite file:line when defending a port.

This rule binds every PR that touches the `_metrics.py` / `_losses.py` modules under `nn/confidence/` and any future reward heads (ipTM, iPLDDT, pAE_interaction, ipSAE).
