"""Inference-time confidence scorer for best-of-N binder search.

Labels each generated binder candidate's *provisional success* using the
trained confidence head (frozen Complexa trunk + multi-head distilling AF2
pLDDT and pairwise PAE), so best-of-N search does not need an AF2 fold to
decide which candidates to keep.

Best-of-N does not steer, so this is a *scorer*, not a reward: it emits a
per-sample interface ipAE and complex pLDDT, and a boolean provisional
success ``(ipae < 7 A) & (complex_plddt > 0.9)`` (2-of-3 AlphaProteo proxy;
the head cannot produce scRMSD).

Frame invariant (load-bearing): the head was distilled on co-diffused native
dimers -- both chains as ordinary diffused residues in a single unextended
frame (``n_concat == 0``), cross-chain pair block from the trunk's normal
``pair_repr_builder``. Binder generation instead injects the target as concat
features (extended ``[binder | target]`` axis, cross-block from
``ConcatPairFeaturesFactory``), which the head never saw. Scoring on that
concat frame is out-of-distribution for the head's interface PAE. So the
primary scorer (:meth:`score_native`) re-presents the generated complex in
the *native* frame the head trained on: both chains as regular residues, no
``x_target``. The cheap concat-latent reuse (:meth:`score_concat_reuse`) is
kept only as an uncalibrated ablation signal for a parity study against AF2.
"""

from __future__ import annotations

import os

import torch

from proteinfoundation.confidence.lightning_module import ConfidenceDistillationModule
from proteinfoundation.nn.confidence._metrics import i_pae, interface_pair_mask
from proteinfoundation.nn.confidence.multi_head import MultiHeadConfidence
from proteinfoundation.utils.coors_utils import ang_to_nm
from proteinfoundation.utils.sample_utils import add_clean_samples

# AlphaProteo success thresholds, applied here to the confidence head's own
# outputs.  ipAE is compared in Angstroms directly (the head's PAE expected
# value is already in A; there is no x31 rescale -- that factor only applies to
# AF2's normalised i_pAE column, see result_analysis/binder_analysis_utils.py).
# pLDDT is compared on the 0-1 scale, so the head's 0-100 expected value is
# divided by 100 before the test.
SUCCESS_IPAE_ANGSTROM = 7.0
SUCCESS_PLDDT_01 = 0.9
PLDDT_EV_SCALE = 100.0


class ConfidenceHeadScorer:
    """Score generated binder complexes with the trained confidence head.

    Args:
        ckpt_path: Path to a trained ``ConfidenceDistillationModule`` (multi-head)
            checkpoint. The Lightning checkpoint carries the frozen trunk, but we
            reuse the live generation ``proteina`` to avoid loading trunk weights
            twice.
        proteina: The already-loaded generation ``Proteina`` (its ``.nn`` trunk,
            ``.fm`` flow matcher, ``.autoencoder`` are reused by the head).
        trunk_eval_t: Interpolation time the trunk is evaluated at (the regime the
            head was trained for). Must match the checkpoint's ``trunk_eval_t``.
        head_config_name: Hydra config (under ``configs/``) whose
            ``confidence.head`` block defines the multi-head architecture the
            checkpoint was trained with. The head is an ``nn.Module`` and is NOT
            reconstructed from the checkpoint's hparams, so we instantiate a fresh
            head of the right shape and let ``load_from_checkpoint`` populate its
            weights (same contract as the ckpt round-trip test).
        device: Optional device to move the module to.
    """

    DEFAULT_HEAD_CONFIG = "confidence/distillation_teddymer_multihead"

    def __init__(
        self,
        ckpt_path: str,
        proteina: torch.nn.Module,
        trunk_eval_t: float = 0.99,
        head_config_name: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        head = self._build_head(head_config_name or self.DEFAULT_HEAD_CONFIG)
        module = ConfidenceDistillationModule.load_from_checkpoint(
            ckpt_path,
            head=head,
            proteina=proteina,
            trunk_eval_t=trunk_eval_t,
        )
        self._init_from_module(module, trunk_eval_t=trunk_eval_t, device=device)

    @staticmethod
    def _build_head(head_config_name: str):
        """Instantiate the multi-head architecture from its training Hydra config.

        The checkpoint stores head weights but not the head architecture, so we
        compose the training config and instantiate its ``confidence.head`` block.
        """
        import hydra
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        configs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "configs",
        )
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=configs_dir):
            cfg = compose(config_name=head_config_name)
        return hydra.utils.instantiate(cfg.confidence.head)

    @classmethod
    def from_module(
        cls,
        module: ConfidenceDistillationModule,
        trunk_eval_t: float = 0.99,
        device: torch.device | str | None = None,
    ) -> "ConfidenceHeadScorer":
        """Build a scorer around an already-constructed module (used by tests)."""
        self = cls.__new__(cls)
        self._init_from_module(module, trunk_eval_t=trunk_eval_t, device=device)
        return self

    def _init_from_module(
        self,
        module: ConfidenceDistillationModule,
        trunk_eval_t: float,
        device: torch.device | str | None,
    ) -> None:
        if not isinstance(module.head, MultiHeadConfidence):
            raise TypeError(
                "ConfidenceHeadScorer requires a MultiHeadConfidence head with "
                "'plddt' and 'pae' children; got "
                f"{type(module.head).__name__}."
            )
        for name in ("plddt", "pae"):
            if name not in module.head.children_heads:
                raise ValueError(
                    f"ConfidenceHeadScorer requires a '{name}' child head; "
                    f"found {list(module.head.children_heads)}."
                )
        if device is not None:
            module = module.to(device)
        module.eval()
        self.module = module
        self.trunk_eval_t = float(trunk_eval_t)

    # ------------------------------------------------------------------
    # Pure reduction (unit-tested in isolation)
    # ------------------------------------------------------------------
    @staticmethod
    def _reduce(
        pae_ev: torch.Tensor,
        plddt_ev: torch.Tensor,
        chain_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Reduce head expected values to per-sample scores.

        Args:
            pae_ev: ``[B, L, L]`` PAE expected value in Angstroms.
            plddt_ev: ``[B, L]`` pLDDT expected value on the 0-100 scale.
            chain_idx: ``[B, L]`` integer per-residue chain ids (>= 2 distinct
                ids define an interface).
            mask: ``[B, L]`` boolean residue validity.

        Returns:
            ``{"ipae": [B] float32 (A, NaN if no interface), "complex_plddt":
            [B] float32 (0-1), "provisional_success": [B] bool}``.

        ``ipae`` is the MEAN over the cross-chain interface (``_metrics.i_pae``,
        the verbatim colabdesign ``i_pAE`` port), matching the definition the
        analyze stage gates on -- not the per-row ``min_ipae``.
        """
        mask_bool = mask.bool()
        pair_valid = mask_bool[:, :, None] & mask_bool[:, None, :]
        inter_mask = interface_pair_mask(chain_idx, pair_valid)
        ipae = i_pae(pae_ev.float(), inter_mask, reduce="per_sample").float()

        mask_f = mask_bool.to(plddt_ev.dtype)
        denom = mask_f.sum(dim=-1).clamp_min(1.0)
        complex_plddt = ((plddt_ev.float() * mask_f).sum(dim=-1) / denom) / PLDDT_EV_SCALE
        complex_plddt = complex_plddt.float()

        ipae_ok = torch.nan_to_num(ipae, nan=float("inf")) < SUCCESS_IPAE_ANGSTROM
        plddt_ok = complex_plddt > SUCCESS_PLDDT_01
        provisional_success = ipae_ok & plddt_ok

        return {
            "ipae": ipae,
            "complex_plddt": complex_plddt,
            "provisional_success": provisional_success,
        }

    # ------------------------------------------------------------------
    # Native-frame batch construction
    # ------------------------------------------------------------------
    def _build_native_batch(self, complex_prots: dict) -> dict:
        """Build a no-concat confidence batch from a joint binder+target complex.

        ``complex_prots`` is the joint complex produced by
        ``prepend_target_to_samples``: ``coors`` ``[B, L, 37, 3]`` in Angstroms,
        ``residue_type`` ``[B, L]``, ``chain_index`` ``[B, L]`` (2-valued),
        ``mask`` ``[B, L]`` bool. Both chains are regular residues, so this is
        the native single frame the head trained on. We deliberately set no
        ``x_target`` / ``seq_target`` / ``target_mask`` keys, so the trunk's
        concat path stays inactive (``n_concat == 0``).
        """
        coors = complex_prots["coors"]
        mask = complex_prots["mask"].bool()
        device = coors.device

        coords_nm = ang_to_nm(coors.float())
        residue_type = complex_prots["residue_type"].to(device)
        chain_idx = complex_prots["chain_index"].to(device).long()

        # atom37 validity: any atom with non-zero coordinate is present. Padded
        # residues carry all-zero coordinates, so this also encodes the residue
        # mask on axis 0/1 of coords.
        atom_mask = coors.abs().sum(dim=-1) > 1e-7  # [B, L, 37]
        atom_mask = atom_mask & mask[..., None]

        # The frozen VAE encoder's feature factories read a top-level
        # ``coord_mask`` [B, L, 37] (float) plus ``coords_nm``/``residue_type``;
        # the trunk pair/seq factories read ``mask_dict``. Provide both, matching
        # the shape the normal dataset transforms emit (`graph.coord_mask`).
        coord_mask = atom_mask.to(coords_nm.dtype)

        # ``process_batch`` derives the residue mask as
        # ``mask_dict["coords"][..., 0, 0]``, so this entry must be [B, L, 37, 3]
        # (coord-level) whose atom-0/coord-0 slice is the per-residue mask.
        coords_mask_4d = mask[:, :, None, None].expand(-1, -1, 37, 3)

        # The encoder/trunk feature factories read atom37 coords in Angstroms
        # under ``coords`` (torsion/bond-angle features), nm coords under
        # ``coords_nm`` (distance features), and a per-residue ``chains`` vector.
        batch: dict = {
            "coords": coors.float(),
            "coords_nm": coords_nm,
            "coors": coors,
            "residue_type": residue_type,
            "mask": mask,
            "chain_idx": chain_idx,
            "chains": chain_idx,
            "coord_mask": coord_mask,
            "mask_dict": {"coords": coords_mask_4d, "residue_type": mask},
        }
        return batch

    # ------------------------------------------------------------------
    # Frozen forward at t = trunk_eval_t (mirrors lightning_module._forward,
    # minus the label-mask / trim machinery which is training-only)
    # ------------------------------------------------------------------
    def _run_head(self, batch: dict) -> dict[str, torch.Tensor]:
        module = self.module
        proteina = module.proteina
        with torch.no_grad():
            batch = add_clean_samples(
                batch,
                proteina.cfg_exp.product_flowmatcher,
                getattr(proteina, "autoencoder", None),
            )
            batch = proteina.fm.corrupt_batch(batch)
            b = batch["mask"].shape[0]
            device = batch["mask"].device
            t_pinned = {
                m: torch.full((b,), self.trunk_eval_t, device=device, dtype=batch["t"][m].dtype)
                for m in batch["t"]
            }
            batch["t"] = t_pinned
            batch["x_t"] = proteina.fm.interpolate(
                x_0=batch["x_0"],
                x_1=batch["x_1"],
                t=t_pinned,
                mask=batch["mask"],
            )
            proteina.nn.expose_intermediates = True
            nn_out = proteina.nn(batch)

            inter = nn_out["trunk_intermediates"]
            s = inter["s"]
            z = inter["z"]
            mask_ext = inter["mask"]
            local_latents = inter["local_latents"]
            ca_coords = inter.get("ca_coords")

            cond = module._compute_cond(batch)
            cond = module._pad_cond_to_n_ext(cond, mask_ext)

            head_out = module.head(s, z, mask_ext, cond, local_latents, ca_coords=ca_coords)
        return head_out

    def score_native(self, complex_prots: dict) -> dict[str, torch.Tensor]:
        """Primary in-distribution score on the native co-diffused frame.

        Returns ``{"ipae": [B] (A), "complex_plddt": [B] (0-1),
        "provisional_success": [B] bool}``.
        """
        batch = self._build_native_batch(complex_prots)
        chain_idx = batch["chain_idx"]
        mask = batch["mask"]

        head_out = self._run_head(batch)
        pae_logits = head_out["pae"]["pae_logits"]
        plddt_logits = head_out["plddt"]["plddt_logits"]

        pae_ev = self.module.head.children_heads["pae"].pae_ev_from_logits(pae_logits)
        plddt_ev = self.module.head.children_heads["plddt"].logits_to_expected_value(plddt_logits)

        return self._reduce(pae_ev, plddt_ev, chain_idx, mask)

    # ------------------------------------------------------------------
    # Ablation: reuse the concat-frame generation latents (uncalibrated)
    # ------------------------------------------------------------------
    def score_concat_reuse(
        self,
        trunk_intermediates: dict,
        target_chains: torch.Tensor,
        n_orig: int,
    ) -> dict[str, torch.Tensor]:
        """Uncalibrated interface ipAE from the generation concat frame.

        For the parity ablation only -- the head never trained on this frame,
        so the number is OOD. ``trunk_intermediates`` is the extended
        ``[binder | target]`` intermediates captured during generation;
        ``n_orig`` is the binder length; ``target_chains`` ``[B, n_target]`` are
        the target's per-residue chain ids (valid positions only).
        """
        with torch.no_grad():
            s = trunk_intermediates["s"]
            z = trunk_intermediates["z"]
            mask_ext = trunk_intermediates["mask"].bool()
            local_latents = trunk_intermediates["local_latents"]
            ca_coords = trunk_intermediates.get("ca_coords")
            cond = trunk_intermediates["cond"]

            head_out = self.module.head(s, z, mask_ext, cond, local_latents, ca_coords=ca_coords)
            pae_logits = head_out["pae"]["pae_logits"]
            pae_ev = self.module.head.children_heads["pae"].pae_ev_from_logits(pae_logits)

            b, n_ext = mask_ext.shape
            device = mask_ext.device
            chain_idx_ext = torch.zeros(b, n_ext, dtype=torch.long, device=device)
            n_target = n_ext - n_orig
            if n_target > 0:
                chain_idx_ext[:, n_orig:] = target_chains[:, :n_target].to(device).long() + 1

            pair_valid = mask_ext[:, :, None] & mask_ext[:, None, :]
            inter_mask = interface_pair_mask(chain_idx_ext, pair_valid)
            ipae_reuse = i_pae(pae_ev.float(), inter_mask, reduce="per_sample").float()
        return {"ipae_reuse": ipae_reuse}
