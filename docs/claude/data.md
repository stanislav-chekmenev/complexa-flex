# Dataset / supervised-pool invariants

Read this file before changing any `configs/dataset/unified/*.yaml` filter or the Teddymer
supervised pool. Confidence-head architecture lives in [confidence.md](confidence.md).

- **Teddymer filter is geometry-only.** Supervised pool is `interface_length > 10` at [configs/dataset/unified/teddymer_with_plddt_and_pae.yaml](../../configs/dataset/unified/teddymer_with_plddt_and_pae.yaml). **Confidence-based pre-filters are a selection-bias antipattern**: the head's targets are per-residue pLDDT and per-pair PAE, so filtering on aggregates of those same quantities (`avg_int_plddt`, `avg_int_pae`) truncates the label distribution by the label itself. AF2-multimer, Boltz-1/-2, and Chai-1 filter on data-source quality (resolution, identity clustering, homology) but never on the model's own confidence; Boltz-2 names the antipattern. Only filter by geometric properties independent of the AF2 confidence map. `complexa_filter` is still on `dimers.parquet` but unused. Contract pinned by [tests/unit/datasets/test_teddymer_dataset_config.py::test_yaml_filter_pins_geometry_only_threshold](../../tests/unit/datasets/test_teddymer_dataset_config.py).
