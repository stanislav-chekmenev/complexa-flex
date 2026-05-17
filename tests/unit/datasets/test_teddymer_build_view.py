"""End-to-end build pipeline test against synthetic Teddymer + inventory inputs.

Exercises ``build_view.run`` (the importable entry point of the CLI) on a tiny
synthetic dataset to confirm that the three artefacts produced by the SLURM
job (``dimers.parquet``, ``locator_rows.parquet``, and a coverage summary)
are written with the expected schemas and contents.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _write_inventory_batch(inv_dir: Path, afdb_ids: list[str]) -> None:
    fields = [
        ("sample_id", pa.string()), ("afdb_id", pa.string()),
        ("uniprot_id", pa.string()), ("taxonomy_id", pa.string()),
        ("source_tar_relpath", pa.string()), ("source_tar_basename", pa.string()),
        ("source_version", pa.int32()), ("bucket", pa.int32()),
        ("has_cif", pa.bool_()), ("has_pae", pa.bool_()), ("has_conf", pa.bool_()),
        ("cif_member_name", pa.string()), ("cif_member_offset", pa.int64()),
        ("cif_member_size", pa.int64()), ("cif_is_gz", pa.bool_()),
        ("pae_member_name", pa.string()), ("pae_member_offset", pa.int64()),
        ("pae_member_size", pa.int64()), ("pae_is_gz", pa.bool_()),
        ("conf_member_name", pa.string()), ("conf_member_offset", pa.int64()),
        ("conf_member_size", pa.int64()), ("conf_is_gz", pa.bool_()),
    ]
    schema = pa.schema(fields)
    rows = []
    for i, aid in enumerate(afdb_ids):
        rows.append({
            "sample_id": aid, "afdb_id": aid,
            "uniprot_id": aid.removeprefix("AF-").removesuffix("-F1"),
            "taxonomy_id": "9606",
            "source_tar_relpath": f"proteome-tax_id-9606-{i}_v4.tar",
            "source_tar_basename": f"proteome-tax_id-9606-{i}_v4.tar",
            "source_version": 4, "bucket": i,
            "has_cif": True, "has_pae": True, "has_conf": True,
            "cif_member_name": f"{aid}-model_v4.cif.gz",
            "cif_member_offset": 1000 * (i + 1), "cif_member_size": 500, "cif_is_gz": True,
            "pae_member_name": f"{aid}-predicted_aligned_error_v4.json.gz",
            "pae_member_offset": 2000 * (i + 1), "pae_member_size": 800, "pae_is_gz": True,
            "conf_member_name": f"{aid}-confidence_v4.json.gz",
            "conf_member_offset": 3000 * (i + 1), "conf_member_size": 300, "conf_is_gz": True,
        })
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, inv_dir / "w0_b000000.parquet")


def _write_teddymer_staging(staging: Path) -> None:
    repdb = staging / "teddymer_repdb"
    repdb.mkdir(parents=True, exist_ok=True)
    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-15\n",
        "7DI_AF-A0A005-F1-model_v4_TED02\tCATH3.40.50.2000_RES25-35\n",
        "34DI_AF-A0A009-F1-model_v4_TED01\tCATH3.40.50.2000_RES1-12\n",
        "34DI_AF-A0A009-F1-model_v4_TED02\tCATH3.40.50.2000_RES15-22\n",
    ]
    h_bytes = b"".join(h.encode() for h in headers)
    (repdb / "teddymer_repdb_h").write_bytes(h_bytes)
    cur, idx_lines = 0, []
    for i, h in enumerate(headers, start=14):
        sz = len(h.encode())
        idx_lines.append(f"{i}\t{cur}\t{sz}\n")
        cur += sz
    (repdb / "teddymer_repdb_h.index").write_text("".join(idx_lines))

    meta = (
        "DimerIndex\tUniProtID\tDomainPair\tMemberCount\tInterfaceLength"
        "\tAvgIntPAE\tAvgIntPlddt\tIntPlddt\n"
        # Dimer 7: clearly above Complexa-filter thresholds.
        "7\tAF-A0A005-F1-model\tTED01:TED02\t10\t20\t3.0\t88.0\t999999988:88888889999\n"
        # Dimer 34: avg_int_pae too high (12 >= 10) so complexa_filter=False.
        "34\tAF-A0A009-F1-model\tTED01:TED02\t5\t15\t12.0\t72.0\t9988998:88997655\n"
    )
    (staging / "nonsingletonrep_metadata.tsv").write_text(meta)


def test_run_end_to_end_writes_both_parquets_and_view_config(tmp_path):
    import yaml

    from proteinfoundation.datasets.teddymer.build_view import run

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_teddymer_staging(staging)

    inv = tmp_path / "inv"
    inv.mkdir()
    _write_inventory_batch(inv, ["AF-A0A005-F1", "AF-A0A009-F1"])

    out = tmp_path / "out"
    out.mkdir()

    result = run(
        staging=staging,
        inventory=inv,
        out=out,
        afdb_proteomes=None,
        sanity_n=0,
        on_missing="raise",
    )
    assert (out / "dimers.parquet").exists()
    assert (out / "locator_rows.parquet").exists()
    assert (out / "view_config.yaml").exists()

    dimers = pd.read_parquet(out / "dimers.parquet")
    assert len(dimers) == 2
    assert dimers["complexa_filter"].tolist() == [True, False]

    locator = pd.read_parquet(out / "locator_rows.parquet")
    assert len(locator) == 4  # 2 dimers x 2 chains
    assert set(locator["chain_id"].unique()) == {"A", "B"}

    assert result["n_dimers"] == 2
    assert result["n_complexa_filter"] == 1
    assert result["n_dimers_with_inventory"] == 2
    assert result["n_locator_rows"] == 4

    config = yaml.safe_load((out / "view_config.yaml").read_text())
    assert config["name"] == "teddymer_v1"
    assert config["row_count"] == 2
    assert config["n_complexa_filter"] == 1
    assert config["n_dimers_with_inventory"] == 2
    assert config["n_locator_rows"] == 4
    assert config["on_missing"] == "raise"
    assert config["inventory_source"] == str(inv)
    assert config["raw_input_dir"] == str(staging)
    assert config["dimers_path"].endswith("dimers.parquet")
    assert config["locator_rows_path"].endswith("locator_rows.parquet")


def test_run_drops_dimers_outside_inventory_when_requested(tmp_path):
    from proteinfoundation.datasets.teddymer.build_view import run

    staging = tmp_path / "staging"
    staging.mkdir()
    _write_teddymer_staging(staging)
    inv = tmp_path / "inv"
    inv.mkdir()
    # Only the first AFDB id is in the inventory.
    _write_inventory_batch(inv, ["AF-A0A005-F1"])

    out = tmp_path / "out"
    out.mkdir()
    result = run(
        staging=staging,
        inventory=inv,
        out=out,
        afdb_proteomes=None,
        sanity_n=0,
        on_missing="drop",
    )
    locator = pd.read_parquet(out / "locator_rows.parquet")
    assert len(locator) == 2
    assert set(locator["dimer_index"].unique()) == {7}
    assert result["n_dimers"] == 2
    assert result["n_dimers_with_inventory"] == 1
    assert result["n_locator_rows"] == 2
