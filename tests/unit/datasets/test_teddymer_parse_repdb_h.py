from __future__ import annotations

import pandas as pd
import pytest


def test_parse_header_line_continuous_domain():
    from proteinfoundation.datasets.teddymer.parse_repdb_h import HeaderRecord, parse_header_line

    line = "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190"
    rec = parse_header_line(line)
    assert rec == HeaderRecord(
        dimer_index=7,
        uniprot_id="A0A005",
        parent_afdb_id="AF-A0A005-F1",
        ted_index=1,
        cath_id="3.40.50",
        residue_intervals=[(7, 190)],
    )


def test_parse_header_line_discontinuous_domain():
    from proteinfoundation.datasets.teddymer.parse_repdb_h import parse_header_line

    line = "34DI_AF-A0A009E3M2-F1-model_v4_TED02\tCATH3.40.50.2000_RES15-30_37-122_289-300"
    rec = parse_header_line(line)
    assert rec.dimer_index == 34
    assert rec.ted_index == 2
    assert rec.residue_intervals == [(15, 30), (37, 122), (289, 300)]
    assert rec.cath_id == "3.40.50.2000"


def test_parse_header_line_rejects_malformed():
    from proteinfoundation.datasets.teddymer.parse_repdb_h import parse_header_line

    with pytest.raises(ValueError, match="header"):
        parse_header_line("not-a-valid-line")


def test_parse_int_plddt_two_chains():
    from proteinfoundation.datasets.teddymer.parse_repdb_h import parse_int_plddt

    s = "5556776777777777756677889:99998998888888887888887765663343"
    a, b = parse_int_plddt(s)
    assert len(a) == 25
    assert len(b) == 32
    assert all(0 <= x <= 9 for x in a + b)
    assert a[0] == 5
    assert b[0] == 9


def test_parse_int_plddt_rejects_more_than_two_chains():
    from proteinfoundation.datasets.teddymer.parse_repdb_h import parse_int_plddt

    with pytest.raises(ValueError, match="exactly two"):
        parse_int_plddt("1:2:3")


def _write_synthetic_repdb(
    tmp_path,
    headers: list[str],
    metadata_rows: list[str],
):
    h_path = tmp_path / "teddymer_repdb_h"
    idx_path = tmp_path / "teddymer_repdb_h.index"
    meta_path = tmp_path / "nonsingletonrep_metadata.tsv"

    h_bytes = "".join(headers).encode()
    h_path.write_bytes(h_bytes)

    idx_lines: list[str] = []
    cur = 0
    for i, h in enumerate(headers, start=14):
        sz = len(h.encode())
        idx_lines.append(f"{i}\t{cur}\t{sz}\n")
        cur += sz
    idx_path.write_text("".join(idx_lines))

    header_row = (
        "DimerIndex\tUniProtID\tDomainPair\tMemberCount\tInterfaceLength"
        "\tAvgIntPAE\tAvgIntPlddt\tIntPlddt\n"
    )
    meta_path.write_text(header_row + "".join(metadata_rows))
    return h_path, idx_path, meta_path


def _row_by_dimer_index(df: pd.DataFrame, dimer_index: int) -> dict:
    rows = df[df["dimer_index"] == dimer_index]
    assert len(rows) == 1, f"expected 1 row for dimer {dimer_index}, got {len(rows)}"
    return rows.iloc[0].to_dict()


def test_build_dimers_table_joins_h_and_metadata(tmp_path):
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table

    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A005-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
        "34DI_AF-A0A009E3M2-F1-model_v4_TED01\tCATH3.40.50.2000_RES15-30_37-122_289-300\n",
        "34DI_AF-A0A009E3M2-F1-model_v4_TED02\tCATH3.40.50.2000_RES125-284\n",
    ]
    metadata_rows = [
        "7\tAF-A0A005-F1-model\tTED01:TED02\t3\t10\t6.5\t72.5\t12345:67890\n",
        "34\tAF-A0A009E3M2-F1-model\tTED01:TED02\t2\t6\t4.0\t85.0\t999:887\n",
    ]
    h_path, idx_path, meta_path = _write_synthetic_repdb(tmp_path, headers, metadata_rows)

    df = build_dimers_table(h_path, idx_path, meta_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    expected_cols = {
        "dimer_id", "dimer_index", "uniprot_id", "parent_afdb_id",
        "ted_index_A", "ted_index_B", "cath_id_A", "cath_id_B",
        "residue_intervals_A", "residue_intervals_B",
        "member_count", "interface_length", "avg_int_pae", "avg_int_plddt",
        "int_plddt_chain_A", "int_plddt_chain_B",
    }
    assert expected_cols.issubset(set(df.columns))

    row7 = _row_by_dimer_index(df, 7)
    assert row7["parent_afdb_id"] == "AF-A0A005-F1"
    assert row7["uniprot_id"] == "A0A005"
    assert row7["dimer_id"] == "7DI_AF-A0A005-F1-model_v4"
    assert row7["ted_index_A"] == 1 and row7["ted_index_B"] == 2
    assert row7["cath_id_A"] == "3.40.50"
    assert row7["cath_id_B"] == "3.40.50.2000"
    assert list(row7["residue_intervals_A"]) == [{"lo": 7, "hi": 190}]
    assert list(row7["residue_intervals_B"]) == [{"lo": 225, "hi": 381}]
    assert list(row7["int_plddt_chain_A"]) == [1, 2, 3, 4, 5]
    assert list(row7["int_plddt_chain_B"]) == [6, 7, 8, 9, 0]
    assert row7["interface_length"] == 10
    assert row7["member_count"] == 3
    assert row7["avg_int_pae"] == pytest.approx(6.5)
    assert row7["avg_int_plddt"] == pytest.approx(72.5)

    row34 = _row_by_dimer_index(df, 34)
    assert list(row34["residue_intervals_A"]) == [
        {"lo": 15, "hi": 30}, {"lo": 37, "hi": 122}, {"lo": 289, "hi": 300}
    ]


def test_build_dimers_table_skips_dimers_absent_from_metadata(tmp_path):
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table

    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A005-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
        "8DI_AF-A0A006-F1-model_v4_TED01\tCATH3.40.50_RES1-100\n",
        "8DI_AF-A0A006-F1-model_v4_TED02\tCATH3.40.50_RES101-200\n",
    ]
    metadata_rows = [
        "7\tAF-A0A005-F1-model\tTED01:TED02\t3\t10\t6.5\t72.5\t12345:67890\n",
    ]
    h_path, idx_path, meta_path = _write_synthetic_repdb(tmp_path, headers, metadata_rows)

    df = build_dimers_table(h_path, idx_path, meta_path)
    assert len(df) == 1
    assert df["dimer_index"].tolist() == [7]


def test_build_dimers_table_asserts_domain_pair_consistent(tmp_path):
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table

    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A005-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
    ]
    metadata_rows = [
        "7\tAF-A0A005-F1-model\tTED01:TED03\t3\t10\t6.5\t72.5\t12345:67890\n",
    ]
    h_path, idx_path, meta_path = _write_synthetic_repdb(tmp_path, headers, metadata_rows)

    with pytest.raises(ValueError, match="DomainPair"):
        build_dimers_table(h_path, idx_path, meta_path)


def test_build_dimers_table_asserts_intra_monomer(tmp_path):
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table

    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A006-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
    ]
    metadata_rows = [
        "7\tAF-A0A005-F1-model\tTED01:TED02\t3\t10\t6.5\t72.5\t12345:67890\n",
    ]
    h_path, idx_path, meta_path = _write_synthetic_repdb(tmp_path, headers, metadata_rows)

    with pytest.raises(ValueError, match="intra-monomer"):
        build_dimers_table(h_path, idx_path, meta_path)


def test_build_dimers_table_asserts_interface_length_consistent(tmp_path):
    from proteinfoundation.datasets.teddymer.parse_repdb_h import build_dimers_table

    headers = [
        "7DI_AF-A0A005-F1-model_v4_TED01\tCATH3.40.50_RES7-190\n",
        "7DI_AF-A0A005-F1-model_v4_TED02\tCATH3.40.50.2000_RES225-381\n",
    ]
    metadata_rows = [
        "7\tAF-A0A005-F1-model\tTED01:TED02\t3\t99\t6.5\t72.5\t12345:67890\n",
    ]
    h_path, idx_path, meta_path = _write_synthetic_repdb(tmp_path, headers, metadata_rows)

    with pytest.raises(ValueError, match="InterfaceLength"):
        build_dimers_table(h_path, idx_path, meta_path)
