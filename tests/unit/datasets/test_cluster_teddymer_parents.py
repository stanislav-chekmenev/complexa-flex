"""Unit tests for the pure helpers in `scripts/cluster_teddymer_parents.py`.

The script extracts one parent-monomer sequence per unique `uniprot_id`,
clusters them with MMseqs2 at 30% identity, and maps the cluster
representative back onto every dimer row as a `seqclust30` column. These
tests cover the three pure helpers that do not touch MMseqs2 or the tar
blob: TSV parsing, dimer-to-cluster assignment, and one-letter sequence
extraction from CIF bytes.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "cluster_teddymer_parents.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("cluster_teddymer_parents", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_script_module()


def _cif_bytes(res_names: list[str], chain_id: str = "A") -> bytes:
    import biotite.structure as bs
    from biotite.structure import Atom, array
    from biotite.structure.io.pdbx import CIFFile, set_structure

    atoms = [
        Atom(
            [float(i), 0.0, 0.0],
            chain_id=chain_id,
            res_id=i,
            res_name=rn,
            atom_name="CA",
            element="C",
        )
        for i, rn in enumerate(res_names, start=1)
    ]
    cif = CIFFile()
    set_structure(cif, array(atoms))
    buf = io.StringIO()
    cif.write(buf)
    return buf.getvalue().encode()


def test_tsv_to_member_map_parses_rep_member_pairs(mod, tmp_path: Path) -> None:
    tsv = tmp_path / "cluster.tsv"
    tsv.write_text(
        "rep1\trep1\n"
        "rep1\tmemberA\n"
        "rep1\tmemberB\n"
        "rep2\trep2\n"
        "rep2\tmemberC\n"
    )
    member_map = mod._tsv_to_member_map(tsv)
    assert member_map == {
        "rep1": "rep1",
        "memberA": "rep1",
        "memberB": "rep1",
        "rep2": "rep2",
        "memberC": "rep2",
    }


def test_assign_clusters_to_dimers_covers_all_rows(mod) -> None:
    dimers = pd.DataFrame(
        {
            "dimer_id": ["d0", "d1", "d2", "d3"],
            "uniprot_id": ["P1", "P2", "P1", "P3"],
        }
    )
    uniprot_to_cluster = {"P1": "rep1", "P2": "rep1", "P3": "rep2"}
    out = mod._assign_clusters_to_dimers(
        dimers, uniprot_to_cluster, uniprot_column="uniprot_id", out_column="seqclust30"
    )
    assert len(out) == len(dimers)
    assert out["seqclust30"].tolist() == ["rep1", "rep1", "rep1", "rep2"]
    # original column count + 1
    assert "seqclust30" in out.columns and "uniprot_id" in out.columns


def test_assign_clusters_raises_on_unmapped_uniprot(mod) -> None:
    dimers = pd.DataFrame({"dimer_id": ["d0"], "uniprot_id": ["P_missing"]})
    with pytest.raises(KeyError):
        mod._assign_clusters_to_dimers(
            dimers, {"P1": "rep1"}, uniprot_column="uniprot_id", out_column="seqclust30"
        )


def test_seq_from_cif_bytes_chain_a_one_letter(mod) -> None:
    cif = _cif_bytes(["ALA", "GLY", "SER", "PRO", "LYS"])
    seq = mod._seq_from_cif_bytes(cif, chain_id="A")
    assert seq == "AGSPK"


def test_seq_from_cif_bytes_selects_named_chain(mod) -> None:
    from biotite.structure import Atom, array
    from biotite.structure.io.pdbx import CIFFile, set_structure

    atoms = []
    for i, rn in enumerate(["ALA", "GLY"], start=1):
        atoms.append(Atom([float(i), 0, 0], chain_id="A", res_id=i, res_name=rn, atom_name="CA", element="C"))
    for i, rn in enumerate(["TRP", "TYR", "PHE"], start=1):
        atoms.append(Atom([float(i), 1, 0], chain_id="B", res_id=i, res_name=rn, atom_name="CA", element="C"))
    cif = CIFFile()
    set_structure(cif, array(atoms))
    buf = io.StringIO()
    cif.write(buf)
    data = buf.getvalue().encode()
    assert mod._seq_from_cif_bytes(data, chain_id="A") == "AG"
    assert mod._seq_from_cif_bytes(data, chain_id="B") == "WYF"
