"""Builds the synthetic atom37 fixture used by `tests/integration/confidence/`.

Re-run only when changing the fixture contents. Output is committed under
`tests/fixtures/afdb_one_protein/synthetic_atom37.pt`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from proteinfoundation.datasets.transforms import Data


def build() -> Data:
    n = 37
    coords = torch.zeros(n, 37, 3, dtype=torch.float32)
    coords[:, 1, 0] = torch.arange(n, dtype=torch.float32) * 3.8

    coord_mask = torch.zeros(n, 37, dtype=torch.bool)
    coord_mask[:, :5] = True

    residue_type = torch.arange(n, dtype=torch.long) % 20

    base = torch.linspace(40.0, 95.0, steps=n, dtype=torch.float32)
    atom_b_factor = base.unsqueeze(-1).repeat(1, 37)

    return Data(
        coords=coords,
        coord_mask=coord_mask,
        residue_type=residue_type,
        atom_b_factor=atom_b_factor,
        id="synthetic_afdb_001",
        database="afdb",
        num_nodes=n,
    )


if __name__ == "__main__":
    out_path = Path(__file__).with_name("synthetic_atom37.pt")
    data = build()
    assert data.atom_b_factor.max().item() > 1.5
    torch.save(data, out_path)
    print(f"Saved fixture to {out_path} (n_residues={data.coords.shape[0]})")
