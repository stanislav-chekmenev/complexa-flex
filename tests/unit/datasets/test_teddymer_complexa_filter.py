from __future__ import annotations

import pandas as pd


def test_complexa_filter_three_thresholds():
    """The Complexa training-set filter is the conjunction of three conditions:
    interface_length > 10, avg_int_plddt > 70, avg_int_pae < 10. Each row below
    pokes exactly one of those conditions across the boundary.
    """
    from proteinfoundation.data.teddymer.parse_repdb_h import add_complexa_filter

    df = pd.DataFrame(
        {
            "dimer_index":      [1,    2,    3,    4,    5],
            "interface_length": [11,   10,   11,   11,   11],
            "avg_int_plddt":    [71.0, 71.0, 70.0, 71.0, 71.0],
            "avg_int_pae":      [9.0,  9.0,  9.0,  10.0, 9.0],
        }
    )
    out = add_complexa_filter(df)
    assert out["complexa_filter"].tolist() == [True, False, False, False, True]


def test_complexa_filter_preserves_other_columns():
    """add_complexa_filter must not drop or modify existing columns."""
    from proteinfoundation.data.teddymer.parse_repdb_h import add_complexa_filter

    df = pd.DataFrame(
        {
            "dimer_index":      [1, 2],
            "interface_length": [20, 5],
            "avg_int_plddt":    [80.0, 80.0],
            "avg_int_pae":      [5.0, 5.0],
            "extra_label":      ["a", "b"],
        }
    )
    out = add_complexa_filter(df)
    assert list(out.columns) == ["dimer_index", "interface_length", "avg_int_plddt",
                                 "avg_int_pae", "extra_label", "complexa_filter"]
    assert out["extra_label"].tolist() == ["a", "b"]
