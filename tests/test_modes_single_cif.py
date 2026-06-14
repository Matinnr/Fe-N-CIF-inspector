"""
Tests for src.modes.single_cif helpers (the pieces that don't need a
running Streamlit session).

The headline test is the duplicate-column regression: when the bond
DataFrame has user_* annotation columns appended *after* the raw
distance columns, the display-table builder must still produce a
unique-named column list. The earlier implementation didn't, and
st.dataframe blew up at render time.
"""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit  # noqa: F401  (macOS import-order)

from src.data_schema import (
    BOND_TABLE_COLUMNS,
    COL_CIF_FILE, COL_DETECTION_METHOD, COL_DISTANCE, COL_DISTANCE_ESD,
    COL_FE_LABEL, COL_N_LABEL, COL_STRUCTURE_ID, COL_SYMMETRY_RELATED,
    COL_TEMPERATURE, COL_WARNING, DetectionMethod,
)
from src.modes.single_cif import _build_display_table


def _minimal_row(d, esd=None):
    return {
        COL_CIF_FILE:         "demo.cif",
        COL_STRUCTURE_ID:     "demo",
        COL_TEMPERATURE:      100.0,
        COL_FE_LABEL:         "Fe1",
        COL_N_LABEL:          "N1",
        COL_DISTANCE:         d,
        COL_DISTANCE_ESD:     esd,
        COL_DETECTION_METHOD: DetectionMethod.FORMAL_BOND.value,
        COL_SYMMETRY_RELATED: False,
        COL_WARNING:          "",
    }


# ---------------------------------------------------------------------
# 1) duplicate-column regression
# ---------------------------------------------------------------------

def test_no_duplicate_columns_when_user_annotations_appended():
    """The bug we just fixed: user_oxidation_state / user_spin_state
    added to the right of the raw distances should not cause the
    display builder to emit two 'distance (Å)' columns."""
    df = pd.DataFrame([
        _minimal_row(1.984, 0.007),
        _minimal_row(1.992, 0.008),
    ], columns=BOND_TABLE_COLUMNS)
    df["user_oxidation_state"] = "Fe(II)"
    df["user_spin_state"]      = "LS"

    out = _build_display_table(df)
    assert out.columns.is_unique, (
        f"display table has duplicate columns: {list(out.columns)}"
    )


# ---------------------------------------------------------------------
# 2) numeric distance is replaced by formatted column
# ---------------------------------------------------------------------

def test_display_distance_column_uses_esd_notation():
    df = pd.DataFrame(
        [_minimal_row(1.984, 0.007)], columns=BOND_TABLE_COLUMNS,
    )
    out = _build_display_table(df)
    assert "distance (Å)" in out.columns
    assert COL_DISTANCE not in out.columns
    assert COL_DISTANCE_ESD not in out.columns
    assert out["distance (Å)"].iloc[0] == "1.984(7)"


def test_display_distance_column_falls_back_to_bare_value_when_no_esd():
    df = pd.DataFrame(
        [_minimal_row(1.984, None)], columns=BOND_TABLE_COLUMNS,
    )
    out = _build_display_table(df)
    assert out["distance (Å)"].iloc[0] == "1.984"


# ---------------------------------------------------------------------
# 3) column order
# ---------------------------------------------------------------------

def test_display_keeps_column_order_with_formatted_distance_in_place():
    df = pd.DataFrame(
        [_minimal_row(1.984, 0.007)], columns=BOND_TABLE_COLUMNS,
    )
    df["user_oxidation_state"] = "Fe(II)"
    df["user_spin_state"]      = "LS"

    out = list(_build_display_table(df).columns)
    # The formatted column sits exactly where COL_DISTANCE used to be.
    distance_idx = list(BOND_TABLE_COLUMNS).index(COL_DISTANCE)
    assert out[distance_idx] == "distance (Å)"
    # The two user annotation columns are at the end.
    assert out[-2:] == ["user_oxidation_state", "user_spin_state"]
