"""
Tests for src/cohort.py — the pure cohort-table helpers.

These don't go through Streamlit. They construct AnalysisResult and
CifBundle stand-ins, run `summaries_for_cif()` /
`build_cohort_dataframe()` / `auto_assign_series()`, and inspect the
output DataFrames.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src.cif_reader import read_cif
from src.fe_n_analysis import analyse
from src.cohort import (
    COHORT_COLUMNS,
    COL_BVS, COL_FE, COL_FILE, COL_FORMULA, COL_MEAN_D,
    COL_OX, COL_REFCODE, COL_SERIES, COL_SIGMA, COL_SPIN, COL_TEMP,
    FeSummary,
    auto_assign_series,
    build_cohort_dataframe,
    make_error_summary,
    summaries_for_cif,
)


# ---------------------------------------------------------------------
# 1) summaries_for_cif on the canonical FeN6 fixture
# ---------------------------------------------------------------------

def test_summaries_one_per_fe_centre(fe_octahedral_cif: Path):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename="fe_octahedral.cif")
    out = summaries_for_cif("fe_octahedral.cif", bundle, result)
    assert len(out) == 1
    s = out[0]
    assert isinstance(s, FeSummary)
    assert s.filename == "fe_octahedral.cif"
    assert s.fe_label == "Fe1"
    assert s.n_FeN == 6
    assert len(s.distances_A) == 6
    # All six bond lengths in the fixture are 2.0 Å exactly.
    assert s.mean_d_A == pytest.approx(2.0, abs=1e-6)


def test_summaries_no_fe_returns_empty(no_iron_cif: Path):
    bundle = read_cif(no_iron_cif)
    result = analyse(bundle, cif_filename="no_iron.cif")
    assert summaries_for_cif("no_iron.cif", bundle, result) == []


# ---------------------------------------------------------------------
# 2) build_cohort_dataframe — column structure + BVS application
# ---------------------------------------------------------------------

def test_cohort_dataframe_columns_canonical(fe_octahedral_cif: Path):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename="fe_octahedral.cif")
    summaries = summaries_for_cif("fe_octahedral.cif", bundle, result)
    df = build_cohort_dataframe(summaries)
    assert list(df.columns) == COHORT_COLUMNS


def test_cohort_dataframe_bvs_blank_when_oxidation_unknown(
    fe_octahedral_cif: Path,
):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename="fe_octahedral.cif")
    summaries = summaries_for_cif("fe_octahedral.cif", bundle, result)
    df = build_cohort_dataframe(summaries)  # no annotations supplied
    assert df.loc[0, COL_BVS] is None or pd.isna(df.loc[0, COL_BVS])


def test_cohort_dataframe_bvs_filled_when_oxidation_annotated(
    fe_octahedral_cif: Path,
):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename="fe_octahedral.cif")
    summaries = summaries_for_cif("fe_octahedral.cif", bundle, result)
    df = build_cohort_dataframe(
        summaries,
        per_file_annotations={
            "fe_octahedral.cif": {"oxidation_state": "Fe(II)",
                                  "spin_state": "LS"},
        },
    )
    bvs = df.loc[0, COL_BVS]
    assert bvs is not None and not pd.isna(bvs)
    # d = 2.0 Å × 6 with R₀ = 1.78 (Fe(II)-LS) gives s ≈ 0.547, BVS ≈ 3.28.
    assert 3.0 < bvs < 3.5


# ---------------------------------------------------------------------
# 3) auto_assign_series — same formula → same letter
# ---------------------------------------------------------------------

def test_auto_assign_series_groups_by_formula():
    df = pd.DataFrame({
        COL_FILE:    ["a.cif", "b.cif", "c.cif", "d.cif"],
        COL_FORMULA: ["C2 H4 Fe N6", "C2 H4 Fe N6",
                      "C5 H10 Fe N3", "C2 H4 Fe N6"],
        COL_SERIES:  ["", "", "", ""],
    })
    # Pad missing columns so build_cohort doesn't need to construct it.
    for c in COHORT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COHORT_COLUMNS]

    out = auto_assign_series(df)
    # Same formula → same letter; first-seen wins A.
    assert out.loc[0, COL_SERIES] == "A"
    assert out.loc[1, COL_SERIES] == "A"
    assert out.loc[2, COL_SERIES] == "B"
    assert out.loc[3, COL_SERIES] == "A"


def test_auto_assign_series_missing_formula_gets_question_mark():
    df = pd.DataFrame([
        {COL_FILE: "x.cif", COL_FORMULA: "—",                   COL_SERIES: ""},
        {COL_FILE: "y.cif", COL_FORMULA: "",                    COL_SERIES: ""},
        {COL_FILE: "z.cif", COL_FORMULA: "PARSE FAILED: nope",  COL_SERIES: ""},
        {COL_FILE: "w.cif", COL_FORMULA: "Fe N6",               COL_SERIES: ""},
    ])
    for c in COHORT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COHORT_COLUMNS]
    out = auto_assign_series(df)
    assert out.loc[0, COL_SERIES] == "?"
    assert out.loc[1, COL_SERIES] == "?"
    assert out.loc[2, COL_SERIES] == "?"
    assert out.loc[3, COL_SERIES] == "A"


def test_auto_assign_series_handles_more_than_26_formulas():
    """Spreadsheet-style overflow: 27th unique formula → AA."""
    rows = []
    for i in range(28):
        rows.append({
            COL_FILE: f"f{i}.cif",
            COL_FORMULA: f"Fe N{i}",   # all distinct
            COL_SERIES: "",
        })
    df = pd.DataFrame(rows)
    for c in COHORT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COHORT_COLUMNS]

    out = auto_assign_series(df)
    assert out.loc[0,  COL_SERIES] == "A"
    assert out.loc[25, COL_SERIES] == "Z"
    assert out.loc[26, COL_SERIES] == "AA"
    assert out.loc[27, COL_SERIES] == "AB"


# ---------------------------------------------------------------------
# 4) Error rows surface in the table rather than silently dropping
# ---------------------------------------------------------------------

def test_error_summary_renders_a_visible_row():
    err = make_error_summary("broken.cif", "ValueError(...)")
    df = build_cohort_dataframe([err])
    assert len(df) == 1
    assert df.loc[0, COL_FILE] == "broken.cif"
    assert "PARSE FAILED" in df.loc[0, COL_FORMULA]
    # BVS is None for the error row.
    assert df.loc[0, COL_BVS] is None or pd.isna(df.loc[0, COL_BVS])
