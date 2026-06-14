"""
Tests for src/fe_n_analysis.py

We feed the analyser tiny synthetic CIFs and check the resulting table
shape, types, and edge-case handling.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest
import streamlit  # noqa: F401  (import-order requirement on macOS)

from src.cif_reader import read_cif
from src.fe_n_analysis import analyse, find_fe_atoms, find_n_atoms
from src.data_schema import (
    BOND_TABLE_COLUMNS, COL_DETECTION_METHOD, COL_DISTANCE,
    COL_FE_LABEL, COL_N_LABEL, DetectionMethod,
)


# ---------------------------------------------------------------
# Happy path — Fe with 6 N neighbours
# ---------------------------------------------------------------

def test_returns_dataframe(fe_octahedral_cif: Path):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename=fe_octahedral_cif.name)
    assert isinstance(result.bonds, pd.DataFrame)


def test_expected_columns_present(fe_octahedral_cif: Path):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename=fe_octahedral_cif.name)
    for col in BOND_TABLE_COLUMNS:
        assert col in result.bonds.columns, f"missing column: {col}"


def test_distances_are_numeric(fe_octahedral_cif: Path):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename=fe_octahedral_cif.name)
    assert pd.api.types.is_numeric_dtype(result.bonds[COL_DISTANCE])
    # Each Fe–N is exactly 2.0 Å in the synthetic fixture (0.2 fractional × 10 Å cell).
    assert all(abs(d - 2.0) < 0.001 for d in result.bonds[COL_DISTANCE])


def test_six_distances_for_octahedral(fe_octahedral_cif: Path):
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename=fe_octahedral_cif.name)
    # Six Fe–N pairs in an octahedral FeN6 fixture.
    assert len(result.bonds) == 6
    assert result.n_fe == 1
    assert result.n_n == 6


def test_detection_method_values_valid(fe_octahedral_cif: Path):
    """Every row's detection_method is one of the four enum values."""
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename=fe_octahedral_cif.name)
    valid = {m.value for m in DetectionMethod}
    assert set(result.bonds[COL_DETECTION_METHOD]).issubset(valid)


# ---------------------------------------------------------------
# Edge cases — graceful degradation
# ---------------------------------------------------------------

def test_no_iron_returns_empty_with_warning(no_iron_cif: Path):
    bundle = read_cif(no_iron_cif)
    result = analyse(bundle, cif_filename=no_iron_cif.name)
    assert result.bonds.empty
    assert result.n_fe == 0
    assert any("no fe" in w.lower() for w in result.warnings)


def test_iron_no_nitrogen_returns_empty_with_warning(iron_no_nitrogen_cif: Path):
    bundle = read_cif(iron_no_nitrogen_cif)
    result = analyse(bundle, cif_filename=iron_no_nitrogen_cif.name)
    assert result.bonds.empty
    assert result.n_fe == 1
    assert result.n_n == 0
    assert any("no n atoms" in w.lower() or "no n " in w.lower()
               for w in result.warnings)


def test_no_temperature_does_not_crash(no_temperature_cif: Path):
    """If the CIF lacks _diffrn_ambient_temperature, analysis still runs."""
    bundle = read_cif(no_temperature_cif)
    result = analyse(bundle, cif_filename=no_temperature_cif.name)
    # Either we got bonds (with NaN temp column) or none — but no crash.
    assert isinstance(result.bonds, pd.DataFrame)


def test_atom_finders(fe_octahedral_cif: Path):
    bundle = read_cif(fe_octahedral_cif)
    fe_atoms = find_fe_atoms(bundle.molecule)
    n_atoms = find_n_atoms(bundle.molecule)
    assert len(fe_atoms) == 1
    assert len(n_atoms) == 6
    assert all(a.atomic_symbol == "Fe" for a in fe_atoms)
    assert all(a.atomic_symbol == "N" for a in n_atoms)


# ---------------------------------------------------------------
# Honesty checks — no silent spin/oxidation guessing
# ---------------------------------------------------------------

def test_no_silent_spin_state_assignment(fe_octahedral_cif: Path):
    """The analyser must not invent spin_state / oxidation_state columns."""
    bundle = read_cif(fe_octahedral_cif)
    result = analyse(bundle, cif_filename=fe_octahedral_cif.name)
    assert "spin_state" not in result.bonds.columns
    assert "oxidation_state" not in result.bonds.columns


# ---------------------------------------------------------------
# Cutoff sensitivity
# ---------------------------------------------------------------

def test_cutoff_filters_geometric_candidates(fe_octahedral_cif: Path):
    """The cutoff applies to GEOMETRIC candidates.

    Formal bond perception (`assign_bond_types`) doesn't use the cutoff —
    it has its own chemistry-based rules. So a 1.5 Å cutoff with geometric
    candidates ON should *not* eliminate the formal bonds, only restrict
    what extra rows show up beyond them.
    """
    bundle = read_cif(fe_octahedral_cif)
    tight = analyse(
        bundle, cif_filename=fe_octahedral_cif.name,
        cutoff_A=1.5, include_geometric=True, include_symmetry=False,
    )
    wide = analyse(
        bundle, cif_filename=fe_octahedral_cif.name,
        cutoff_A=3.0, include_geometric=True, include_symmetry=False,
    )
    # Tight cutoff finds <= the same number of rows as a wide one.
    assert len(tight.bonds) <= len(wide.bonds)
    # Every row reported under a tight cutoff is a formal bond
    # (geometric candidates would have been filtered out).
    if not tight.bonds.empty:
        from src.data_schema import COL_DETECTION_METHOD, DetectionMethod
        assert set(tight.bonds[COL_DETECTION_METHOD]) == {
            DetectionMethod.FORMAL_BOND.value
        }
