"""
Tests for the disorder pipeline:

  parse_disorder_atoms()   — pulls partial-occupancy atoms out of the
                             _atom_site loop, with assembly/group tags.
  read_cif().disorder_atoms — surfaces the list on the CifBundle.
  analyse(min_occupancy)   — filters bonds whose atoms are below the
                             occupancy threshold.

The fixture `fe_disordered.cif` has one N atom split across two
disorder components: N3 (0.7) and N3B (0.3), in assembly A, groups
1 and 2 respectively.
"""

from __future__ import annotations
from pathlib import Path

import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src.cif_reader import (
    DisorderAtom,
    parse_disorder_atoms,
    read_cif,
)
from src.fe_n_analysis import analyse
from src.data_schema import COL_FE_LABEL, COL_N_LABEL


# ---------------------------------------------------------------------
# Pure parser
# ---------------------------------------------------------------------

def test_parse_disorder_atoms_finds_partial_occupancies(fe_disordered_cif: Path):
    out = parse_disorder_atoms(fe_disordered_cif)
    labels = {a.label for a in out}
    assert "N3"  in labels
    assert "N3B" in labels
    # Atoms at full occupancy should NOT be reported.
    assert "Fe1" not in labels
    assert "N1"  not in labels


def test_parse_disorder_atoms_records_occupancy_and_tags(fe_disordered_cif):
    out = {a.label: a for a in parse_disorder_atoms(fe_disordered_cif)}
    n3 = out["N3"]
    assert isinstance(n3, DisorderAtom)
    assert n3.occupancy == pytest.approx(0.7, abs=1e-9)
    assert n3.disorder_assembly == "A"
    assert n3.disorder_group    == "1"
    n3b = out["N3B"]
    assert n3b.occupancy == pytest.approx(0.3, abs=1e-9)
    assert n3b.disorder_assembly == "A"
    assert n3b.disorder_group    == "2"


def test_parse_disorder_atoms_empty_for_full_occupancy_cif(fe_octahedral_cif):
    """The unflagged-disorder fixture has every atom at occ = 1."""
    assert parse_disorder_atoms(fe_octahedral_cif) == []


# ---------------------------------------------------------------------
# Bundle integration
# ---------------------------------------------------------------------

def test_read_cif_attaches_disorder_atoms(fe_disordered_cif):
    bundle = read_cif(fe_disordered_cif)
    labels = {a.label for a in bundle.disorder_atoms}
    assert labels == {"N3", "N3B"}


def test_read_cif_disorder_atoms_empty_when_no_disorder(fe_octahedral_cif):
    bundle = read_cif(fe_octahedral_cif)
    assert bundle.disorder_atoms == []


# ---------------------------------------------------------------------
# analyse() filtering
# ---------------------------------------------------------------------

def test_analyse_major_only_drops_minor_partner(fe_disordered_cif):
    """With min_occupancy = 0.5, the N3B partner (occ 0.3) is dropped
    from the analysis. N3 (occ 0.7) survives.

    The exact number of formal bonds depends on what ccdc's
    bond-perception decides about a partial-occupancy atom — both
    'N3 only' and 'N3+N3B' are reasonable. The invariant we test is
    that the major-only result has *no more* bonds than the all-
    components result, and that no N3B partner appears at min_occ=0.5.
    """
    bundle = read_cif(fe_disordered_cif)
    major = analyse(bundle, cif_filename="x", min_occupancy=0.5)
    all_c = analyse(bundle, cif_filename="x", min_occupancy=0.0)

    n_partners_major = set(major.bonds[COL_N_LABEL]) if not major.bonds.empty else set()
    n_partners_all   = set(all_c.bonds[COL_N_LABEL]) if not all_c.bonds.empty else set()

    # 'major' must be a subset (the filter can only remove rows).
    assert n_partners_major.issubset(n_partners_all)
    # The minor partner N3B must NOT appear in the major-only result.
    assert "N3B" not in n_partners_major


def test_analyse_records_disorder_filter_warning(fe_disordered_cif):
    """When min_occupancy filters out an atom, the user-facing warning
    list should say so by name."""
    bundle = read_cif(fe_disordered_cif)
    result = analyse(bundle, cif_filename="x", min_occupancy=0.5)
    joined = "\n".join(result.warnings)
    # Either N3B (occ 0.3, below 0.5) is named, or the generic
    # 'Disorder filter ... excluded' message is present. Both are fine.
    assert "Disorder filter" in joined or "N3B" in joined


def test_analyse_default_min_occupancy_is_inclusive(fe_disordered_cif):
    """min_occupancy defaults to 0.0 → no atom dropped on this fixture."""
    bundle = read_cif(fe_disordered_cif)
    result = analyse(bundle, cif_filename="x")  # default min_occupancy = 0.0
    # No disorder-filter warning should be emitted at the inclusive default.
    assert not any("Disorder filter" in w for w in result.warnings)
