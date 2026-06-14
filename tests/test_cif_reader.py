"""
Tests for src/cif_reader.py

These run against tiny synthetic CIFs in tests/fixtures/.
They verify that the reader returns the right data structure and
surfaces warnings for known edge cases.
"""

from __future__ import annotations
from pathlib import Path

import pytest

# Streamlit is imported by app.py before ccdc; tests don't need it,
# but importing it first matches production import order on macOS.
import streamlit  # noqa: F401

from src.cif_reader import read_cif, CifBundle, extract_temperature


def test_reads_valid_cif(fe_octahedral_cif: Path):
    """A valid CIF returns a CifBundle with a molecule and an id."""
    bundle = read_cif(fe_octahedral_cif)
    assert isinstance(bundle, CifBundle)
    assert bundle.molecule is not None
    assert bundle.structure_id  # non-empty
    assert bundle.crystal is not None


def test_temperature_extracted(fe_octahedral_cif: Path):
    """`temperature_K` is parsed from the CIF header."""
    bundle = read_cif(fe_octahedral_cif)
    assert bundle.temperature_K == pytest.approx(100.0)


def test_temperature_missing_does_not_crash(no_temperature_cif: Path):
    """When the header has no temperature, `temperature_K` is None and
    we get a warning — but no exception."""
    bundle = read_cif(no_temperature_cif)
    assert bundle.temperature_K is None
    assert any("temperature" in w.lower() for w in bundle.warnings)


def test_missing_file_raises():
    """Pointing at a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_cif("/tmp/does_not_exist_4242.cif")


def test_parses_bond_esds_from_geom_bond_loop(fe_with_esds_cif: Path):
    """The fe_with_esds fixture has Fe1–N1 1.976(8) in its
    _geom_bond_distance loop. read_cif should surface the esd."""
    from src.cif_reader import parse_geom_bond_distances
    esds = parse_geom_bond_distances(fe_with_esds_cif)
    assert frozenset(["Fe1", "N1"]) in esds
    value, esd = esds[frozenset(["Fe1", "N1"])]
    assert value == pytest.approx(1.976)
    assert esd  == pytest.approx(0.008)
    # Fe1-N5 in the fixture is 2.001(11) → esd 0.011
    value5, esd5 = esds[frozenset(["Fe1", "N5"])]
    assert value5 == pytest.approx(2.001)
    assert esd5  == pytest.approx(0.011)


def test_bundle_carries_bond_esds(fe_with_esds_cif: Path):
    bundle = read_cif(fe_with_esds_cif)
    assert frozenset(["Fe1", "N3"]) in bundle.bond_esds
    _, esd = bundle.bond_esds[frozenset(["Fe1", "N3"])]
    assert esd == pytest.approx(0.009)


def test_no_geom_bond_loop_gives_empty_dict(fe_octahedral_cif: Path):
    """The original test fixture has no _geom_bond_distance loop;
    parsing should return {} without crashing."""
    from src.cif_reader import parse_geom_bond_distances
    assert parse_geom_bond_distances(fe_octahedral_cif) == {}
