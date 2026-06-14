"""
Shared pytest fixtures and path setup.

Why this exists: pytest discovers `conftest.py` automatically and applies
its fixtures to every test in this directory. Putting the import-path
hack here means we don't repeat it in every test file.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pytest

# Make `from src.something import x` resolve when running pytest from
# anywhere — we just add the parent (the repo root) to sys.path.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))


FIXTURE_DIR = HERE / "fixtures"


@pytest.fixture
def fe_octahedral_cif() -> Path:
    """A synthetic FeN6 octahedral structure with T = 100 K."""
    return FIXTURE_DIR / "fe_octahedral.cif"


@pytest.fixture
def no_iron_cif() -> Path:
    """A nitrogen-containing CIF with no iron atoms."""
    return FIXTURE_DIR / "no_iron.cif"


@pytest.fixture
def iron_no_nitrogen_cif() -> Path:
    """An iron-containing CIF with no nitrogen atoms."""
    return FIXTURE_DIR / "iron_no_nitrogen.cif"


@pytest.fixture
def no_temperature_cif() -> Path:
    """An FeN6 CIF whose header omits temperature fields."""
    return FIXTURE_DIR / "no_temperature.cif"


@pytest.fixture
def fe_with_esds_cif() -> Path:
    """An FeN6 CIF whose _geom_bond_distance loop carries esds."""
    return FIXTURE_DIR / "fe_with_esds.cif"


@pytest.fixture
def fe_disordered_cif() -> Path:
    """FeN6 CIF with one N atom split across two disorder components
    (N3 occupancy 0.7 / N3B occupancy 0.3, assembly A, groups 1 and 2)."""
    return FIXTURE_DIR / "fe_disordered.cif"


@pytest.fixture
def fe_with_provenance_cif() -> Path:
    """FeN6 CIF with full bibliographic + refinement metadata:
       CCDC refcode, DOI (in citation loop), formula, space group,
       Z, T, R / wR / GooF / reflns / parameters."""
    return FIXTURE_DIR / "fe_with_provenance.cif"
