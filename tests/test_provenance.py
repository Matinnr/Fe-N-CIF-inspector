"""
Tests for the Provenance pipeline:

  parse_provenance()              — pulls every relevant tag from the CIF
  CifBundle.provenance            — surfaces the result on the bundle
  _refinement_quality_badge()     — classifies R / data-per-parameter
                                    into good / marginal / warning

The full-metadata fixture (fe_with_provenance_cif) is built so the
quality badge fires GREEN: R = 3.50%, data/parameter = 15. The
sparse-metadata fixture (fe_octahedral_cif) tests the all-None
fallback path.
"""

from __future__ import annotations
from pathlib import Path

import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src.cif_reader import (
    Provenance,
    parse_provenance,
    read_cif,
)
from src.modes.single_cif import _refinement_quality_badge


# ----------------------------------------------------------------------
# 1) Parser — full metadata fixture
# ----------------------------------------------------------------------

def test_parse_provenance_extracts_every_field(fe_with_provenance_cif: Path):
    p = parse_provenance(fe_with_provenance_cif)
    assert isinstance(p, Provenance)
    assert p.ccdc_refcode    == "CCDC 999999"
    assert p.chemical_formula == "C2 H4 Fe N6"
    assert p.space_group     == "P -1"
    assert p.Z               == 1
    assert p.R_factor_gt     == pytest.approx(0.0350)
    assert p.R_factor_all    == pytest.approx(0.0450)
    assert p.wR_factor_ref   == pytest.approx(0.0950)
    assert p.wR_factor_gt    == pytest.approx(0.0900)
    assert p.goodness_of_fit == pytest.approx(1.030)
    assert p.n_reflns        == 1500
    assert p.n_parameters    == 100


def test_parse_provenance_prefers_citation_doi_over_audit_fallback(
    fe_with_provenance_cif: Path,
):
    """The audit_block_doi is the CCDC archive DOI (10.5517/...).
    The citation_doi is the paper DOI (10.9999/...). We must pick the
    paper, not the archive."""
    p = parse_provenance(fe_with_provenance_cif)
    assert p.doi == "10.9999/test.demo.doi"


# ----------------------------------------------------------------------
# 2) Parser — sparse fixture
# ----------------------------------------------------------------------

def test_parse_provenance_returns_all_none_for_sparse_cif(
    fe_octahedral_cif: Path,
):
    """The minimal-test fixture only carries cell + atom_site data."""
    p = parse_provenance(fe_octahedral_cif)
    assert p.ccdc_refcode    is None
    assert p.R_factor_gt     is None
    assert p.doi             is None
    assert p.n_parameters    is None


# ----------------------------------------------------------------------
# 3) Bundle integration
# ----------------------------------------------------------------------

def test_read_cif_attaches_provenance(fe_with_provenance_cif: Path):
    bundle = read_cif(fe_with_provenance_cif)
    assert bundle.provenance is not None
    assert bundle.provenance.ccdc_refcode == "CCDC 999999"


def test_read_cif_attaches_empty_provenance_for_sparse_cif(
    fe_octahedral_cif: Path,
):
    bundle = read_cif(fe_octahedral_cif)
    assert bundle.provenance is not None
    assert bundle.provenance.ccdc_refcode is None


# ----------------------------------------------------------------------
# 4) Quality badge logic — the three bands
# ----------------------------------------------------------------------

@pytest.mark.parametrize("R, n_reflns, n_params, expected_level", [
    # GREEN: R < 5% AND data/param > 8
    (0.0350, 1500, 100, "good"),       # R=3.5%, dp=15
    (0.0499, 900,  100, "good"),       # R=4.99%, dp=9
    # RED: R > 10%
    (0.1010, 5000, 100, "warning"),    # R=10.1% — just over
    (0.2500, 5000, 100, "warning"),
    # YELLOW (marginal): everything else
    (0.0500, 1500, 100, "marginal"),   # R=5% boundary → not good
    (0.0774, 9710, 581, "marginal"),   # the 254204 case → marginal
    (0.0350, 700,  100, "marginal"),   # R good, dp=7 (≤ 8) → marginal
    (0.0900, 5000, 100, "marginal"),   # R<10 but >5 → marginal
])
def test_refinement_quality_badge_thresholds(
    R, n_reflns, n_params, expected_level,
):
    level, _msg = _refinement_quality_badge(R, n_reflns, n_params)
    assert level == expected_level


def test_quality_badge_no_R_factor_is_marginal():
    """A CIF without R-factor information should fall in the marginal
    band with a clear 'unable to assess' message."""
    level, msg = _refinement_quality_badge(None, 1000, 100)
    assert level == "marginal"
    assert "unable to assess" in msg.lower()


def test_quality_badge_missing_dp_doesnt_block_green():
    """If n_reflns / n_parameters aren't reported, a low R alone
    should still qualify as 'good' (we can't fault the dp_ratio when
    we don't know it)."""
    level, _msg = _refinement_quality_badge(0.030, None, None)
    assert level == "good"


# ----------------------------------------------------------------------
# 5) Quality badge — the actual 254204 case the user uploaded
# ----------------------------------------------------------------------

def test_quality_badge_for_real_254204_case():
    """Real numbers from the iron porphyrin CIF the user uploaded:
       R(gt) = 0.0774, 9710 reflns, 581 parameters → marginal."""
    level, msg = _refinement_quality_badge(0.0774, 9710, 581)
    assert level == "marginal"
    # The message should quote R as a percent and include the
    # data/parameter ratio in human-readable form.
    assert "7.74" in msg
    assert "16.7" in msg or "16.7%" in msg or "data/parameter" in msg


# ----------------------------------------------------------------------
# 6) Defensive rendering — cache-stale CifBundle without `provenance`
# ----------------------------------------------------------------------

class _LegacyBundle:
    """Stand-in for a CifBundle from before the `provenance` field
    existed — what @st.cache_resource hangs onto across hot-reloads."""
    def __init__(self):
        self.temperature_K = 100.0


def test_render_provenance_does_not_crash_on_legacy_bundle():
    """Regression: when Streamlit's cache returns a pre-Prompt-7
    bundle, the panel should silently skip rather than AttributeError."""
    from src.modes.single_cif import _render_provenance
    # Should not raise — even though _LegacyBundle has no .provenance.
    _render_provenance(_LegacyBundle())
