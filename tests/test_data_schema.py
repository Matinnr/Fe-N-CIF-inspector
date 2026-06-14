"""Tests for src/data_schema.py — the pure-chemistry helpers."""

from __future__ import annotations
import streamlit  # noqa: F401  (import-order requirement)

from src.data_schema import (
    SPIN_BANDS, select_bands, suggest_spin_state,
)


# ---- select_bands -----------------------------------------------------

def test_select_bands_fe_ii_returns_two_bands():
    bands = select_bands("Fe(II)")
    assert set(bands.keys()) == {"Fe(II) LS", "Fe(II) HS"}


def test_select_bands_fe_iii_returns_two_bands():
    bands = select_bands("Fe(III)")
    assert set(bands.keys()) == {"Fe(III) LS", "Fe(III) HS"}


def test_select_bands_unknown_returns_combined():
    bands = select_bands(None)
    # Generic regions, no oxidation-specific labels.
    assert "LS region" in bands
    assert "HS region" in bands
    assert all("Fe(" not in k for k in bands)


def test_select_bands_labels_dont_overlap():
    """The whole point of this refactor: chosen bands' midpoints must
    be far enough apart that labels don't collide on the chart edge."""
    for ox in ("Fe(II)", "Fe(III)", None):
        midpoints = sorted((lo + hi) / 2 for lo, hi in select_bands(ox).values())
        # Two bands per call; their midpoints must differ by > 0.05 Å.
        assert len(midpoints) == 2
        assert midpoints[1] - midpoints[0] > 0.05


# ---- suggest_spin_state -----------------------------------------------

def test_suggest_short_fe_ii_is_low_spin():
    s = suggest_spin_state(1.97, "Fe(II)")
    assert "low-spin" in s


def test_suggest_long_fe_ii_is_high_spin():
    s = suggest_spin_state(2.18, "Fe(II)")
    assert "high-spin" in s


def test_suggest_borderline_fe_ii_is_borderline():
    s = suggest_spin_state(2.07, "Fe(II)")
    assert "borderline" in s.lower() or "equilibrium" in s.lower()


def test_suggest_short_fe_iii_is_low_spin():
    s = suggest_spin_state(1.97, "Fe(III)")
    assert "low-spin" in s


def test_suggest_long_fe_iii_is_high_spin():
    s = suggest_spin_state(2.10, "Fe(III)")
    assert "high-spin" in s


def test_suggest_no_oxidation_returns_none():
    """Without an oxidation state we cannot guess — return None,
    don't fabricate."""
    assert suggest_spin_state(2.0, None) is None
    assert suggest_spin_state(2.0, "(unknown)") is None
    assert suggest_spin_state(2.0, "Fe(IV)") is None
