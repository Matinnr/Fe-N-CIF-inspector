"""
Tests for src/analysed_store.py — session-scoped CIF state shared
between modes. Pure session-state mutation tests; no ccdc, no
Streamlit widget rendering.
"""

from __future__ import annotations

import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src import analysed_store as store


@pytest.fixture(autouse=True)
def _clean_store_between_tests():
    """Reset the session-state store before AND after every test so
    each one runs against an empty slate."""
    store.clear()
    yield
    store.clear()


# ----------------------------------------------------------------------
# record / get / all_analysed / remove / clear
# ----------------------------------------------------------------------

def test_get_returns_none_when_empty():
    assert store.get("nope.cif") is None


def test_record_then_get_returns_what_was_stored():
    store.record("a.cif",
                 content_hash="h_a", cif_bytes=b"x",
                 temperature_K=100.0, fe_centres={"Fe1": {"n_FeN": 6}})
    entry = store.get("a.cif")
    assert entry is not None
    assert entry["content_hash"] == "h_a"
    assert entry["cif_bytes"] == b"x"
    assert entry["temperature_K"] == 100.0
    assert entry["fe_centres"] == {"Fe1": {"n_FeN": 6}}


def test_record_updates_existing_entry_in_place():
    store.record("a.cif", content_hash="h1", temperature_K=100.0)
    store.record("a.cif", content_hash="h1",
                 fe_centres={"Fe1": {"n_FeN": 6}})
    entry = store.get("a.cif")
    # Both writes should be merged.
    assert entry["temperature_K"] == 100.0
    assert "fe_centres" in entry


def test_all_analysed_returns_snapshot():
    store.record("a.cif", content_hash="h_a")
    store.record("b.cif", content_hash="h_b")
    out = store.all_analysed()
    assert set(out.keys()) == {"a.cif", "b.cif"}
    # Mutating the returned dict must not affect the store.
    out["c.cif"] = {}
    assert "c.cif" not in store.all_analysed()


def test_remove_drops_one_entry():
    store.record("a.cif", content_hash="h_a")
    store.record("b.cif", content_hash="h_b")
    store.remove("a.cif")
    assert store.get("a.cif") is None
    assert store.get("b.cif") is not None


def test_clear_wipes_all():
    store.record("a.cif", content_hash="h_a")
    store.record("b.cif", content_hash="h_b")
    store.clear()
    assert store.all_analysed() == {}


# ----------------------------------------------------------------------
# record_from_analysis — the convenience helper that extracts Fe-centre
# metrics from an AnalysisResult-shaped object.
# ----------------------------------------------------------------------

class _FakeOctaDistortion:
    """Mimics OctaDistortion's attribute surface for the store helper."""
    def __init__(self, zeta=None, delta=None, sigma=None, theta=None):
        self.zeta = zeta
        self.delta = delta
        self.sigma = sigma
        self.theta = theta


class _FakeCoordGeom:
    def __init__(self, classification):
        self.classification = classification


class _FakeProv:
    def __init__(self, refcode=None, formula=None, doi=None):
        self.ccdc_refcode = refcode
        self.chemical_formula = formula
        self.doi = doi


class _FakeBundle:
    def __init__(self, temperature_K=None, provenance=None):
        self.temperature_K = temperature_K
        self.provenance = provenance


def _fake_result(fe_label="Fe1",
                 distances=(1.97, 1.98, 1.99, 2.00, 2.01, 2.02),
                 sigma=12.7, theta=17.1, zeta=0.06,
                 coord="octahedral"):
    """Build a minimal AnalysisResult-shaped stand-in with one Fe centre."""
    import pandas as pd
    from src.data_schema import (
        COL_DETECTION_METHOD, COL_DISTANCE, COL_FE_LABEL,
        COL_N_LABEL, DetectionMethod,
    )

    rows = []
    for i, d in enumerate(distances):
        rows.append({
            COL_FE_LABEL:         fe_label,
            COL_N_LABEL:          f"N{i+1}",
            COL_DISTANCE:         d,
            COL_DETECTION_METHOD: DetectionMethod.FORMAL_BOND.value,
        })
    df = pd.DataFrame(rows)

    class _R:
        bonds = df
        geometry = {fe_label: _FakeOctaDistortion(zeta, 0.30e-4, sigma, theta)}
        coord_geom = {fe_label: _FakeCoordGeom(coord)}
    return _R()


def test_record_from_analysis_persists_fe_centre_metrics():
    bundle = _FakeBundle(
        temperature_K=100.0,
        provenance=_FakeProv(refcode="CCDC 999", formula="Fe N6",
                             doi="10.x/yyy"),
    )
    result = _fake_result()
    store.record_from_analysis("demo.cif", b"raw bytes",
                               bundle, result)

    entry = store.get("demo.cif")
    assert entry is not None
    assert entry["temperature_K"] == 100.0
    assert entry["refcode"] == "CCDC 999"
    assert entry["formula"] == "Fe N6"
    assert entry["doi"] == "10.x/yyy"

    fe = entry["fe_centres"]["Fe1"]
    assert fe["n_FeN"] == 6
    assert fe["mean_FeN_A"] == pytest.approx(1.995, abs=1e-3)
    assert fe["sigma_deg"] == pytest.approx(12.7)
    assert fe["theta_deg"] == pytest.approx(17.1)
    assert fe["coordination"] == "octahedral"


def test_record_from_analysis_handles_no_fe_centres():
    """A CIF with no N-coordinated Fe should still be recorded so
    Mode 3 can show 'this was analysed, no Fe-N here' rather than
    silently dropping the upload."""
    bundle = _FakeBundle(provenance=_FakeProv(refcode="CCDC zero"))
    class _Empty:
        import pandas as pd
        from src.data_schema import BOND_TABLE_COLUMNS
        bonds = pd.DataFrame(columns=BOND_TABLE_COLUMNS)
        geometry = {}
        coord_geom = {}
    store.record_from_analysis("empty.cif", b"raw", bundle, _Empty())
    entry = store.get("empty.cif")
    assert entry is not None
    assert entry["fe_centres"] == {}
    assert entry["refcode"] == "CCDC zero"


def test_record_from_analysis_overwrites_on_repeat_call():
    """Re-analysing the same filename (e.g. user re-uploaded after
    editing) should refresh the entry, not duplicate it."""
    bundle = _FakeBundle(temperature_K=100.0)
    store.record_from_analysis("same.cif", b"v1", bundle, _fake_result())
    # Re-record with a different temperature.
    bundle2 = _FakeBundle(temperature_K=293.0)
    store.record_from_analysis("same.cif", b"v2", bundle2, _fake_result())
    entry = store.get("same.cif")
    assert entry["temperature_K"] == 293.0
    assert entry["cif_bytes"] == b"v2"
    # Single entry, not two.
    assert len(store.all_analysed()) == 1
