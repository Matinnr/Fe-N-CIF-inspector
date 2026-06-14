"""
Tests for src/library.py — the reference-library loader.

Covers:
  - happy path: well-formed manifest → list of LibraryEntry
  - empty / missing manifest → empty list (UI shows a hint)
  - bad JSON / missing fields → LibraryError
  - cif_exists flag set correctly for present vs missing files
  - expected_vs_measured tolerance comparison
"""

from __future__ import annotations
import json
from pathlib import Path

import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src.library import (
    LibraryEntry,
    LibraryError,
    expected_vs_measured,
    load_library,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def library_dir(tmp_path: Path) -> Path:
    """A temp directory standing in for `data/reference/`."""
    return tmp_path / "library"


def _write_manifest(library_dir: Path, payload: dict) -> Path:
    library_dir.mkdir(parents=True, exist_ok=True)
    manifest = library_dir / "library.json"
    manifest.write_text(json.dumps(payload, indent=2))
    return manifest


# ---------------------------------------------------------------------
# 1) Happy path
# ---------------------------------------------------------------------

def test_load_library_returns_entries(library_dir: Path):
    # CIF file exists; entry should be marked cif_exists=True.
    cif = library_dir / "test.cif"
    library_dir.mkdir(parents=True, exist_ok=True)
    cif.write_text("data_test\n")
    _write_manifest(library_dir, {
        "schema_version": "1",
        "entries": [{
            "id": "test",
            "cif_filename": "test.cif",
            "display_name": "Test entry",
            "compound_class": "test class",
            "oxidation_state": "Fe(II)",
            "spin_state": "LS",
            "temperature_K": 100,
            "cod_id": "1234567",
            "doi": "10.1234/test",
            "license": "CC0",
            "description": "demo",
            "teaching_notes": ["note 1", "note 2"],
            "expected_metrics": {"mean_FeN_A": 1.97},
        }],
    })

    entries = load_library(library_dir)
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e, LibraryEntry)
    assert e.id == "test"
    assert e.cif_exists is True
    assert e.cif_path == cif
    assert e.oxidation_state == "Fe(II)"
    assert e.spin_state == "LS"
    assert e.temperature_K == pytest.approx(100.0)
    assert e.cod_id == "1234567"
    assert e.teaching_notes == ["note 1", "note 2"]
    assert e.expected_metrics == {"mean_FeN_A": pytest.approx(1.97)}


def test_load_library_marks_missing_cifs(library_dir: Path):
    """Entry can be present in the manifest but its CIF missing on disk."""
    library_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(library_dir, {
        "schema_version": "1",
        "entries": [{
            "id": "ghost",
            "cif_filename": "i_do_not_exist.cif",
            "display_name": "Aspirational entry",
        }],
    })
    entries = load_library(library_dir)
    assert len(entries) == 1
    assert entries[0].cif_exists is False


# ---------------------------------------------------------------------
# 2) Empty / absent manifest
# ---------------------------------------------------------------------

def test_load_library_returns_empty_when_no_folder(tmp_path: Path):
    """A non-existent library directory returns []."""
    assert load_library(tmp_path / "nope") == []


def test_load_library_returns_empty_when_no_manifest(library_dir: Path):
    library_dir.mkdir(parents=True, exist_ok=True)
    # Folder exists but library.json doesn't
    assert load_library(library_dir) == []


# ---------------------------------------------------------------------
# 3) Malformed manifest
# ---------------------------------------------------------------------

def test_bad_json_raises(library_dir: Path):
    library_dir.mkdir(parents=True, exist_ok=True)
    (library_dir / "library.json").write_text("{not valid json")
    with pytest.raises(LibraryError, match="not valid JSON"):
        load_library(library_dir)


def test_missing_entries_array_raises(library_dir: Path):
    _write_manifest(library_dir, {"schema_version": "1"})
    with pytest.raises(LibraryError, match="entries"):
        load_library(library_dir)


def test_entries_must_be_array(library_dir: Path):
    _write_manifest(library_dir, {
        "schema_version": "1",
        "entries": "not an array",
    })
    with pytest.raises(LibraryError, match="must be an array"):
        load_library(library_dir)


def test_entry_missing_required_field_raises(library_dir: Path):
    _write_manifest(library_dir, {
        "schema_version": "1",
        "entries": [{"id": "x"}],   # missing cif_filename + display_name
    })
    with pytest.raises(LibraryError, match="missing required"):
        load_library(library_dir)


# ---------------------------------------------------------------------
# 4) expected_vs_measured
# ---------------------------------------------------------------------

def test_expected_vs_measured_within_tolerance():
    out = expected_vs_measured(
        expected={"mean_FeN_A": 1.97, "sigma_deg_approx": 40},
        measured={"mean_FeN_A": 1.972, "sigma_deg_approx": 41.0},
    )
    # 5% relative default tolerance is generous for these values.
    assert out["mean_FeN_A"][2] is True
    assert out["sigma_deg_approx"][2] is True


def test_expected_vs_measured_out_of_tolerance():
    out = expected_vs_measured(
        expected={"mean_FeN_A": 1.97},
        measured={"mean_FeN_A": 2.20},
    )
    assert out["mean_FeN_A"][2] is False


def test_expected_vs_measured_missing_measurement():
    out = expected_vs_measured(
        expected={"sigma_deg_approx": 40},
        measured={},
    )
    val = out["sigma_deg_approx"]
    assert val[1] is None
    assert val[2] is False


# ---------------------------------------------------------------------
# 5) Integration with the project's actual library
# ---------------------------------------------------------------------

def test_repo_library_loads_without_errors():
    """The shipped data/reference/library.json must load cleanly —
    a manifest with a typo would break Mode 3 for every user.

    Two kinds of entries are valid:
      - Real CIFs with bundled files (cif_exists = True).
      - Literature templates with no bundled CIF — these are
        comparison targets with published expected metrics only,
        recognisable by the `(literature_template).cif` filename
        sentinel. cif_exists = False is acceptable for these; the
        UI disables the 'Inspect in Single CIF mode' button.
    """
    project_root = Path(__file__).resolve().parent.parent
    real_library = project_root / "data" / "reference"
    entries = load_library(real_library)
    assert len(entries) >= 1
    for e in entries:
        assert e.display_name
        assert e.id
        is_literature_template = e.cif_filename == "(literature_template).cif"
        if is_literature_template:
            # Literature templates should carry expected_metrics so
            # the compare view has something to display.
            assert e.expected_metrics, (
                f"Literature template {e.id} has no expected_metrics; "
                "it would render as a card with no comparable values."
            )
        else:
            # Real CIF entries: the file must be on disk.
            if not e.cif_exists:
                pytest.fail(
                    f"Seed library entry {e.id} points at missing CIF "
                    f"{e.cif_filename}; check data/reference/."
                )


# ---------------------------------------------------------------------
# 6) UploadedFeCentre — the Mode-3 ad-hoc-upload path
# ---------------------------------------------------------------------

def test_uploaded_fe_centre_dataclass_is_pickleable():
    """@st.cache_data pickles its outputs; the upload dataclass must
    serialise cleanly via the asdict() round-trip the cache uses."""
    from dataclasses import asdict
    from src.modes.reference_library import UploadedFeCentre

    u = UploadedFeCentre(
        id="upload__t.cif__Fe1",
        cif_filename="t.cif",
        fe_label="Fe1",
        display_name="t.cif · Fe1",
        description="demo",
        teaching_notes=["note one", "note two"],
        measured_metrics={"mean_FeN_A": 1.97, "sigma_deg": 12.0},
        n_FeN=6,
        temperature_K=100.0,
        refcode=None,
        coordination="octahedral",
    )
    d = asdict(u)
    # Round-trip through dict (what @st.cache_data does internally
    # when pickling/unpickling) should restore every field.
    u2 = UploadedFeCentre(**d)
    assert u2.measured_metrics == u.measured_metrics
    assert u2.is_uploaded is True


def test_uploaded_fe_centre_is_renderable_by_card():
    """The unified card renderer accepts both LibraryEntry and
    UploadedFeCentre via duck typing — verify the attribute surface."""
    from src.modes.reference_library import UploadedFeCentre
    u = UploadedFeCentre(
        id="x", cif_filename="x.cif", fe_label="Fe1",
        display_name="x", description="d",
        teaching_notes=[], measured_metrics={},
        n_FeN=6, temperature_K=None, refcode=None,
        coordination=None,
    )
    # Attributes the card renderer reads:
    for attr in (
        "display_name", "compound_class", "description",
        "teaching_notes", "expected_metrics", "measured_metrics",
        "is_uploaded", "license",
    ):
        assert hasattr(u, attr), f"UploadedFeCentre missing .{attr}"


def test_autonotes_describes_short_bond_as_LS_like():
    """Mean Fe–N at the LS end of the typical range should generate a
    teaching note mentioning 'low-spin'."""
    from src.modes.reference_library import _autonotes_for_uploaded
    notes = _autonotes_for_uploaded(
        mean_d=1.97, sigma_deg=40.0, theta_deg=20.0,
        n_FeN=6, coordination="octahedral",
    )
    joined = "\n".join(notes)
    assert "low-spin" in joined.lower() or "ls" in joined.lower()


def test_autonotes_describes_long_bond_as_HS_like():
    """Mean Fe–N at the HS end of the range should mention 'high-spin'."""
    from src.modes.reference_library import _autonotes_for_uploaded
    notes = _autonotes_for_uploaded(
        mean_d=2.18, sigma_deg=95.0, theta_deg=120.0,
        n_FeN=6, coordination="octahedral",
    )
    joined = "\n".join(notes)
    assert "high-spin" in joined.lower() or "hs" in joined.lower()


def test_autonotes_flags_macrocyclic_at_low_sigma():
    """Σ below 25° should trigger the 'macrocyclic/rigid framework' note."""
    from src.modes.reference_library import _autonotes_for_uploaded
    notes = _autonotes_for_uploaded(
        mean_d=2.00, sigma_deg=12.0, theta_deg=18.0,
        n_FeN=6, coordination="octahedral",
    )
    joined = "\n".join(notes)
    assert ("macrocyclic" in joined.lower()
            or "porphyrin" in joined.lower()
            or "rigid" in joined.lower())
