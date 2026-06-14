"""
Reference library — manifest schema, loader, validation.

A curated library is a small set of CIFs sitting under
`data/reference/` together with a `library.json` describing each one.
The dashboard renders the library as a gallery of teaching cards;
new entries are added by dropping a CIF in the folder and registering
it in the JSON. No code change required.

The split between this module and `src/modes/reference_library.py`:
this file knows nothing about Streamlit. It just defines the data
contract and the I/O. The Streamlit module reads the loaded entries
and renders widgets.

Manifest schema (JSON):

    {
      "schema_version": "1",
      "entries": [
        {
          "id":               "fe_phen3_LS_at_100K",        # stable identifier
          "cif_filename":     "Fe_phen3.cif",                # relative to library folder
          "display_name":     "[Fe(phen)₃](BF₄)₂",
          "compound_class":   "polypyridyl iron(II)",
          "oxidation_state":  "Fe(II)",                      # or null
          "spin_state":       "LS",                          # or null
          "temperature_K":    100,                           # or null
          "cod_id":           "1234567",                     # or null
          "doi":              "10.xxxx/yyyy",                # or null
          "license":          "Crystallography Open Database, CC0",
          "description":      "Classic Fe(II) LS structure…",
          "teaching_notes":   ["Mean Fe-N ≈ 1.97 Å is textbook LS Fe(II)…",
                                "Σ ≈ 30-45° — typical of polypyridyl…"],
          "expected_metrics": {                              # optional sanity values
            "mean_FeN_A":      1.97,
            "sigma_deg_approx": 40
          }
        }
      ]
    }
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LibraryEntry:
    """One curated reference structure.

    `cif_path` is computed at load time from `cif_filename` + the
    library folder, so consumers don't have to handle the join.
    `cif_exists` is True iff the CIF file is currently on disk —
    cards for missing CIFs are still rendered (with the metadata
    and teaching notes) but the "open in Mode 1" affordance is
    disabled. That way the manifest can list aspirational entries
    that you've cited but not yet bundled, without crashing the UI.
    """
    id: str
    cif_filename: str
    cif_path: Path
    cif_exists: bool
    display_name: str
    compound_class: str | None = None
    oxidation_state: str | None = None
    spin_state: str | None = None
    temperature_K: float | None = None
    cod_id: str | None = None
    doi: str | None = None
    license: str | None = None
    description: str = ""
    teaching_notes: list[str] = field(default_factory=list)
    expected_metrics: dict[str, float] = field(default_factory=dict)


class LibraryError(ValueError):
    """Raised when the manifest is malformed."""


def _resolve_entry(raw: dict[str, Any], library_dir: Path) -> LibraryEntry:
    """Build a LibraryEntry from one manifest dict, computing the
    on-disk path and existence check."""
    required = ("id", "cif_filename", "display_name")
    missing = [k for k in required if k not in raw or not raw[k]]
    if missing:
        raise LibraryError(
            f"Manifest entry missing required field(s) "
            f"{missing}: {raw!r}"
        )
    cif_filename = raw["cif_filename"]
    cif_path = library_dir / cif_filename
    return LibraryEntry(
        id=raw["id"],
        cif_filename=cif_filename,
        cif_path=cif_path,
        cif_exists=cif_path.exists(),
        display_name=raw["display_name"],
        compound_class=raw.get("compound_class"),
        oxidation_state=raw.get("oxidation_state"),
        spin_state=raw.get("spin_state"),
        temperature_K=(
            float(raw["temperature_K"]) if raw.get("temperature_K") is not None
            else None
        ),
        cod_id=raw.get("cod_id"),
        doi=raw.get("doi"),
        license=raw.get("license"),
        description=raw.get("description", ""),
        teaching_notes=list(raw.get("teaching_notes", [])),
        expected_metrics={
            str(k): float(v) for k, v in (raw.get("expected_metrics", {}) or {}).items()
            if v is not None
        },
    )


def load_library(library_dir: Path | str) -> list[LibraryEntry]:
    """Read `library.json` from `library_dir` and return its entries.

    Returns an empty list when the folder doesn't exist or has no
    manifest — the UI then shows a "no library configured" hint
    rather than crashing.

    Raises
    ------
    LibraryError
        If the manifest is present but malformed (bad JSON, missing
        required fields, wrong schema version).
    """
    library_dir = Path(library_dir)
    manifest = library_dir / "library.json"
    if not manifest.exists():
        return []

    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LibraryError(
            f"library.json is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise LibraryError("library.json must be a JSON object at the top level.")

    if "entries" not in data:
        raise LibraryError("library.json must contain an 'entries' array.")
    if not isinstance(data["entries"], list):
        raise LibraryError("'entries' in library.json must be an array.")

    return [_resolve_entry(raw, library_dir) for raw in data["entries"]]


# ----------------------------------------------------------------------
# Helpers used by the UI layer to summarise expected vs measured metrics.
# ----------------------------------------------------------------------

def expected_vs_measured(
    expected: dict[str, float],
    measured: dict[str, float],
    tolerance: dict[str, float] | None = None,
) -> dict[str, tuple[float, float | None, bool]]:
    """Compare expected and measured values key-by-key.

    Returns {key: (expected, measured, within_tolerance)} for every
    key in `expected`. `measured` may have None for missing values.
    Default tolerance: ±5% relative.
    """
    if tolerance is None:
        tolerance = {}
    out: dict[str, tuple[float, float | None, bool]] = {}
    for key, exp_val in expected.items():
        meas_val = measured.get(key)
        if meas_val is None:
            out[key] = (exp_val, None, False)
            continue
        tol = tolerance.get(key, abs(exp_val) * 0.05)
        out[key] = (exp_val, meas_val, abs(meas_val - exp_val) <= tol)
    return out
