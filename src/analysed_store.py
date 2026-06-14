"""
Session-state store for CIFs the user has analysed in any mode.

When Mode 1 parses a CIF and runs analyse(), it records the measured
metrics here so Mode 3 can render the same structure as a comparison
card without requiring a re-upload. Mode 3's own upload widget writes
to the same store, so 'I analysed it in Mode 1 — now show it in
Mode 3' and 'I uploaded it in Mode 3' are the same code path.

Separated from src.annotations on purpose: that module holds *user-set*
tags (oxidation, spin) keyed by filename; this module holds the
*structural* result of analyse() — measured metrics, Fe centres,
provenance. Mode 3 joins both at render time.

The store lives in `st.session_state["analysed_cifs"]` so it survives
mode switches but resets when the Streamlit session ends.

Schema per filename:

    {
        "content_hash":   "sha256 of the original cif bytes",
        "cif_bytes":      b"data_test\\n...",   # so Mode 3 can re-analyse
                                               # without forcing a re-upload
        "temperature_K":  100.0,               # from CIF header (or None)
        "refcode":        "CCDC 254204",       # from Provenance (or None)
        "formula":        "C56 H62.67 Fe N8 O1.33",
        "doi":            "10.1021/...",       # paper DOI (or None)
        "fe_centres": {
            "Fe1": {
                "n_FeN":         6,
                "mean_FeN_A":    1.973,
                "min_FeN_A":     1.970,
                "max_FeN_A":     1.979,
                "sigma_deg":     48.78,
                "theta_deg":     107.63,
                "zeta_A":        0.023,
                "coordination":  "octahedral",
                "distances_A":   [1.97, 1.98, ...],
            },
            ...
        },
    }
"""

from __future__ import annotations
import hashlib
from typing import Any

import streamlit as st

_KEY = "analysed_cifs"


# ----------------------------------------------------------------------
# Low-level store helpers
# ----------------------------------------------------------------------

def _store() -> dict[str, dict[str, Any]]:
    return st.session_state.setdefault(_KEY, {})


def record(filename: str, *, content_hash: str, **fields: Any) -> None:
    """Save or update the entry for `filename`. Last write wins."""
    entry = _store().setdefault(filename, {})
    entry["content_hash"] = content_hash
    entry.update(fields)


def get(filename: str) -> dict[str, Any] | None:
    """Return the stored entry for `filename`, or None."""
    return _store().get(filename)


def all_analysed() -> dict[str, dict[str, Any]]:
    """Snapshot of every analysed CIF in the session."""
    return dict(_store())


def remove(filename: str) -> None:
    """Drop one entry from the store."""
    _store().pop(filename, None)


def clear() -> None:
    """Wipe the entire store."""
    _store().clear()


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------
# Convenience: record from an analyse() result
# ----------------------------------------------------------------------

def record_from_analysis(filename: str,
                         cif_bytes: bytes,
                         bundle,
                         result) -> None:
    """Extract measured metrics from an AnalysisResult and persist.

    Called from Mode 1 (and Mode 3's upload widget) right after
    analyse() returns. No-op when the result has no N-coordinated Fe
    centres — there's nothing useful to compare downstream in that
    case.
    """
    # Local imports keep this module ccdc-free at import time.
    from src.data_schema import (
        COL_DETECTION_METHOD, COL_DISTANCE, COL_FE_LABEL, DetectionMethod,
    )

    fe_centres: dict[str, dict[str, Any]] = {}

    for fe_label, oct_d in (getattr(result, "geometry", {}) or {}).items():
        # Pull only formal bonds for this Fe centre. Geometric /
        # symmetry candidates aren't kept here because they corrupt the
        # mean / sigma for downstream comparison against templates.
        formal = result.bonds[
            (result.bonds[COL_FE_LABEL] == fe_label)
            & (result.bonds[COL_DETECTION_METHOD]
               == DetectionMethod.FORMAL_BOND.value)
        ]
        if formal.empty:
            continue
        distances = [float(d) for d in formal[COL_DISTANCE].tolist()]
        mean_d = sum(distances) / len(distances)

        cg = (result.coord_geom.get(fe_label)
              if getattr(result, "coord_geom", None) else None)

        fe_centres[fe_label] = {
            "n_FeN":         len(distances),
            "mean_FeN_A":    round(mean_d, 4),
            "min_FeN_A":     round(min(distances), 4),
            "max_FeN_A":     round(max(distances), 4),
            "sigma_deg":     round(oct_d.sigma, 2)
                              if oct_d.sigma is not None else None,
            "theta_deg":     round(oct_d.theta, 2)
                              if oct_d.theta is not None else None,
            "zeta_A":        round(oct_d.zeta, 4)
                              if oct_d.zeta is not None else None,
            "delta_x1e4":    round(oct_d.delta * 1e4, 2)
                              if oct_d.delta is not None else None,
            "coordination":  cg.classification if cg else None,
            "distances_A":   distances,
        }

    if not fe_centres:
        # No N-coordinated Fe in this CIF — nothing to compare. We
        # still record the bare provenance so Mode 3 can show 'this
        # was analysed but had no Fe-N' rather than dropping silently.
        prov = getattr(bundle, "provenance", None)
        record(
            filename,
            content_hash=hash_bytes(cif_bytes),
            cif_bytes=cif_bytes,
            temperature_K=getattr(bundle, "temperature_K", None),
            refcode=(prov.ccdc_refcode if prov else None),
            formula=(prov.chemical_formula if prov else None),
            doi=(prov.doi if prov else None),
            fe_centres={},
        )
        return

    prov = getattr(bundle, "provenance", None)
    record(
        filename,
        content_hash=hash_bytes(cif_bytes),
        cif_bytes=cif_bytes,
        temperature_K=getattr(bundle, "temperature_K", None),
        refcode=(prov.ccdc_refcode if prov else None),
        formula=(prov.chemical_formula if prov else None),
        doi=(prov.doi if prov else None),
        fe_centres=fe_centres,
    )
