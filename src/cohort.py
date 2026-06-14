"""
Cohort-table construction — pure helpers used by Mode 2 (Batch).

Lives outside `modes/batch.py` so the table-building logic is
testable without spinning up Streamlit. The Streamlit layer only
deals with widgets, caching, and parallelism; *what* a row is and
*how* a series is assigned both live here.

One **cohort row = one Fe centre in one CIF**. A Z′ = 2 structure
with two crystallographically-distinct Fe atoms contributes two
rows; a counterion-style Fe with no Fe–N bonds contributes none.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

import pandas as pd

from .cif_reader import CifBundle
from .fe_n_analysis import AnalysisResult
from .bvs import compute_bvs
from .data_schema import (
    COL_DETECTION_METHOD, COL_DISTANCE, COL_FE_LABEL, DetectionMethod,
)


# ----------------------------------------------------------------------
# Canonical cohort column names — single source of truth.
# ----------------------------------------------------------------------
COL_FILE     = "filename"
COL_REFCODE  = "refcode"
COL_FORMULA  = "formula"
COL_TEMP     = "T_K"
COL_FE       = "Fe_label"
COL_N_FEN    = "n_FeN"
COL_MEAN_D   = "mean_FeN_A"
COL_ZETA     = "zeta_A"
COL_DELTA    = "delta_x1e4"
COL_SIGMA    = "sigma_deg"
COL_THETA    = "theta_deg"
COL_BVS      = "BVS"
COL_OX       = "oxidation_state"
COL_SPIN     = "spin_state"
COL_SERIES   = "series"
COL_R_FACTOR = "R_factor_pct"

# Display order — left to right in the data_editor and CSV.
COHORT_COLUMNS: list[str] = [
    COL_FILE, COL_REFCODE, COL_FORMULA, COL_TEMP, COL_FE, COL_N_FEN,
    COL_MEAN_D, COL_ZETA, COL_DELTA, COL_SIGMA, COL_THETA, COL_BVS,
    COL_OX, COL_SPIN, COL_SERIES, COL_R_FACTOR,
]

# Columns the user may edit in the cohort table.
COHORT_EDITABLE: tuple[str, ...] = (COL_OX, COL_SPIN, COL_SERIES)


@dataclass
class FeSummary:
    """Pickleable per-Fe-centre summary kept in the @st.cache_data store.

    Keeping the formal-bond distance list on the summary means the
    UI can recompute BVS on the fly when the user edits oxidation /
    spin without re-parsing the CIF.
    """
    filename: str
    fe_label: str
    refcode: str | None
    formula: str | None
    temperature_K: float | None
    n_FeN: int
    distances_A: list[float]      # formal-bond Fe-N distances
    mean_d_A: float
    zeta_A: float | None
    delta: float | None           # dimensionless, NOT scaled
    sigma_deg: float | None
    theta_deg: float | None
    R_factor_gt: float | None
    parse_error: str | None = None    # populated when read_cif failed


def summaries_for_cif(
    filename: str,
    bundle: CifBundle,
    result: AnalysisResult,
) -> list[FeSummary]:
    """Build one FeSummary per N-coordinated Fe centre in this CIF.

    Returns an empty list if no Fe centre has any formal Fe–N bond.
    """
    out: list[FeSummary] = []
    prov = bundle.provenance

    for fe_label, oct_d in result.geometry.items():
        formal = result.bonds[
            (result.bonds[COL_FE_LABEL] == fe_label)
            & (result.bonds[COL_DETECTION_METHOD]
               == DetectionMethod.FORMAL_BOND.value)
        ]
        if formal.empty:
            continue
        distances = [float(d) for d in formal[COL_DISTANCE].tolist()]
        out.append(FeSummary(
            filename=filename,
            fe_label=fe_label,
            refcode=(prov.ccdc_refcode if prov else None),
            formula=(prov.chemical_formula if prov else None),
            temperature_K=bundle.temperature_K,
            n_FeN=len(distances),
            distances_A=distances,
            mean_d_A=sum(distances) / len(distances),
            zeta_A=oct_d.zeta,
            delta=oct_d.delta,
            sigma_deg=oct_d.sigma,
            theta_deg=oct_d.theta,
            R_factor_gt=(prov.R_factor_gt if prov else None),
        ))
    return out


def make_error_summary(filename: str, error_message: str) -> FeSummary:
    """A placeholder summary representing a parse failure.

    Surfaces in the cohort table so the user sees *which* upload broke,
    rather than a silent drop.
    """
    return FeSummary(
        filename=filename, fe_label="—",
        refcode=None, formula=None, temperature_K=None,
        n_FeN=0, distances_A=[], mean_d_A=float("nan"),
        zeta_A=None, delta=None, sigma_deg=None, theta_deg=None,
        R_factor_gt=None, parse_error=error_message,
    )


# ----------------------------------------------------------------------
# Annotation merge — apply per-filename ox/spin from a dict
# ----------------------------------------------------------------------

def _bvs_for(distances: list[float], ox: str, spin: str) -> float | None:
    if ox in ("(unknown)", None, ""):
        return None
    spin_in = spin if spin in ("LS", "HS") else None
    res = compute_bvs(distances, ox, spin_in)
    return res.bvs if res is not None else None


def build_cohort_dataframe(
    summaries: Iterable[FeSummary],
    *,
    per_file_annotations: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Turn FeSummary list + annotations into the displayed DataFrame.

    `per_file_annotations` is a {filename: {"oxidation_state": "Fe(II)",
    "spin_state": "LS"}} map. When absent the row is annotated
    "(unknown)" / "(unknown)" and BVS is left blank.

    Series is auto-assigned by `auto_assign_series()` after the rows
    are built — pass the result through that function before display.
    """
    rows: list[dict] = []
    per_file_annotations = per_file_annotations or {}
    for s in summaries:
        if s.parse_error is not None:
            rows.append({
                COL_FILE:     s.filename,
                COL_REFCODE:  "—",
                COL_FORMULA:  f"PARSE FAILED: {s.parse_error}",
                COL_TEMP:     None,
                COL_FE:       "—",
                COL_N_FEN:    0,
                COL_MEAN_D:   None,
                COL_ZETA:     None,
                COL_DELTA:    None,
                COL_SIGMA:    None,
                COL_THETA:    None,
                COL_BVS:      None,
                COL_OX:       "(unknown)",
                COL_SPIN:     "(unknown)",
                COL_SERIES:   "?",
                COL_R_FACTOR: None,
            })
            continue

        ann = per_file_annotations.get(s.filename, {})
        ox = ann.get("oxidation_state", "(unknown)")
        spin = ann.get("spin_state", "(unknown)")
        bvs = _bvs_for(s.distances_A, ox, spin)

        rows.append({
            COL_FILE:     s.filename,
            COL_REFCODE:  s.refcode or "—",
            COL_FORMULA:  s.formula or "—",
            COL_TEMP:     s.temperature_K,
            COL_FE:       s.fe_label,
            COL_N_FEN:    s.n_FeN,
            COL_MEAN_D:   round(s.mean_d_A, 4),
            COL_ZETA:     round(s.zeta_A, 4)        if s.zeta_A     is not None else None,
            COL_DELTA:    round(s.delta * 1e4, 2)   if s.delta      is not None else None,
            COL_SIGMA:    round(s.sigma_deg, 2)     if s.sigma_deg  is not None else None,
            COL_THETA:    round(s.theta_deg, 2)     if s.theta_deg  is not None else None,
            COL_BVS:      round(bvs, 2)             if bvs          is not None else None,
            COL_OX:       ox,
            COL_SPIN:     spin,
            COL_SERIES:   "",
            COL_R_FACTOR: round(s.R_factor_gt * 100, 2)
                          if s.R_factor_gt is not None else None,
        })
    df = pd.DataFrame(rows, columns=COHORT_COLUMNS)
    return df


# ----------------------------------------------------------------------
# Auto-grouping by chemical formula → series letters A, B, C, ...
# ----------------------------------------------------------------------

def auto_assign_series(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each unique `_chemical_formula_sum` value a series letter.

    Returns a new DataFrame (doesn't mutate the input).
    Rows whose formula is missing / parse-failed are tagged "?".

    Letter order: A, B, ..., Z, then AA, AB, AC, ... so the scheme
    survives cohorts larger than 26 unique formulas.
    """
    df = df.copy()
    formulas_seen: dict[str, str] = {}
    counter = 0

    def _letter(idx: int) -> str:
        # Spreadsheet-style: 0→A, 25→Z, 26→AA, …
        out = ""
        idx += 1   # shift to 1-based so the math works
        while idx > 0:
            idx, rem = divmod(idx - 1, 26)
            out = chr(ord("A") + rem) + out
        return out

    new_series: list[str] = []
    for _, row in df.iterrows():
        formula = (row.get(COL_FORMULA) or "").strip()
        if not formula or formula == "—" or formula.startswith("PARSE FAILED"):
            new_series.append("?")
            continue
        if formula not in formulas_seen:
            formulas_seen[formula] = _letter(counter)
            counter += 1
        new_series.append(formulas_seen[formula])
    df[COL_SERIES] = new_series
    return df
