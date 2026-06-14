"""
Fe–N detection and distance extraction.

This module turns a parsed CIF (a CifBundle from cif_reader) into a tidy
pandas DataFrame of Fe–N distances, with each row labelled by *how* the
distance was found.

Three detection methods are supported (data_schema.DetectionMethod):

  FORMAL_BOND
      The Fe–N pair is in `mol.bonds` after assign_bond_types(). This is
      what the CCDC chemistry algorithm classifies as a real bond.

  GEOMETRIC_CANDIDATE
      The Fe–N pair is within `cutoff_A` Å in 3D space, but is NOT in
      `mol.bonds`. Useful for catching long coordinative bonds that the
      bond-perception algorithm missed.

  SYMMETRY_CONTACT
      The Fe and N are not in the same asymmetric-unit molecule, but a
      symmetry-equivalent N exists within `cutoff_A` of the Fe. Picked
      up via crystal.contacts() if available.

Beginner notes:
  - `pd.DataFrame` is a 2-D table — like an Excel sheet in memory.
  - We build the table by collecting one dict per row, then handing the
    list of dicts to pandas. This is the canonical pandas idiom.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .cif_reader import CifBundle
from .data_schema import (
    BOND_TABLE_COLUMNS,
    COL_CIF_FILE, COL_DETECTION_METHOD, COL_DISTANCE, COL_DISTANCE_ESD,
    COL_FE_LABEL, COL_N_LABEL, COL_STRUCTURE_ID,
    COL_SYMMETRY_RELATED, COL_TEMPERATURE, COL_WARNING,
    DetectionMethod,
)
from .geometry import (
    CoordinationGeometry, OctaDistortion,
    coordination_geometry, octahedral_distortion,
)


# -----------------------------------------------------------------------
# Atom helpers
# -----------------------------------------------------------------------

def find_fe_atoms(molecule: Any) -> list[Any]:
    """Return all Fe atoms in the molecule."""
    return [a for a in molecule.atoms if a.atomic_symbol == "Fe"]


def find_n_atoms(molecule: Any) -> list[Any]:
    """Return all N atoms in the molecule."""
    return [a for a in molecule.atoms if a.atomic_symbol == "N"]


def _euclidean_distance(a: Any, b: Any) -> float:
    """Cartesian distance (Å) between two atoms with .coordinates."""
    p, q = a.coordinates, b.coordinates
    return ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5


# -----------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Output of `analyse()`.

    Attributes
    ----------
    bonds : pd.DataFrame
        One row per Fe–N distance. Columns are listed in
        data_schema.BOND_TABLE_COLUMNS.
    warnings : list[str]
        Human-readable notes the UI should display.
    n_fe : int
        Total Fe atoms found in the asymmetric unit (including
        counterion / non-N-coordinated centres that we exclude
        from the analysis).
    n_coord_fe : int
        Number of Fe atoms with at least one formal Fe–N bond.
        These are the only centres that contribute rows to `bonds`.
    n_n : int
        Number of N atoms found in the asymmetric unit.
    excluded_fe_labels : list[str]
        Labels of Fe atoms that were skipped because they had no
        formal Fe–N bonds (typical counterion case).
    """
    bonds: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    n_fe: int = 0
    n_coord_fe: int = 0
    n_n: int = 0
    excluded_fe_labels: list[str] = field(default_factory=list)
    # Per-Fe distortion parameters (ζ, Δ, Σ, Θ). Keyed by Fe atom label.
    # Empty when no N-coordinated Fe centres were found. Computed using
    # formal-bond Fe–N neighbours only — geometric/symmetry candidates
    # would inject non-coordinating N atoms and corrupt the metrics.
    geometry: dict[str, OctaDistortion] = field(default_factory=dict)
    # Per-Fe coordination-geometry classification (octahedral / TBP /
    # square pyramidal / tetrahedral / square planar / distorted /
    # unusual). Same keys as `geometry`; same formal-bond-only rule.
    coord_geom: dict[str, CoordinationGeometry] = field(default_factory=dict)


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def _atom_passes_occupancy(atom: Any, min_occupancy: float) -> bool:
    """True when `atom`'s occupancy is at or above the threshold.

    Atoms with occupancy == None are treated as fully occupied —
    'None' generally means 'not refined as partial', which is the
    safer default than dropping the atom.
    """
    occ = getattr(atom, "occupancy", 1.0)
    if occ is None:
        return True
    try:
        return float(occ) >= min_occupancy
    except (TypeError, ValueError):
        return True


def analyse(
    bundle: CifBundle,
    *,
    cif_filename: str,
    cutoff_A: float = 2.7,
    include_geometric: bool = False,
    include_symmetry: bool = False,
    min_occupancy: float = 0.0,
) -> AnalysisResult:
    """Find every Fe–N distance in `bundle`.

    Parameters
    ----------
    bundle : CifBundle
        The parsed CIF.
    cif_filename : str
        The original filename (so the CSV export shows where the data
        came from). We pass this in rather than reading it off the
        bundle, because Streamlit names uploads opaquely.
    cutoff_A : float
        Distances greater than this are ignored. 2.7 Å covers all
        chemically reasonable Fe–N coordinative bonds plus a margin.
    include_geometric : bool
        If True, include Fe–N pairs within cutoff that aren't in the
        formal bond list.
    include_symmetry : bool
        If True, attempt symmetry expansion via crystal.contacts() to
        catch Fe–N pairs that span asymmetric units.

    Returns
    -------
    AnalysisResult
    """
    warnings = list(bundle.warnings)
    # Apply the occupancy filter at the very top — every downstream
    # step (formal bonds, geometric candidates, geometry / BVS metrics)
    # then operates on the same filtered atom set. min_occupancy = 0.0
    # is the inclusive default; the UI flips it to 0.5 when the user
    # selects 'Use major component only'.
    fe_atoms = [a for a in find_fe_atoms(bundle.molecule)
                if _atom_passes_occupancy(a, min_occupancy)]
    n_atoms  = [a for a in find_n_atoms(bundle.molecule)
                if _atom_passes_occupancy(a, min_occupancy)]
    if min_occupancy > 0.0 and bundle.disorder_atoms:
        dropped = [d.label for d in bundle.disorder_atoms
                   if d.occupancy < min_occupancy]
        if dropped:
            warnings.append(
                f"Disorder filter (occupancy ≥ {min_occupancy:.2f}) "
                f"excluded {len(dropped)} partial-occupancy atom(s): "
                f"{', '.join(dropped)}."
            )

    # Early exits — no data to analyse.
    if not fe_atoms:
        warnings.append("No Fe atoms found in this structure.")
        return AnalysisResult(
            bonds=_empty_bonds_df(),
            warnings=warnings,
            n_fe=0, n_coord_fe=0, n_n=len(n_atoms),
        )
    if not n_atoms:
        warnings.append(
            f"{len(fe_atoms)} Fe atom(s) present but no N atoms — no Fe–N data."
        )
        return AnalysisResult(
            bonds=_empty_bonds_df(),
            warnings=warnings,
            n_fe=len(fe_atoms), n_coord_fe=0, n_n=0,
        )

    if len(fe_atoms) > 1:
        warnings.append(f"Multiple Fe centres detected: {len(fe_atoms)}.")

    # --- Step 1: collect FORMAL bonds from mol.bonds ---------------------
    rows: list[dict] = []
    # `formal_pairs` is a set of (fe_label, n_label, rounded_distance)
    # tuples. Using labels (instead of Python id()) makes the dedup
    # robust across symmetry-generated atom copies, which CCDC creates
    # as fresh objects with the same label.
    formal_pairs: set[tuple[str, str, float]] = set()
    # `n_coordinated_fe_labels` records which Fe atoms have at least
    # one formal Fe–N bond. Any Fe NOT in this set is treated as a
    # counterion / non-N-coordinated centre and is skipped during the
    # geometric and symmetry steps — this prevents structures with an
    # iron-containing counterion (e.g. FeOCl3⁻, FeCl4⁻) from polluting
    # the analysis with spurious long-range distances.
    n_coordinated_fe_labels: set[str] = set()

    def _signature(fe_atom: Any, n_atom: Any, distance: float
                   ) -> tuple[str, str, float]:
        return (fe_atom.label, n_atom.label, round(float(distance), 2))

    try:
        bundle.molecule.assign_bond_types(which="all")
    except Exception:
        # Some odd CIFs reject assign_bond_types; we proceed anyway.
        pass

    for bond in bundle.molecule.bonds:
        a1, a2 = bond.atoms
        if {a1.atomic_symbol, a2.atomic_symbol} != {"Fe", "N"}:
            continue
        # Skip bonds whose atoms have been filtered out by occupancy.
        if not (_atom_passes_occupancy(a1, min_occupancy)
                and _atom_passes_occupancy(a2, min_occupancy)):
            continue
        fe = a1 if a1.atomic_symbol == "Fe" else a2
        n_ = a2 if fe is a1 else a1
        # Look up the esd from the _geom_bond_distance loop, if the CIF
        # provided one for this pair. Atom labels match the asymmetric-
        # unit labels ccdc exposes here.
        _, esd_val = bundle.bond_esds.get(
            frozenset([fe.label, n_.label]), (None, None),
        )
        rows.append(_make_row(
            bundle, cif_filename, fe, n_,
            distance=bond.length,
            method=DetectionMethod.FORMAL_BOND,
            symmetry_related=False,
            distance_esd=esd_val,
        ))
        formal_pairs.add(_signature(fe, n_, bond.length))
        n_coordinated_fe_labels.add(fe.label)

    # Identify counterion-style Fe atoms (no formal Fe–N bonds at all).
    counterion_fe = [fe for fe in fe_atoms
                     if fe.label not in n_coordinated_fe_labels]
    if counterion_fe:
        labels = ", ".join(fe.label for fe in counterion_fe)
        warnings.append(
            f"Excluded Fe atom(s) with no formal Fe–N bonds (likely "
            f"counterion or non-N-coordinated centre): {labels}. These "
            f"contribute zero rows to the analysis."
        )

    # Subset of Fe atoms that ARE genuinely N-coordinated. Used by the
    # geometric and symmetry steps below.
    coord_fe = [fe for fe in fe_atoms
                if fe.label in n_coordinated_fe_labels]

    # --- Step 2: geometric candidates (within cutoff, not formal) --------
    if include_geometric and coord_fe:
        for fe in coord_fe:
            for n_ in n_atoms:
                d = _euclidean_distance(fe, n_)
                if _signature(fe, n_, d) in formal_pairs:
                    continue
                if d <= cutoff_A:
                    rows.append(_make_row(
                        bundle, cif_filename, fe, n_,
                        distance=d,
                        method=DetectionMethod.GEOMETRIC_CANDIDATE,
                        symmetry_related=False,
                        warning="long Fe–N — verify chemistry"
                                if d > 2.4 else "",
                    ))

    # --- Step 3: symmetry-generated contacts -----------------------------
    # We drop `path_length_range=(1, 1)` here because that filter
    # interacts badly with symmetry expansion: it returned pairs whose
    # `c.length` was the original-molecule bond length even when the
    # symmetry-equivalent partner was 5+ Å away in space. Without it
    # contacts() returns true through-space pairs only, which we then
    # post-filter strictly on `c.length <= cutoff_A` to be safe.
    if include_symmetry and coord_fe:
        coord_fe_labels = {fe.label for fe in coord_fe}
        try:
            contacts = bundle.crystal.contacts(
                distance_range=(0.1, cutoff_A),
                intermolecular="Inter",
            )
            for c in contacts:
                if c.length > cutoff_A:        # belt-and-braces
                    continue
                a1, a2 = c.atoms
                if {a1.atomic_symbol, a2.atomic_symbol} != {"Fe", "N"}:
                    continue
                fe = a1 if a1.atomic_symbol == "Fe" else a2
                n_ = a2 if fe is a1 else a1
                # Skip contacts on counterion Fe atoms.
                if fe.label not in coord_fe_labels:
                    continue
                if _signature(fe, n_, c.length) in formal_pairs:
                    continue
                rows.append(_make_row(
                    bundle, cif_filename, fe, n_,
                    distance=c.length,
                    method=DetectionMethod.SYMMETRY_CONTACT,
                    symmetry_related=True,
                ))
        except Exception as exc:                       # noqa: BLE001
            warnings.append(
                f"Symmetry expansion not attempted ({exc!r}). "
                "May have missed inter-asymmetric-unit Fe–N contacts."
            )

    # --- Step 4: per-Fe distortion parameters (ζ, Δ, Σ, Θ)
    #             and coordination-geometry classification ---------------
    # We use ONLY formal-bond N partners here — geometric/symmetry
    # candidates can pull in non-coordinating N atoms and corrupt the
    # metrics. The geometry module is happy with any ligand count: it
    # computes ζ/Δ for any n ≥ 1 and skips Σ/Θ when n ≠ 6.
    geometry: dict[str, OctaDistortion] = {}
    coord_geom: dict[str, CoordinationGeometry] = {}
    if coord_fe:
        # Build a map: fe.label → list of bonded N atom objects.
        # The bond iteration above didn't keep the N atom references,
        # so re-walk mol.bonds. (Tens of bonds, no perf concern.)
        bonded_ns_by_fe: dict[str, list[Any]] = {fe.label: [] for fe in coord_fe}
        for bond in bundle.molecule.bonds:
            a1, a2 = bond.atoms
            if {a1.atomic_symbol, a2.atomic_symbol} != {"Fe", "N"}:
                continue
            fe = a1 if a1.atomic_symbol == "Fe" else a2
            n_ = a2 if fe is a1 else a1
            if fe.label in bonded_ns_by_fe:
                bonded_ns_by_fe[fe.label].append(n_)

        for fe in coord_fe:
            n_partners = bonded_ns_by_fe.get(fe.label, [])
            partner_coords = [n.coordinates for n in n_partners]
            try:
                geometry[fe.label] = octahedral_distortion(
                    fe.coordinates, partner_coords,
                )
            except Exception as exc:                          # noqa: BLE001
                warnings.append(
                    f"Distortion params skipped for {fe.label}: {exc!r}."
                )
            try:
                coord_geom[fe.label] = coordination_geometry(
                    fe.coordinates, partner_coords,
                )
            except Exception as exc:                          # noqa: BLE001
                warnings.append(
                    f"Coordination geometry skipped for {fe.label}: {exc!r}."
                )

    # --- Build the DataFrame --------------------------------------------
    if not rows:
        warnings.append("No Fe–N bonds or contacts found within the cutoff.")
        df = _empty_bonds_df()
    else:
        df = pd.DataFrame(rows, columns=BOND_TABLE_COLUMNS)
        # Sort by Fe centre, then by distance — tidy output.
        df = df.sort_values([COL_FE_LABEL, COL_DISTANCE]).reset_index(drop=True)

    return AnalysisResult(
        bonds=df,
        warnings=warnings,
        n_fe=len(fe_atoms),
        n_coord_fe=len(coord_fe),
        n_n=len(n_atoms),
        excluded_fe_labels=[fe.label for fe in counterion_fe],
        geometry=geometry,
        coord_geom=coord_geom,
    )


# -----------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------

def _empty_bonds_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column set."""
    return pd.DataFrame(columns=BOND_TABLE_COLUMNS)


def _make_row(
    bundle: CifBundle,
    cif_filename: str,
    fe: Any,
    n_: Any,
    *,
    distance: float,
    method: DetectionMethod,
    symmetry_related: bool,
    warning: str = "",
    distance_esd: float | None = None,
) -> dict:
    """Build one row of the bond table."""
    return {
        COL_CIF_FILE:         cif_filename,
        COL_STRUCTURE_ID:     bundle.structure_id,
        COL_TEMPERATURE:      bundle.temperature_K,
        COL_FE_LABEL:         fe.label,
        COL_N_LABEL:          n_.label,
        COL_DISTANCE:         round(float(distance), 4),
        COL_DISTANCE_ESD:     distance_esd,
        COL_DETECTION_METHOD: method.value,
        COL_SYMMETRY_RELATED: bool(symmetry_related),
        COL_WARNING:          warning,
    }
