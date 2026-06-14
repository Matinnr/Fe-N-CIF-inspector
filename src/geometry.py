"""
Octahedral distortion parameters for an MN6 (or, more generally, M-L6) centre.

All four canonical SCO descriptors are implemented as pure functions of
Cartesian coordinates — no dependency on ccdc or any other parser. This
keeps the module testable with hand-rolled point clouds.

References
----------
Halcrow, M. A., Chem. Soc. Rev. 2011, 40, 4119–4142.    (review)
McCusker et al., Inorg. Chem. 1996, 35, 2100.            (Σ)
Marchivie et al., Acta Cryst. B 2005, 61, 25.             (Θ definition)
Ketkaew et al., Dalton Trans. 2021, 50, 1086.             (Θ algorithm,
                                                           OctaDist tool)

Definitions
-----------
ζ (zeta)   =  Σ |d_i − d_mean|                                     (Å)
              i=1..6
Δ (delta)  =  (1/n) Σ ((d_i − d_mean) / d_mean)²                   (dimensionless)
              i=1..n
Σ (sigma)  =  Σ |90° − φ_i|     over the 12 cis N–Fe–N angles      (°)
Θ (theta)  =  Σ |60° − θ_i|     over 4 face-pairs × 6 projected
              angles each = 24 angles total                         (°)

For a perfect octahedron ζ = Δ = Σ = Θ = 0. For a regular trigonal
prism Σ and Θ both become large; for a bond-elongated octahedron only
ζ and Δ are non-zero while Σ and Θ stay at zero.

Beginner notes
--------------
- We use numpy throughout. A "vector" is a (3,) ndarray; a "list of
  vectors" is an (n, 3) ndarray. Operations like `v / np.linalg.norm(v)`
  vectorise across rows when shapes match.
- Functions raise ValueError on input that doesn't have the right shape
  rather than silently returning rubbish.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

import numpy as np


# Tolerance used when comparing dot products against ±1 (to keep arccos
# inside its valid domain after floating-point round-off).
_CLIP = (-1.0, 1.0)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class OctaDistortion:
    """All four distortion parameters for one Fe centre.

    Any field is allowed to be None when the geometry doesn't support
    it (e.g. Σ and Θ are None for coordination number ≠ 6).
    """
    zeta: float | None = None        # Å
    delta: float | None = None       # dimensionless
    sigma: float | None = None       # degrees
    theta: float | None = None       # degrees
    coordination_number: int = 0
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Scalar metrics (bond-length-based)
# ----------------------------------------------------------------------

def zeta(distances: Iterable[float]) -> float:
    """Bond-length distortion ζ = Σ|d_i − d_mean|.

    For a perfect octahedron with equal Fe–N distances, ζ = 0.
    The canonical definition assumes 6 distances; we accept any
    non-empty list because the formula is well-defined for any n.
    """
    d = np.asarray(list(distances), dtype=float)
    if d.size == 0:
        raise ValueError("zeta requires at least one distance")
    return float(np.sum(np.abs(d - d.mean())))


def delta(distances: Iterable[float]) -> float:
    """Dimensionless bond-length variance Δ = (1/n) Σ ((d_i − d_mean)/d_mean)².

    For canonical FeN6 use n = 6. For n ≠ 6 we still divide by n so the
    metric stays bounded.
    """
    d = np.asarray(list(distances), dtype=float)
    if d.size == 0:
        raise ValueError("delta requires at least one distance")
    mean = d.mean()
    if mean == 0.0:
        raise ValueError("delta undefined when d_mean = 0")
    return float(np.mean(((d - mean) / mean) ** 2))


# ----------------------------------------------------------------------
# Angle-based metrics (Σ, Θ)
# ----------------------------------------------------------------------

def _coords_to_array(fe: object, ns: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    """Normalise the two input shapes we accept.

    Accepts either raw (x, y, z) tuples / lists / arrays, OR objects
    that expose `.x`, `.y`, `.z` attributes (ccdc Atom.coordinates).
    """
    def to_xyz(obj) -> np.ndarray:
        if hasattr(obj, "x") and hasattr(obj, "y") and hasattr(obj, "z"):
            return np.array([obj.x, obj.y, obj.z], dtype=float)
        return np.asarray(obj, dtype=float)

    fe_arr = to_xyz(fe)
    n_arr = np.vstack([to_xyz(n) for n in ns])
    if fe_arr.shape != (3,):
        raise ValueError(f"Fe coordinate must be 3-D, got {fe_arr.shape}")
    if n_arr.shape[1] != 3:
        raise ValueError(f"N coordinates must be (n, 3), got {n_arr.shape}")
    return fe_arr, n_arr


def _unit_vectors(fe: np.ndarray, ns: np.ndarray) -> np.ndarray:
    """Return (n, 3) array of unit vectors Fe → Nᵢ."""
    v = ns - fe[None, :]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalise: at least one N coincides with Fe")
    return v / norms


def _trans_pairs(unit_vecs: np.ndarray) -> list[tuple[int, int]]:
    """Greedy pairing of 6 unit vectors into 3 'trans' pairs.

    At each step we pick the still-unpaired vector with the smallest
    (most negative) dot product against the first unpaired vector and
    declare them trans. This is the OctaDist convention and works
    cleanly for any geometry near an octahedron.
    """
    if unit_vecs.shape != (6, 3):
        raise ValueError(
            f"trans pairing requires 6 unit vectors, got {unit_vecs.shape[0]}"
        )
    remaining = list(range(6))
    pairs: list[tuple[int, int]] = []
    while remaining:
        i = remaining[0]
        # Find j minimising dot(uvᵢ, uvⱼ) — most antiparallel.
        best_j, best_dot = None, np.inf
        for j in remaining[1:]:
            d = float(np.dot(unit_vecs[i], unit_vecs[j]))
            if d < best_dot:
                best_dot, best_j = d, j
        if best_j is None:           # pragma: no cover  (only if <2 remain)
            break
        pairs.append((i, best_j))
        remaining.remove(i)
        remaining.remove(best_j)
    return pairs


def sigma(fe: object, ns: Iterable[object]) -> float:
    """Σ = Σ|90° − φ_i| over the 12 cis N–Fe–N angles. Requires 6 ligands."""
    fe_arr, n_arr = _coords_to_array(fe, ns)
    if n_arr.shape[0] != 6:
        raise ValueError(f"Σ requires 6 ligands, got {n_arr.shape[0]}")

    uv = _unit_vectors(fe_arr, n_arr)
    trans = _trans_pairs(uv)
    trans_set = {tuple(sorted(p)) for p in trans}

    total = 0.0
    count = 0
    for i, j in combinations(range(6), 2):
        if tuple(sorted((i, j))) in trans_set:
            continue
        dot = np.clip(float(np.dot(uv[i], uv[j])), *_CLIP)
        ang_deg = float(np.degrees(np.arccos(dot)))
        total += abs(90.0 - ang_deg)
        count += 1
    if count != 12:                                          # pragma: no cover
        raise AssertionError(f"expected 12 cis angles, summed {count}")
    return total


def _octahedral_faces(trans_set: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Return the 8 triangular faces of an octahedron: every 3-subset of
    {0..5} that contains no trans pair."""
    faces = []
    for combo in combinations(range(6), 3):
        if any(tuple(sorted((a, b))) in trans_set
               for a, b in combinations(combo, 2)):
            continue
        faces.append(combo)
    return faces


def _antipodal_face_pairs(faces: list[tuple[int, int, int]]
                          ) -> list[tuple[tuple[int, int, int],
                                           tuple[int, int, int]]]:
    """Pair the 8 faces into 4 antipodal pairs (a face and the face of
    the 3 OTHER ligands)."""
    face_set = {tuple(sorted(f)) for f in faces}
    seen: set[tuple[int, int, int]] = set()
    pairs = []
    all_ix = {0, 1, 2, 3, 4, 5}
    for f in faces:
        fk = tuple(sorted(f))
        if fk in seen:
            continue
        opp = tuple(sorted(all_ix - set(f)))
        if opp not in face_set:                              # pragma: no cover
            continue
        pairs.append((fk, opp))
        seen.add(fk)
        seen.add(opp)
    return pairs


def theta(fe: object, ns: Iterable[object]) -> float:
    """Θ = Σ|60° − θ_i| over 24 projected ligand-pair angles.

    OctaDist (Ketkaew 2021) convention:
      1. Identify the 8 octahedral faces (triangles with no trans pair).
      2. Pair them into 4 antipodal face pairs (each pair contains all
         6 ligands between them).
      3. For each pair, project all 6 ligands onto the plane normal to
         the line through the two face centroids.
      4. Sort the 6 projections by polar angle around that axis.
      5. Compute the 6 *adjacent* angular gaps (each ≈ 60° in a perfect
         octahedron); sum |60° − gap_i| across all 24 gaps.

    Returns 0 for a perfect octahedron and grows monotonically with
    twist towards a trigonal prism.
    """
    fe_arr, n_arr = _coords_to_array(fe, ns)
    if n_arr.shape[0] != 6:
        raise ValueError(f"Θ requires 6 ligands, got {n_arr.shape[0]}")

    uv = _unit_vectors(fe_arr, n_arr)
    trans = _trans_pairs(uv)
    trans_set = {tuple(sorted(p)) for p in trans}
    faces = _octahedral_faces(trans_set)
    if len(faces) != 8:
        raise ValueError(
            f"Expected 8 octahedral faces, got {len(faces)} — geometry "
            f"too distorted to apply the OctaDist Θ algorithm."
        )
    pairs = _antipodal_face_pairs(faces)
    if len(pairs) != 4:                                      # pragma: no cover
        raise ValueError(f"Expected 4 antipodal face pairs, got {len(pairs)}")

    total = 0.0
    for face_a, face_b in pairs:
        # Centroid → centroid axis (passes near Fe in any near-octahedron).
        c_a = n_arr[list(face_a)].mean(axis=0)
        c_b = n_arr[list(face_b)].mean(axis=0)
        axis = c_b - c_a
        axis_len = np.linalg.norm(axis)
        if axis_len == 0:                                    # pragma: no cover
            continue
        axis /= axis_len

        # Project each ligand into the plane normal to `axis`, through Fe.
        rel = n_arr - fe_arr
        proj = rel - np.outer(rel @ axis, axis)
        proj_norms = np.linalg.norm(proj, axis=1)
        if np.any(proj_norms == 0):                          # pragma: no cover
            # A ligand sits exactly on the axis — geometry degenerate.
            continue

        # Build an orthonormal basis (ref, tangent) of the projection plane.
        # ref starts from the first projection; tangent = axis × ref.
        ref = proj[0] / proj_norms[0]
        tangent = np.cross(axis, ref)
        # ref and tangent are unit vectors orthogonal to axis.
        angles = np.degrees(np.arctan2(proj @ tangent, proj @ ref)) % 360.0
        angles_sorted = np.sort(angles)
        gaps = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 360.0]]))
        # `gaps` has 6 entries — one between each pair of consecutive
        # projected ligands going round the axis. Sum |60 − gap|.
        total += float(np.sum(np.abs(60.0 - gaps)))
    return total


# ----------------------------------------------------------------------
# Coordination-geometry classification (τ₅, τ₄, trans angles)
# ----------------------------------------------------------------------

@dataclass
class CoordinationGeometry:
    """Classification of an Fe centre's coordination geometry.

    Always populated:
      coordination_number   number of N partners (or whatever ligands)
      classification         human-readable shape name
      notes                  any caveats

    Conditional fields:
      tau / tau_label        τ₅ for 5-coord, τ₄ for 4-coord (else None)
      trans_angles_deg       three closest-to-180° angles for 6-coord
                             (else None)
    """
    coordination_number: int
    classification: str
    tau: float | None = None
    tau_label: str | None = None
    trans_angles_deg: list[float] | None = None
    notes: list[str] = field(default_factory=list)


def _all_pairwise_angles(fe: np.ndarray, n_arr: np.ndarray) -> list[float]:
    """All L–M–L angles (degrees) for the given Fe + N positions."""
    uv = _unit_vectors(fe, n_arr)
    angles: list[float] = []
    n_lig = uv.shape[0]
    for i, j in combinations(range(n_lig), 2):
        dot = np.clip(float(np.dot(uv[i], uv[j])), *_CLIP)
        angles.append(float(np.degrees(np.arccos(dot))))
    return angles


def tau5(fe: object, ns: Iterable[object]) -> float:
    """Addison parameter τ₅ for a 5-coordinate centre.

        τ₅ = (β − α) / 60

    where β is the largest L–M–L angle and α the second-largest.

    Reference values:
      τ₅ = 0  →  ideal square pyramidal
      τ₅ = 1  →  ideal trigonal bipyramidal

    Reference: Addison, A. W. et al., J. Chem. Soc. Dalton Trans.
    1984, 1349. The parameter is widely used as a scalar measure of
    progression from square-pyramidal to trigonal-bipyramidal in
    5-coordinate complexes.
    """
    fe_arr, n_arr = _coords_to_array(fe, ns)
    if n_arr.shape[0] != 5:
        raise ValueError(f"τ₅ requires 5 ligands, got {n_arr.shape[0]}")
    angles = sorted(_all_pairwise_angles(fe_arr, n_arr), reverse=True)
    beta, alpha = angles[0], angles[1]
    return (beta - alpha) / 60.0


def tau4(fe: object, ns: Iterable[object]) -> float:
    """Yang parameter τ₄ for a 4-coordinate centre.

        τ₄ = (360 − (α + β)) / 141

    where α and β are the two largest L–M–L angles.

    Reference values:
      τ₄ = 0  →  ideal square planar
      τ₄ = 1  →  ideal tetrahedral

    Reference: Yang, L. et al., Dalton Trans. 2007, 955. Note the
    141 in the denominator is the empirical normalisation
    360 − 2 × 109.47°.
    """
    fe_arr, n_arr = _coords_to_array(fe, ns)
    if n_arr.shape[0] != 4:
        raise ValueError(f"τ₄ requires 4 ligands, got {n_arr.shape[0]}")
    angles = sorted(_all_pairwise_angles(fe_arr, n_arr), reverse=True)
    beta, alpha = angles[0], angles[1]
    return (360.0 - (alpha + beta)) / 141.0


def trans_angles(fe: object, ns: Iterable[object]) -> list[float]:
    """For 6-coord: return the three trans L–M–L angles, in degrees.

    Trans pairs are identified the same way as for Σ — by greedily
    pairing each unit vector with its most-antiparallel partner. In a
    perfect octahedron all three trans angles are 180°.
    """
    fe_arr, n_arr = _coords_to_array(fe, ns)
    if n_arr.shape[0] != 6:
        raise ValueError(
            f"trans angles require 6 ligands, got {n_arr.shape[0]}"
        )
    uv = _unit_vectors(fe_arr, n_arr)
    pairs = _trans_pairs(uv)
    out: list[float] = []
    for i, j in pairs:
        dot = np.clip(float(np.dot(uv[i], uv[j])), *_CLIP)
        out.append(float(np.degrees(np.arccos(dot))))
    return out


# Thresholds for naming τ₅ and τ₄ classes. Edges are conservative —
# the "distorted" region between SP and TBP (or square-planar and
# tetrahedral) is wide because real complexes rarely sit exactly at
# the idealised extremes.
_TAU5_SP_MAX = 0.20
_TAU5_TBP_MIN = 0.80
_TAU4_SQP_MAX = 0.10
_TAU4_TET_MIN = 0.85


def coordination_geometry(fe: object,
                          ns: Iterable[object]) -> CoordinationGeometry:
    """Classify the coordination geometry of an Fe centre.

      n = 6 → 'octahedral' + the three trans angles
      n = 5 → τ₅ + name (square pyramidal / TBP / distorted)
      n = 4 → τ₄ + name (square planar / tetrahedral / distorted)
      n ∈ {0..3, 7+} → 'unusual' (no standard τ parameter)
    """
    fe_arr, n_arr = _coords_to_array(fe, ns) if list(ns) else (
        np.zeros(3), np.zeros((0, 3)))
    n = n_arr.shape[0]

    if n == 0:
        return CoordinationGeometry(
            coordination_number=0,
            classification="no ligands",
            notes=["No N partners — coordination geometry undefined."],
        )

    if n == 6:
        try:
            ta = trans_angles(fe_arr, n_arr)
            return CoordinationGeometry(
                coordination_number=6,
                classification="octahedral",
                # Sort descending so users see the most-trans-like
                # angle first, followed by the most distorted.
                trans_angles_deg=sorted(ta, reverse=True),
            )
        except ValueError as exc:                            # pragma: no cover
            return CoordinationGeometry(
                coordination_number=6,
                classification="octahedral (trans-angle calc failed)",
                notes=[f"trans_angles: {exc}"],
            )

    if n == 5:
        t5 = tau5(fe_arr, n_arr)
        if t5 <= _TAU5_SP_MAX:
            name = "square pyramidal"
        elif t5 >= _TAU5_TBP_MIN:
            name = "trigonal bipyramidal"
        else:
            name = "distorted (between square-pyramidal and TBP)"
        return CoordinationGeometry(
            coordination_number=5,
            classification=name,
            tau=t5,
            tau_label="τ₅",
        )

    if n == 4:
        t4 = tau4(fe_arr, n_arr)
        if t4 <= _TAU4_SQP_MAX:
            name = "square planar"
        elif t4 >= _TAU4_TET_MIN:
            name = "tetrahedral"
        else:
            name = "distorted (between square-planar and tetrahedral)"
        return CoordinationGeometry(
            coordination_number=4,
            classification=name,
            tau=t4,
            tau_label="τ₄",
        )

    # Anything else: 1, 2, 3, 7, 8, ...
    return CoordinationGeometry(
        coordination_number=n,
        classification=f"unusual ({n}-coordinate)",
        notes=[f"No standard τ parameter for {n}-coordinate centres."],
    )


# ----------------------------------------------------------------------
# Composite entry point
# ----------------------------------------------------------------------

def octahedral_distortion(fe: object, ns: list[object]) -> OctaDistortion:
    """Compute ζ, Δ, Σ, Θ for an Fe centre.

    Behaviour by coordination number:
      n = 0     all four None; note records "no ligands".
      n = 6     all four computed (Σ/Θ may still be None if the
                trans-pair / face-identification algorithm fails on a
                wildly distorted geometry).
      other n   ζ and Δ computed over the available distances; Σ and Θ
                are None, with a note explaining why.
    """
    notes: list[str] = []
    coord = len(ns)
    if coord == 0:
        return OctaDistortion(notes=["No N ligands for this Fe centre."])

    fe_arr, n_arr = _coords_to_array(fe, ns)
    dists = np.linalg.norm(n_arr - fe_arr[None, :], axis=1)
    z_val = zeta(dists)
    d_val = delta(dists)

    if coord != 6:
        notes.append(
            f"Coordination number = {coord}; Σ and Θ require exactly "
            f"6 N ligands and are skipped."
        )
        return OctaDistortion(
            zeta=z_val, delta=d_val, sigma=None, theta=None,
            coordination_number=coord, notes=notes,
        )

    s_val: float | None
    try:
        s_val = sigma(fe_arr, n_arr)
    except ValueError as e:
        s_val = None
        notes.append(f"Σ skipped: {e}")

    t_val: float | None
    try:
        t_val = theta(fe_arr, n_arr)
    except ValueError as e:
        t_val = None
        notes.append(f"Θ skipped: {e}")

    return OctaDistortion(
        zeta=z_val, delta=d_val, sigma=s_val, theta=t_val,
        coordination_number=coord, notes=notes,
    )
