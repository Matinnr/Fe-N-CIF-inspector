"""
Tests for src/geometry.py — pure-math distortion parameter functions.

These don't touch ccdc; every fixture is a small numpy array
constructed in the test. The perfect-octahedron case is the single
most important regression check — any silently-broken Σ/Θ
implementation will fail it first.
"""

from __future__ import annotations
import math

import numpy as np
import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src.geometry import (
    CoordinationGeometry,
    OctaDistortion,
    coordination_geometry,
    delta,
    octahedral_distortion,
    sigma,
    tau4,
    tau5,
    theta,
    trans_angles,
    zeta,
)


# ----------------------------------------------------------------------
# Reusable point clouds
# ----------------------------------------------------------------------

R = 2.0  # bond length used across fixtures

def _perfect_octahedron(r: float = R) -> tuple[np.ndarray, np.ndarray]:
    """Fe at origin, 6 N atoms at ±r along x, y, z axes."""
    fe = np.array([0.0, 0.0, 0.0])
    ns = np.array([
        [+r,  0,  0], [-r,  0,  0],
        [ 0, +r,  0], [ 0, -r,  0],
        [ 0,  0, +r], [ 0,  0, -r],
    ])
    return fe, ns


def _bond_elongated(r_long: float = 2.2, r_short: float = 2.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """One bond elongated; geometry still cubic so angles unchanged."""
    fe = np.array([0.0, 0.0, 0.0])
    ns = np.array([
        [+r_long, 0, 0], [-r_short, 0, 0],
        [0, +r_short, 0], [0, -r_short, 0],
        [0, 0, +r_short], [0, 0, -r_short],
    ])
    return fe, ns


def _trigonal_prism(r: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Regular trigonal prism: two parallel equilateral triangles
    *aligned* (not staggered) along the z-axis."""
    h = math.sqrt(3) / 2 * r              # vertical half-height for regular prism
    triangle = np.array([
        [r,                 0,             0],
        [-r/2,  r * math.sqrt(3)/2,        0],
        [-r/2, -r * math.sqrt(3)/2,        0],
    ])
    top    = triangle + np.array([0, 0,  h])
    bottom = triangle + np.array([0, 0, -h])
    fe = np.array([0.0, 0.0, 0.0])
    return fe, np.vstack([top, bottom])


# ----------------------------------------------------------------------
# Scalar metrics
# ----------------------------------------------------------------------

def test_zeta_zero_for_perfect_octahedron():
    fe, ns = _perfect_octahedron()
    dists = np.linalg.norm(ns - fe, axis=1)
    assert zeta(dists) == pytest.approx(0.0, abs=1e-12)


def test_delta_zero_for_perfect_octahedron():
    fe, ns = _perfect_octahedron()
    dists = np.linalg.norm(ns - fe, axis=1)
    assert delta(dists) == pytest.approx(0.0, abs=1e-12)


def test_zeta_matches_hand_calculation_for_elongation():
    """One bond at 2.2, five at 2.0 → ζ = 0.333... Å."""
    distances = [2.2, 2.0, 2.0, 2.0, 2.0, 2.0]
    # d_mean = (2.2 + 5·2.0)/6 = 2.03333
    # |2.2-2.0333| + 5·|2.0-2.0333| = 0.16667 + 5·0.03333 = 0.3333
    assert zeta(distances) == pytest.approx(1.0/3.0, abs=1e-6)


def test_delta_positive_for_elongation():
    """Δ should be small but non-zero for the elongated case."""
    distances = [2.2, 2.0, 2.0, 2.0, 2.0, 2.0]
    d_mean = sum(distances) / 6
    expected = sum(((d - d_mean) / d_mean) ** 2 for d in distances) / 6
    assert delta(distances) == pytest.approx(expected, abs=1e-10)


# ----------------------------------------------------------------------
# Σ (sigma)
# ----------------------------------------------------------------------

def test_sigma_zero_for_perfect_octahedron():
    fe, ns = _perfect_octahedron()
    assert sigma(fe, ns) == pytest.approx(0.0, abs=1e-9)


def test_sigma_zero_for_bond_elongation():
    """Stretching one bond doesn't change angles — Σ stays at 0."""
    fe, ns = _bond_elongated()
    assert sigma(fe, ns) == pytest.approx(0.0, abs=1e-9)


def test_sigma_large_for_trigonal_prism():
    """Eclipsed geometry has lots of 60°/120° cis angles."""
    fe, ns = _trigonal_prism()
    val = sigma(fe, ns)
    # For a regular trigonal prism Σ ~ several hundred degrees;
    # we just require it's clearly much greater than 0.
    assert val > 100.0


def test_sigma_rejects_wrong_count():
    fe = np.array([0.0, 0.0, 0.0])
    ns = np.array([[1, 0, 0], [-1, 0, 0]])
    with pytest.raises(ValueError):
        sigma(fe, ns)


# ----------------------------------------------------------------------
# Θ (theta)
# ----------------------------------------------------------------------

def test_theta_zero_for_perfect_octahedron():
    fe, ns = _perfect_octahedron()
    assert theta(fe, ns) == pytest.approx(0.0, abs=1e-9)


def test_theta_small_for_bond_elongation():
    """Pure radial elongation gives a SMALL non-zero Θ in OctaDist's
    convention, because the algorithm projects onto a plane whose normal
    is the face-centroid axis; elongating one bond shifts that
    centroid (it's an average of 3 ligand positions) and tilts the
    projection axis slightly. The cis angles between Fe–N vectors are
    unchanged (Σ stays 0), but Θ picks up the centroid drift.

    The invariant we care about: Θ_elongation ≪ Θ_trigonal_prism.
    The exact value (~25° for this fixture) is reproducible across
    runs and matches OctaDist's published behaviour for similar inputs.
    """
    fe, ns = _bond_elongated()
    val = theta(fe, ns)
    assert val < 50.0, f"Θ {val:.2f}° unexpectedly large for pure elongation"
    # And it must be far below the trigonal-prism case.
    fe_p, ns_p = _trigonal_prism()
    assert val < theta(fe_p, ns_p) / 10.0


def test_theta_maximal_for_trigonal_prism():
    """Trigonal prism is the canonical worst-case for Θ."""
    fe, ns = _trigonal_prism()
    val = theta(fe, ns)
    # OctaDist's published value for a regular trigonal prism is ~1440°.
    # We require Θ_prism is unambiguously large — well above any
    # near-octahedral value.
    assert val > 500.0


# ----------------------------------------------------------------------
# Composite entry point
# ----------------------------------------------------------------------

def test_composite_perfect_octahedron_all_zero():
    fe, ns = _perfect_octahedron()
    r = octahedral_distortion(fe, ns)
    assert isinstance(r, OctaDistortion)
    assert r.coordination_number == 6
    assert r.zeta  == pytest.approx(0.0, abs=1e-9)
    assert r.delta == pytest.approx(0.0, abs=1e-9)
    assert r.sigma == pytest.approx(0.0, abs=1e-9)
    assert r.theta == pytest.approx(0.0, abs=1e-9)
    assert r.notes == []


def test_composite_elongation_only_radial():
    """ζ and Δ pick up the elongation; Σ stays exactly 0 (cis angles
    unchanged); Θ stays small (see test_theta_small_for_bond_elongation
    for why it isn't exactly zero)."""
    fe, ns = _bond_elongated()
    r = octahedral_distortion(fe, ns)
    assert r.zeta  == pytest.approx(1.0/3.0, abs=1e-6)
    assert r.sigma == pytest.approx(0.0, abs=1e-9)
    assert r.theta is not None
    assert r.theta < 50.0


def test_composite_five_coordinate_skips_sigma_theta():
    fe = np.array([0.0, 0.0, 0.0])
    ns = np.array([[2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2]])
    r = octahedral_distortion(fe, ns)
    assert r.coordination_number == 5
    assert r.zeta is not None
    assert r.delta is not None
    assert r.sigma is None
    assert r.theta is None
    assert any("coordination" in note.lower() for note in r.notes)


def test_composite_no_ligands_returns_empty():
    fe = np.array([0.0, 0.0, 0.0])
    r = octahedral_distortion(fe, [])
    assert r.coordination_number == 0
    assert r.zeta is None
    assert r.theta is None
    assert any("no n ligands" in n.lower() for n in r.notes)


# ----------------------------------------------------------------------
# Coordinate-source independence
# ----------------------------------------------------------------------

class _XYZ:
    """Minimal stand-in for ccdc Atom.coordinates: just .x .y .z."""
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z


# ----------------------------------------------------------------------
# Idealised geometry fixtures for τ₅, τ₄, trans-angle tests
# ----------------------------------------------------------------------

def _perfect_tbp(r: float = R) -> tuple[np.ndarray, np.ndarray]:
    """Trigonal bipyramid: 2 axial along ±z, 3 equatorial in xy plane."""
    sqrt3 = math.sqrt(3)
    ns = np.array([
        [ 0,           0,        +r],   # axial top
        [ 0,           0,        -r],   # axial bottom
        [ r,           0,         0],   # eq 1
        [-r / 2,  r * sqrt3 / 2,  0],   # eq 2
        [-r / 2, -r * sqrt3 / 2,  0],   # eq 3
    ])
    return np.zeros(3), ns


def _perfect_sp(r: float = R) -> tuple[np.ndarray, np.ndarray]:
    """Ideal square pyramid: 1 apical (+z), 4 basal in xy plane."""
    ns = np.array([
        [ 0,  0, +r],          # apical
        [+r,  0,  0],          # basal
        [-r,  0,  0],
        [ 0, +r,  0],
        [ 0, -r,  0],
    ])
    return np.zeros(3), ns


def _perfect_tetrahedron(r: float = R) -> tuple[np.ndarray, np.ndarray]:
    """Regular tetrahedron with vertices at (±1, ±1, ±1) directions."""
    raw = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float)
    # Scale so each vertex is exactly r from origin.
    raw *= r / math.sqrt(3)
    return np.zeros(3), raw


def _perfect_square_planar(r: float = R) -> tuple[np.ndarray, np.ndarray]:
    """Square planar: 4 atoms in the xy plane at ±x, ±y."""
    ns = np.array([
        [+r,  0,  0],
        [-r,  0,  0],
        [ 0, +r,  0],
        [ 0, -r,  0],
    ])
    return np.zeros(3), ns


# ----------------------------------------------------------------------
# τ₅ (Addison)
# ----------------------------------------------------------------------

def test_tau5_perfect_tbp_equals_1():
    fe, ns = _perfect_tbp()
    assert tau5(fe, ns) == pytest.approx(1.0, abs=1e-9)


def test_tau5_perfect_sp_equals_0():
    fe, ns = _perfect_sp()
    assert tau5(fe, ns) == pytest.approx(0.0, abs=1e-9)


def test_tau5_rejects_wrong_count():
    fe, ns = _perfect_octahedron()
    with pytest.raises(ValueError):
        tau5(fe, ns)


# ----------------------------------------------------------------------
# τ₄ (Yang)
# ----------------------------------------------------------------------

def test_tau4_perfect_tetrahedron_close_to_1():
    """The denominator 141 in Yang's formula approximates 360 − 2·109.47°.
    For an ideal tetrahedron the exact value is 141.06°/141 ≈ 1.0004."""
    fe, ns = _perfect_tetrahedron()
    assert tau4(fe, ns) == pytest.approx(1.0, abs=5e-3)


def test_tau4_perfect_square_planar_equals_0():
    fe, ns = _perfect_square_planar()
    assert tau4(fe, ns) == pytest.approx(0.0, abs=1e-9)


def test_tau4_rejects_wrong_count():
    fe, ns = _perfect_octahedron()
    with pytest.raises(ValueError):
        tau4(fe, ns)


# ----------------------------------------------------------------------
# Trans angles for octahedral
# ----------------------------------------------------------------------

def test_trans_angles_perfect_octahedron_all_180():
    fe, ns = _perfect_octahedron()
    angles = trans_angles(fe, ns)
    assert len(angles) == 3
    for a in angles:
        assert a == pytest.approx(180.0, abs=1e-6)


def test_trans_angles_rejects_wrong_count():
    fe, ns = _perfect_sp()
    with pytest.raises(ValueError):
        trans_angles(fe, ns)


# ----------------------------------------------------------------------
# coordination_geometry composite — classification + value
# ----------------------------------------------------------------------

def test_coordination_geometry_octahedral():
    fe, ns = _perfect_octahedron()
    g = coordination_geometry(fe, ns)
    assert isinstance(g, CoordinationGeometry)
    assert g.coordination_number == 6
    assert g.classification == "octahedral"
    assert g.trans_angles_deg is not None
    assert len(g.trans_angles_deg) == 3
    for a in g.trans_angles_deg:
        assert a == pytest.approx(180.0, abs=1e-6)


def test_coordination_geometry_tbp():
    fe, ns = _perfect_tbp()
    g = coordination_geometry(fe, ns)
    assert g.coordination_number == 5
    assert g.classification == "trigonal bipyramidal"
    assert g.tau == pytest.approx(1.0, abs=1e-9)
    assert g.tau_label == "τ₅"


def test_coordination_geometry_square_pyramidal():
    fe, ns = _perfect_sp()
    g = coordination_geometry(fe, ns)
    assert g.coordination_number == 5
    assert g.classification == "square pyramidal"
    assert g.tau == pytest.approx(0.0, abs=1e-9)


def test_coordination_geometry_tetrahedral():
    fe, ns = _perfect_tetrahedron()
    g = coordination_geometry(fe, ns)
    assert g.coordination_number == 4
    assert g.classification == "tetrahedral"
    assert g.tau is not None and g.tau > 0.9
    assert g.tau_label == "τ₄"


def test_coordination_geometry_square_planar():
    fe, ns = _perfect_square_planar()
    g = coordination_geometry(fe, ns)
    assert g.coordination_number == 4
    assert g.classification == "square planar"
    assert g.tau == pytest.approx(0.0, abs=1e-9)


def test_coordination_geometry_no_ligands():
    g = coordination_geometry(np.zeros(3), [])
    assert g.coordination_number == 0
    assert g.classification == "no ligands"


def test_coordination_geometry_distorted_5_coord():
    """A 5-coord geometry midway between SP and TBP should land in the
    'distorted' band (not classified as either pure form).

    Build the SP and rotate one basal atom slightly out of the plane so
    that τ₅ ≈ 0.5.
    """
    fe = np.zeros(3)
    # Start from TBP, then 'flatten' one axial slightly toward the eq
    # plane until τ₅ ~ 0.5.
    r = R
    sqrt3 = math.sqrt(3)
    ns = np.array([
        [ 0, 0, +r],
        # 'tilted' second axial — 30° away from -z toward +x:
        [ r * math.sin(math.radians(30)), 0, -r * math.cos(math.radians(30))],
        [ r,           0,                 0],
        [-r/2,  r * sqrt3/2,              0],
        [-r/2, -r * sqrt3/2,              0],
    ])
    g = coordination_geometry(fe, ns)
    assert g.coordination_number == 5
    assert 0.20 < g.tau < 0.80
    assert "distorted" in g.classification


def test_accepts_object_coordinates():
    """Σ should work when ligands expose .x .y .z (the ccdc shape)."""
    fe = _XYZ(0, 0, 0)
    ns = [
        _XYZ(+R, 0, 0), _XYZ(-R, 0, 0),
        _XYZ(0, +R, 0), _XYZ(0, -R, 0),
        _XYZ(0, 0, +R), _XYZ(0, 0, -R),
    ]
    assert sigma(fe, ns) == pytest.approx(0.0, abs=1e-9)
    assert theta(fe, ns) == pytest.approx(0.0, abs=1e-9)
