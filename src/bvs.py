"""
Bond-valence sum (BVS) for Fe–N coordination spheres.

A bond valence sᵢ is the contribution of one M–L bond to the formal
oxidation state of the metal:

    sᵢ = exp((R₀ − Rᵢ) / B)

with R₀ a tabulated reference distance and B a universal soft-shell
parameter (≈ 0.37 Å). The bond-valence sum

    BVS = Σᵢ sᵢ

should sit near the formal oxidation state of M. Significant
deviation suggests one of:
  - the wrong oxidation state has been assigned,
  - the wrong R₀ value has been used (e.g. spin-state-specific vs
    generic for Fe–N),
  - the coordination sphere is missing atoms or contains long
    secondary contacts.

References
----------
Brown & Altermatt   (1985) Acta Cryst. B 41, 244.   original universal R₀ table
Brese & O'Keeffe    (1991) Acta Cryst. B 47, 192.   anion extension
Liebschner et al.   (2017) Acta Cryst. D 73, 148.   spin-state-specific R₀ for Fe

Caveat (a teaching point baked into this module)
------------------------------------------------
With the Liebschner R₀ values, BVS for a typical Fe(II)-LS hexaamine
at d ≈ 1.97 Å overshoots 2 by ~1.5 valence units. This is why the
R₀ values are exposed as editable in the UI: chemists routinely tune
them ±0.03 Å for their specific ligand chemistry, and the module is
designed to make that adjustment visible rather than hidden.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable


# -----------------------------------------------------------------------
# Constants — literature R₀ values keyed by (oxidation, spin) tuple.
# -----------------------------------------------------------------------
DEFAULT_B: float = 0.37

# Spin-state-specific values (Liebschner 2017) take precedence when the
# user annotates spin. The "generic" entries are the historic universal
# values used as a fallback when spin is unknown.
R0_LITERATURE: dict[tuple[str, str], float] = {
    ("Fe(II)",  "LS"):      1.78,    # Liebschner 2017
    ("Fe(II)",  "HS"):      1.91,    # Liebschner 2017
    ("Fe(III)", "LS"):      1.70,    # Liebschner 2017
    ("Fe(III)", "HS"):      1.83,    # Liebschner 2017
    ("Fe(II)",  "generic"): 1.769,   # Brown & Altermatt 1985
    ("Fe(III)", "generic"): 1.815,   # Brese & O'Keeffe 1991
}

# Human-readable provenance for each entry (used in UI captions).
R0_SOURCE: dict[tuple[str, str], str] = {
    ("Fe(II)",  "LS"):      "Liebschner 2017 (Fe(II) LS)",
    ("Fe(II)",  "HS"):      "Liebschner 2017 (Fe(II) HS)",
    ("Fe(III)", "LS"):      "Liebschner 2017 (Fe(III) LS)",
    ("Fe(III)", "HS"):      "Liebschner 2017 (Fe(III) HS)",
    ("Fe(II)",  "generic"): "Brown & Altermatt 1985 (Fe(II) generic)",
    ("Fe(III)", "generic"): "Brese & O'Keeffe 1991 (Fe(III) generic)",
}


# -----------------------------------------------------------------------
# Core formula — pure function, the single source of arithmetic truth.
# -----------------------------------------------------------------------

def bond_valence_sum(distances: Iterable[float],
                     R0: float,
                     B: float = DEFAULT_B) -> float:
    """Sum of bond valences: BVS = Σᵢ exp((R₀ − Rᵢ) / B).

    Parameters
    ----------
    distances : iterable of float
        Fe–L bond lengths in Å. May be any non-empty length; for a
        true 'hex-coordinate Fe-N' BVS this is six values.
    R0 : float
        Reference distance in Å for the (metal, donor, spin) combo.
    B : float, optional
        Universal BVS parameter, default 0.37 Å.

    Returns
    -------
    float
        BVS. Should be close to the formal oxidation state Z.
    """
    return float(sum(math.exp((R0 - d) / B) for d in distances))


# -----------------------------------------------------------------------
# R₀ selection from annotations
# -----------------------------------------------------------------------

def choose_R0(oxidation_state: str,
              spin_state: str | None = None) -> tuple[float, str]:
    """Return (R₀, source_label) for the given annotations.

    Logic:
      spin = "LS" or "HS" → spin-specific R₀
      spin = anything else (None, "(unknown)", "IS", …) → generic R₀

    Raises
    ------
    KeyError
        If oxidation_state isn't "Fe(II)" or "Fe(III)".
    """
    key_spin = spin_state if spin_state in ("LS", "HS") else "generic"
    key = (oxidation_state, key_spin)
    return R0_LITERATURE[key], R0_SOURCE[key]


# -----------------------------------------------------------------------
# Result container + composite entry point
# -----------------------------------------------------------------------

@dataclass
class BVSResult:
    """One BVS computation, fully annotated for display.

    The UI renders bvs and R0_source, and uses (oxidation_state, bvs)
    to compute the consistency badge with consistency_status().
    """
    bvs: float
    R0: float
    R0_source: str
    B: float
    distances: list[float]
    oxidation_state: str


SUPPORTED_OXIDATION_STATES: tuple[str, ...] = (
    "Fe(0)", "Fe(II)", "Fe(III)", "Fe(IV)",
)


def has_literature_R0(oxidation_state: str,
                      spin_state: str | None = None) -> bool:
    """True if choose_R0(ox, spin) would succeed without raising."""
    key_spin = spin_state if spin_state in ("LS", "HS") else "generic"
    return (oxidation_state, key_spin) in R0_LITERATURE


def compute_bvs(distances: list[float],
                oxidation_state: str | None,
                spin_state: str | None = None,
                R0_override: float | None = None,
                B: float = DEFAULT_B) -> BVSResult | None:
    """Compute BVS for an (oxidation, spin) annotation pair.

    Returns None when:
      - oxidation_state isn't a supported Fe(X) string, or
      - distances is empty, or
      - oxidation_state has no literature R₀ AND no R0_override was
        supplied (this is how Fe(0) / Fe(IV) annotations land when the
        user hasn't yet entered a custom R₀ — we don't invent values).

    If R0_override is supplied it takes precedence over the literature
    value, and the source label becomes "user-supplied". This is how
    the editable-R₀ teaching feature is wired in.
    """
    if oxidation_state not in SUPPORTED_OXIDATION_STATES:
        return None
    distances = list(distances)
    if not distances:
        return None

    if R0_override is not None:
        R0, source = float(R0_override), "user-supplied"
    elif has_literature_R0(oxidation_state, spin_state):
        R0, source = choose_R0(oxidation_state, spin_state)
    else:
        # Annotated as Fe(0) or Fe(IV) but no custom R₀ yet — the UI
        # should prompt the user to supply one.
        return None

    return BVSResult(
        bvs=bond_valence_sum(distances, R0, B),
        R0=R0,
        R0_source=source,
        B=B,
        distances=distances,
        oxidation_state=oxidation_state,
    )


# -----------------------------------------------------------------------
# Consistency check against annotated oxidation state
# -----------------------------------------------------------------------

def consistency_status(bvs: float,
                       oxidation_state: str) -> tuple[str, str]:
    """Compare BVS to the nominal integer Z of the annotated oxidation
    state. Returns (level, message).

    Bands:
      level == "good"     |BVS − Z| ≤ 0.4   → green check ✓
      level == "caution"  0.4 < |Δ| ≤ 0.8   → yellow ⚠
      level == "warning"  |Δ| > 0.8         → red 🚫
    """
    Z_map = {"Fe(0)": 0, "Fe(II)": 2, "Fe(III)": 3, "Fe(IV)": 4}
    Z = Z_map.get(oxidation_state)
    if Z is None:
        return "warning", "oxidation state not recognised"

    deviation = abs(bvs - Z)

    if deviation <= 0.4:
        return "good", (
            f"consistent with {oxidation_state} "
            f"(|BVS − {Z}| = {deviation:.2f})"
        )
    if deviation <= 0.8:
        return "caution", (
            f"marginal — |BVS − {Z}| = {deviation:.2f}. "
            "Check R₀, distances, or coordination completeness."
        )

    # Suggest the nearest integer oxidation state.
    nearest = int(round(bvs))
    roman = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}.get(
        nearest, str(nearest)
    )
    alt = f"Fe({roman})"
    return "warning", (
        f"BVS = {bvs:.2f} is more consistent with {alt} "
        f"(|Δ| from {oxidation_state} = {deviation:.2f}). "
        f"Check the annotated oxidation state."
    )


# -----------------------------------------------------------------------
# Multi-oxidation-state probe (used when neither oxidation nor spin
# is annotated — let the user see which integer BVS lands closest to).
# -----------------------------------------------------------------------

def probe_both_oxidation_states(distances: list[float]
                                ) -> dict[str, BVSResult] | None:
    """Compute BVS for both Fe(II) and Fe(III) generic R₀.

    Useful as a sanity probe when the user hasn't annotated the
    oxidation state — the result closer to its integer Z is the more
    plausible assignment, with all usual caveats.
    """
    if not distances:
        return None
    out: dict[str, BVSResult] = {}
    for ox in ("Fe(II)", "Fe(III)"):
        result = compute_bvs(distances, ox, spin_state=None)
        if result is not None:
            out[ox] = result
    return out or None


# -----------------------------------------------------------------------
# Auto-inference — combine BVS with bond-length plausibility to suggest
# (oxidation, spin) without the user trial-and-erroring four combinations.
# -----------------------------------------------------------------------

@dataclass
class InferenceCandidate:
    """One row of the 4-combo ranking table the auto-inference produces."""
    oxidation: str               # Fe(II) or Fe(III)
    spin: str                    # LS or HS
    R0: float                    # Liebschner 2017 spin-specific value
    bvs: float                   # computed
    deviation: float             # |BVS − Z|
    bond_length_consistent: bool # is mean Fe-N in the Halcrow band for this combo?
    reason: str                  # one-line explanation


@dataclass
class InferenceResult:
    """All four candidates plus a recommended pick and confidence label."""
    mean_distance_A: float
    candidates: list[InferenceCandidate]   # ranked by deviation
    best_oxidation: str | None
    best_spin: str | None
    confidence: str                       # 'high' | 'medium' | 'low'
    caveats: list[str]


# Halcrow-style typical bond-length bands (Halcrow 2011, with a
# small margin around the published 1.95-2.00, 2.15-2.25, etc.
# centres). Calibrated so:
#   - 2.18 Å lands only in Fe(II)-HS (not Fe(III)-HS, whose upper
#     edge is ~2.16);
#   - 2.12 Å lands only in Fe(III)-HS (Fe(II)-HS lower edge ~2.13);
#   - the LS bands overlap intentionally — bond length alone cannot
#     distinguish Fe(II)-LS from Fe(III)-LS, and the inference falls
#     through to the smallest-|BVS-Z| tie-breaker.
_BAND_FE_II_LS  = (1.91, 2.05)
_BAND_FE_II_HS  = (2.13, 2.30)
_BAND_FE_III_LS = (1.92, 2.05)
_BAND_FE_III_HS = (2.02, 2.16)


def _bond_length_consistent_with(mean_d: float,
                                 oxidation: str, spin: str) -> tuple[bool, str]:
    """True if `mean_d` falls in the Halcrow band for this (ox, spin).

    Returns (consistent, reason_string). The reason explains *how* the
    distance compares, even when consistent — useful for the UI to
    show next to each candidate row.
    """
    bands = {
        ("Fe(II)",  "LS"):  _BAND_FE_II_LS,
        ("Fe(II)",  "HS"):  _BAND_FE_II_HS,
        ("Fe(III)", "LS"):  _BAND_FE_III_LS,
        ("Fe(III)", "HS"):  _BAND_FE_III_HS,
    }
    lo, hi = bands.get((oxidation, spin), (0.0, 10.0))
    if lo <= mean_d <= hi:
        return True, f"mean {mean_d:.3f} Å in Halcrow band {lo:.2f}–{hi:.2f} Å"
    if mean_d < lo:
        return False, (
            f"mean {mean_d:.3f} Å is below the typical "
            f"{oxidation} {spin} band ({lo:.2f}–{hi:.2f} Å)"
        )
    return False, (
        f"mean {mean_d:.3f} Å is above the typical "
        f"{oxidation} {spin} band ({lo:.2f}–{hi:.2f} Å)"
    )


_Z_MAP = {"Fe(II)": 2, "Fe(III)": 3}


def infer_oxidation_spin(distances: list[float]) -> InferenceResult | None:
    """Rank Fe(II)/Fe(III) × LS/HS by BVS deviation + bond-length plausibility.

    The recommendation prefers candidates whose Halcrow bond-length
    band the actual mean falls into; among consistent candidates the
    smallest |BVS − Z| wins. When *no* combo is bond-length-consistent
    we still return the smallest-deviation pick but flag the
    inference as `confidence = "low"` and surface the issue in
    `caveats`.

    Returns None for empty / missing distances.

    This is a heuristic — see docs/methodology.md for the bands and
    the porphyrin-overshoot teaching case.
    """
    if not distances:
        return None
    distances = list(distances)
    mean_d = sum(distances) / len(distances)

    candidates: list[InferenceCandidate] = []
    for ox in ("Fe(II)", "Fe(III)"):
        Z = _Z_MAP[ox]
        for spin in ("LS", "HS"):
            R0, _src = choose_R0(ox, spin)
            bvs = bond_valence_sum(distances, R0)
            consistent, reason = _bond_length_consistent_with(mean_d, ox, spin)
            candidates.append(InferenceCandidate(
                oxidation=ox, spin=spin, R0=R0, bvs=bvs,
                deviation=abs(bvs - Z),
                bond_length_consistent=consistent,
                reason=reason,
            ))

    candidates.sort(key=lambda c: c.deviation)

    consistent_pool = [c for c in candidates if c.bond_length_consistent]
    caveats: list[str] = []

    if consistent_pool:
        # Pick smallest deviation among bond-length-consistent combos.
        consistent_pool.sort(key=lambda c: c.deviation)
        best = consistent_pool[0]
        if best.deviation <= 0.4:
            confidence = "high"
        elif best.deviation <= 0.8:
            confidence = "medium"
        else:
            confidence = "low"
        best_ox, best_spin = best.oxidation, best.spin
    else:
        best = candidates[0]
        best_ox, best_spin = best.oxidation, best.spin
        confidence = "low"
        caveats.append(
            "No (oxidation, spin) combination is fully consistent with "
            "the bond length. The recommendation is the best BVS match "
            "but should be cross-checked against the original "
            "publication or magnetic data."
        )

    # Porphyrin / macrocycle caveat. If the actual chemistry is Fe(II)
    # LS but the rigid macrocycle pushes BVS up so much that the
    # nearest-integer best fit lands at Fe(III), warn about it.
    fe_ii_ls_dev = next(
        c.deviation for c in candidates
        if c.oxidation == "Fe(II)" and c.spin == "LS"
    )
    if (best_ox == "Fe(III)" and best_spin in ("LS", "HS")
            and 1.95 <= mean_d <= 2.02
            and fe_ii_ls_dev > 1.0):
        caveats.append(
            "Bond length is in the LS region (1.95–2.02 Å) but the "
            "smallest-deviation BVS pick is Fe(III). This is the "
            "documented **porphyrin / macrocycle case** — the "
            "Liebschner Fe(II)-LS R₀ systematically overshoots for "
            "rigid macrocyclic ligands. If your structure has a "
            "porphyrin / corrole / phthalocyanine ring, the "
            "chemistry probably says **Fe(II) LS** regardless of "
            "what this BVS suggests. Tune R₀ down to ~1.72 in Mode 1 "
            "to see this directly."
        )

    return InferenceResult(
        mean_distance_A=mean_d,
        candidates=candidates,
        best_oxidation=best_ox,
        best_spin=best_spin,
        confidence=confidence,
        caveats=caveats,
    )
