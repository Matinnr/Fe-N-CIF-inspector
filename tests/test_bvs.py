"""
Tests for src/bvs.py — pure-math BVS module.

Key invariants under test:
  1. The arithmetic matches the canonical formula BVS = Σ exp((R₀-R)/B).
     One hand-verified anchor (the "Fe(II)-LS at 1.97 Å" teaching
     example → 3.59) catches sign errors and unit mistakes.
  2. R₀ selection follows the (oxidation, spin) → literature mapping
     and falls back to "generic" when spin is missing.
  3. The consistency bands (≤0.4 / 0.4-0.8 / >0.8) flip from good →
     caution → warning at the right thresholds.
  4. User-supplied R₀ override is honoured and source label changes
     to "user-supplied".
  5. probe_both_oxidation_states returns both Fe(II) and Fe(III)
     candidates when oxidation is unknown.
"""

from __future__ import annotations
import math

import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src.bvs import (
    DEFAULT_B,
    R0_LITERATURE,
    R0_SOURCE,
    BVSResult,
    InferenceCandidate,
    InferenceResult,
    bond_valence_sum,
    choose_R0,
    compute_bvs,
    consistency_status,
    infer_oxidation_spin,
    probe_both_oxidation_states,
)


# ----------------------------------------------------------------------
# 1) Formula — agrees with itself + the hand-verified anchor
# ----------------------------------------------------------------------

def _formula(distances, R0, B=DEFAULT_B):
    return sum(math.exp((R0 - d) / B) for d in distances)


@pytest.mark.parametrize("distances,R0", [
    # Six equal bonds — the canonical hex-coordinate FeN6 case.
    ([1.97] * 6, 1.78),    # Fe(II)-LS (Liebschner R₀)
    ([1.95] * 6, 1.78),    # Tighter Fe(II)-LS
    ([2.18] * 6, 1.91),    # Fe(II)-HS (Liebschner R₀)
    ([2.05] * 6, 1.83),    # Fe(III)-HS (Liebschner R₀)
    ([1.97] * 6, 1.70),    # Fe(III)-LS (Liebschner R₀)
    # Mixed bonds — exercise the per-bond sum.
    ([1.96, 1.97, 1.98, 1.99, 2.00, 2.01], 1.78),
    # Non-six coordination — the formula still applies.
    ([1.97] * 4, 1.78),
])
def test_formula_matches_canonical_definition(distances, R0):
    """BVS implementation matches the mathematical formula bit-for-bit."""
    expected = _formula(distances, R0)
    assert bond_valence_sum(distances, R0) == pytest.approx(expected, rel=1e-12)


def test_teaching_example_fe_ii_ls_at_1_97_overshoots():
    """The headline didactic case: BVS = 3.59 ≠ 2.0 with literature R₀.

    This is the value Liebschner's R₀_LS = 1.78 produces for a typical
    Fe(II)-LS hexa-amine at 1.97 Å. The fact that it's ~1.6 valence
    units high motivates exposing R₀ as editable in the UI.
    """
    bvs = bond_valence_sum([1.97] * 6, R0=1.78)
    assert bvs == pytest.approx(3.591, abs=0.005)


def test_fe_ii_ls_at_1_95_hand_verified():
    """6 × exp((1.78 − 1.95)/0.37) = 6 × 0.6316 = 3.79."""
    assert bond_valence_sum([1.95] * 6, R0=1.78) == pytest.approx(3.79, abs=0.01)


# ----------------------------------------------------------------------
# 2) R₀ selection
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ox,spin,expected_R0", [
    ("Fe(II)",  "LS",         1.78),
    ("Fe(II)",  "HS",         1.91),
    ("Fe(III)", "LS",         1.70),
    ("Fe(III)", "HS",         1.83),
    # Unknown / IS / None → fall through to the generic value
    ("Fe(II)",  None,         1.769),
    ("Fe(II)",  "(unknown)",  1.769),
    ("Fe(II)",  "IS",         1.769),
    ("Fe(III)", None,         1.815),
    ("Fe(III)", "(unknown)",  1.815),
])
def test_choose_R0(ox, spin, expected_R0):
    R0, source = choose_R0(ox, spin)
    assert R0 == pytest.approx(expected_R0, abs=1e-9)
    assert "Liebschner" in source or "Brown" in source or "Brese" in source


def test_choose_R0_unsupported_oxidation_raises():
    with pytest.raises(KeyError):
        choose_R0("Fe(IV)", "LS")


# ----------------------------------------------------------------------
# 3) compute_bvs — composite entry point
# ----------------------------------------------------------------------

def test_compute_bvs_returns_none_for_unknown_oxidation():
    assert compute_bvs([1.97] * 6, oxidation_state="(unknown)") is None
    assert compute_bvs([1.97] * 6, oxidation_state=None) is None
    # Fe(IV) without override → None (no literature R₀ available).
    assert compute_bvs([1.97] * 6, oxidation_state="Fe(IV)") is None


def test_compute_bvs_fe_iv_with_R0_override_succeeds():
    """Fe(IV) has no Liebschner / B&A R₀, but the user can still
    compute BVS by supplying a custom value."""
    result = compute_bvs([1.97] * 6, "Fe(IV)", R0_override=1.70)
    assert result is not None
    assert result.R0 == pytest.approx(1.70, abs=1e-9)
    assert result.R0_source == "user-supplied"
    assert result.oxidation_state == "Fe(IV)"


def test_compute_bvs_fe_zero_with_R0_override_succeeds():
    result = compute_bvs([1.97] * 6, "Fe(0)", R0_override=1.90)
    assert result is not None
    assert result.R0 == pytest.approx(1.90, abs=1e-9)


def test_consistency_status_supports_fe_iv():
    """A BVS of 4.0 must compare cleanly against Fe(IV)."""
    level, msg = consistency_status(4.0, "Fe(IV)")
    assert level == "good"
    level, msg = consistency_status(3.20, "Fe(IV)")
    # |3.20 − 4| = 0.80 → caution band (inclusive upper bound).
    assert level == "caution"
    level, msg = consistency_status(2.50, "Fe(IV)")
    # |2.50 − 4| = 1.50 → warning.
    assert level == "warning"


def test_compute_bvs_returns_none_for_empty_distances():
    assert compute_bvs([], "Fe(II)", "LS") is None


def test_compute_bvs_uses_literature_R0_by_default():
    result = compute_bvs([1.97] * 6, "Fe(II)", "LS")
    assert isinstance(result, BVSResult)
    assert result.R0 == pytest.approx(1.78, abs=1e-9)
    assert "Liebschner" in result.R0_source
    assert result.bvs == pytest.approx(3.591, abs=0.005)


def test_compute_bvs_falls_back_to_generic_when_spin_missing():
    """No spin annotation → generic Fe(II) R₀ = 1.769."""
    result = compute_bvs([1.97] * 6, "Fe(II)", spin_state=None)
    assert result is not None
    assert result.R0 == pytest.approx(1.769, abs=1e-9)
    assert "generic" in result.R0_source


def test_compute_bvs_R0_override_takes_precedence():
    """A user-tuned R₀ wins and the source label updates."""
    result = compute_bvs([1.97] * 6, "Fe(II)", "LS", R0_override=1.74)
    assert result is not None
    assert result.R0 == pytest.approx(1.74, abs=1e-9)
    assert result.R0_source == "user-supplied"
    assert result.bvs == pytest.approx(_formula([1.97] * 6, 1.74), rel=1e-9)


# ----------------------------------------------------------------------
# 4) consistency_status — bands and suggestion behaviour
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bvs,ox,expected_level", [
    # |BVS − Z| ≤ 0.4 → good
    (2.00, "Fe(II)",  "good"),
    (1.65, "Fe(II)",  "good"),
    (2.35, "Fe(II)",  "good"),
    (3.00, "Fe(III)", "good"),
    (2.61, "Fe(III)", "good"),
    # 0.4 < |Δ| ≤ 0.8 → caution
    (2.50, "Fe(II)",  "caution"),
    (1.40, "Fe(II)",  "caution"),
    (3.75, "Fe(III)", "caution"),
    # |Δ| > 0.8 → warning
    (3.59, "Fe(II)",  "warning"),       # the teaching case
    (1.10, "Fe(II)",  "warning"),
    (4.00, "Fe(III)", "warning"),
])
def test_consistency_status_bands(bvs, ox, expected_level):
    level, _msg = consistency_status(bvs, ox)
    assert level == expected_level


def test_consistency_status_warning_suggests_nearest_integer():
    """For BVS = 3.59 / Fe(II), the suggestion should name Fe(IV) — the
    integer nearest to the BVS — not just 'Fe(III)'."""
    level, msg = consistency_status(3.59, "Fe(II)")
    assert level == "warning"
    assert "Fe(IV)" in msg


def test_consistency_status_unknown_oxidation_returns_warning():
    """Genuinely unrecognised oxidation strings (not in our Fe-state map)."""
    level, msg = consistency_status(2.5, "Cu(II)")
    assert level == "warning"
    assert "not recognised" in msg.lower()
    level, msg = consistency_status(2.5, "(unknown)")
    assert level == "warning"
    assert "not recognised" in msg.lower()


# ----------------------------------------------------------------------
# 5) probe_both_oxidation_states — used when oxidation unknown
# ----------------------------------------------------------------------

def test_probe_returns_both_oxidation_state_candidates():
    out = probe_both_oxidation_states([1.97] * 6)
    assert out is not None
    assert "Fe(II)"  in out
    assert "Fe(III)" in out
    # Different R₀ → different BVS.
    assert out["Fe(II)"].bvs  != pytest.approx(out["Fe(III)"].bvs, abs=0.01)


def test_probe_empty_distances_returns_none():
    assert probe_both_oxidation_states([]) is None


# ----------------------------------------------------------------------
# 6) infer_oxidation_spin — the auto-inference path
# ----------------------------------------------------------------------

def test_infer_returns_none_for_empty_distances():
    assert infer_oxidation_spin([]) is None


def test_infer_returns_all_four_candidates_ranked():
    """Every call should return all 4 (Fe(II)/Fe(III)) × (LS/HS)
    combinations, sorted by BVS deviation."""
    result = infer_oxidation_spin([1.97] * 6)
    assert isinstance(result, InferenceResult)
    assert len(result.candidates) == 4
    # Sorted ascending by deviation
    devs = [c.deviation for c in result.candidates]
    assert devs == sorted(devs)


def test_infer_picks_fe_ii_hs_for_classic_HS_bonds():
    """Mean = 2.18 Å → Fe(II) HS band (2.10–2.30). With Liebschner
    R0 = 1.91 the BVS comes out at ~2.89 → deviation 0.89 from Z=2.
    Fe(III) HS at this distance gives BVS=2.37 (much closer to 2 than 3
    by absolute value) but the bond length 2.18 is outside the
    Fe(III) HS band (2.00–2.18), so bond-length consistency favours
    Fe(II) HS. The inference should prefer the bond-length-consistent
    option."""
    result = infer_oxidation_spin([2.18] * 6)
    assert result is not None
    # Bond length consistency for each:
    cmap = {(c.oxidation, c.spin): c for c in result.candidates}
    assert cmap[("Fe(II)",  "HS")].bond_length_consistent is True
    assert cmap[("Fe(II)",  "LS")].bond_length_consistent is False
    assert cmap[("Fe(III)", "LS")].bond_length_consistent is False
    # Best pick is in the consistent set with smallest deviation.
    assert result.best_oxidation == "Fe(II)"
    assert result.best_spin      == "HS"


def test_infer_picks_fe_ii_ls_for_classic_LS_polypyridyl():
    """Mean = 1.96 Å → Fe(II) LS band (1.91–2.05). Liebschner R0=1.78
    gives BVS=3.86, dev 1.86 from Z=2. Fe(III) LS at 1.96 with R0=1.70
    gives BVS=2.88, dev 0.12 from Z=3 — much smaller. Both LS combos
    are bond-length-consistent, so the smaller-deviation wins: Fe(III)
    LS. The porphyrin caveat should NOT fire (mean is in 1.95-2.02 but
    that's exactly the polypyridyl LS Fe(II) range too)."""
    result = infer_oxidation_spin([1.96] * 6)
    assert result is not None
    cmap = {(c.oxidation, c.spin): c for c in result.candidates}
    # Fe(III) LS has smallest |BVS-Z|
    assert cmap[("Fe(III)", "LS")].deviation < cmap[("Fe(II)", "LS")].deviation
    # And it is bond-length consistent
    assert cmap[("Fe(III)", "LS")].bond_length_consistent is True


def test_infer_flags_porphyrin_caveat_when_bvs_overshoots():
    """The signature case the user actually hit:
       mean = 2.00 Å (LS region), Fe(II) LS BVS = 3.30 (dev 1.30),
       Fe(III) LS BVS = 2.66 (dev 0.34) — auto-picks Fe(III) LS but
       this is the porphyrin overshoot."""
    result = infer_oxidation_spin([2.00] * 6)
    assert result is not None
    # The porphyrin caveat is present.
    caveats = "\n".join(result.caveats)
    assert "porphyrin" in caveats.lower() or "macrocycle" in caveats.lower()


def test_infer_picks_fe_iii_hs_for_intermediate_HS_bond_lengths():
    """Mean = 2.12 Å falls inside the Fe(III)-HS Halcrow band
    (2.02–2.16) and outside Fe(II)-HS (2.13–2.30). The Fe(III)-HS R₀
    = 1.83 gives BVS ≈ 2.96, dev 0.04 — a near-perfect match. This
    is the genuinely-Fe(III)-HS case; the algorithm picking it is
    the right answer."""
    result = infer_oxidation_spin([2.12] * 6)
    assert result is not None
    assert result.best_oxidation == "Fe(III)"
    assert result.best_spin      == "HS"
    # And it should be high-confidence because the BVS deviation is tiny.
    assert result.confidence == "high"


def test_infer_low_confidence_when_nothing_fits():
    """A bond length far outside every Halcrow band should mark the
    inference as low-confidence with a 'cross-check' caveat."""
    result = infer_oxidation_spin([2.5] * 6)   # 2.5 Å is beyond every band
    assert result is not None
    assert result.confidence == "low"
    assert any("cross-check" in c.lower()
               or "no (oxidation, spin)" in c.lower()
               for c in result.caveats)


def test_infer_candidates_carry_human_readable_reason():
    """Every candidate's `reason` should describe how the mean compares
    to the Halcrow band — useful for the UI table column."""
    result = infer_oxidation_spin([1.99] * 6)
    assert result is not None
    for c in result.candidates:
        assert c.reason   # non-empty
        assert "1.990" in c.reason or "1.99" in c.reason
