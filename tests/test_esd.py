"""
Tests for src/esd.py — parse / format / propagate esd values.

The "user's four cases" parametrised test below is the key acceptance
criterion: each case is taken verbatim from Prompt 5 and corresponds
to a specific crystallographic convention (1-sig-fig esd, round value
to match esd precision, missing esd → bare value, trailing zero
needed to indicate precision).
"""

from __future__ import annotations
import math

import pytest
import streamlit  # noqa: F401  (macOS import-order rule)

from src.esd import (
    format_with_esd,
    parse_value_with_esd,
    propagate_mean_esd,
)


# ----------------------------------------------------------------------
# 1) format_with_esd — the spec from Prompt 5
# ----------------------------------------------------------------------

@pytest.mark.parametrize("value,esd,expected", [
    # The four named cases in the prompt:
    (1.984, 0.007,  "1.984(7)"),
    (1.984, 0.012,  "1.98(1)"),
    (1.984, None,   "1.984"),
    (1.984, 0.0007, "1.9840(7)"),
    # A handful of natural extensions:
    (2.0,   0.1,    "2.0(1)"),         # esd at 10⁻¹
    (2.0,   None,   "2.000"),          # default_decimals = 3
    (1.999, 0.005,  "1.999(5)"),
    (10.42, 0.05,   "10.42(5)"),
    # Carrying when the esd rounds to ≥ 10 at its native position:
    (1.984, 0.0095, "1.98(1)"),        # 9.5×10⁻³ rounds to 1×10⁻²
    # Big esd >= 1:
    (1.984, 1.5,    "2(2)"),
])
def test_format_with_esd_named_cases(value, esd, expected):
    assert format_with_esd(value, esd) == expected


def test_format_with_esd_zero_or_negative_is_treated_as_missing():
    """A non-positive esd is meaningless and should be ignored."""
    assert format_with_esd(1.984, 0.0)   == "1.984"
    assert format_with_esd(1.984, -0.01) == "1.984"
    assert format_with_esd(1.984, math.nan) == "1.984"


def test_format_with_esd_default_decimals_param():
    """When esd is missing, default_decimals controls the precision."""
    assert format_with_esd(1.984, None, default_decimals=4) == "1.9840"
    assert format_with_esd(1.984, None, default_decimals=2) == "1.98"


# ----------------------------------------------------------------------
# 2) parse_value_with_esd
# ----------------------------------------------------------------------

@pytest.mark.parametrize("s,expected_value,expected_esd", [
    ("1.976(8)",   1.976,  0.008),
    ("1.976",      1.976,  None),
    ("2.0",        2.0,    None),
    ("100.00(12)", 100.0,  0.12),
    ("-3.50(15)",  -3.5,   0.15),
    ("0.005(1)",   0.005,  0.001),
    ("12(3)",      12.0,   3.0),          # esd applies to the units place
    ("  1.97 ",    1.97,   None),         # whitespace
])
def test_parse_value_with_esd(s, expected_value, expected_esd):
    v, e = parse_value_with_esd(s)
    assert v == pytest.approx(expected_value, rel=1e-12, abs=1e-12)
    if expected_esd is None:
        assert e is None
    else:
        assert e == pytest.approx(expected_esd, rel=1e-12, abs=1e-12)


def test_parse_value_with_esd_rejects_garbage():
    with pytest.raises(ValueError):
        parse_value_with_esd("not a number")
    with pytest.raises(ValueError):
        parse_value_with_esd("1.5 km")


def test_parse_format_round_trip_for_single_digit_esds():
    """parse → format → parse is exact for single-digit esds.

    Multi-digit esds (e.g. "100.00(12)") are deliberately rounded to
    one significant figure on format — that's the user's spec
    (see test_format_with_esd_named_cases[1.984-0.012] → "1.98(1)").
    So we test the round-trip only on inputs whose esd already has
    one significant digit.
    """
    samples = ["1.976(8)", "2.001(3)", "0.005(1)"]
    for s in samples:
        v, e = parse_value_with_esd(s)
        s2 = format_with_esd(v, e)
        v2, e2 = parse_value_with_esd(s2)
        assert v == pytest.approx(v2)
        assert e == pytest.approx(e2)


# ----------------------------------------------------------------------
# 3) propagate_mean_esd — standard error of the mean
# ----------------------------------------------------------------------

def test_propagate_mean_esd_uniform_uncertainties():
    """SEM with N identical σ-values is σ/√N."""
    # σ = 0.01, N = 6 → SEM = 0.01 / √6 ≈ 0.00408
    assert propagate_mean_esd([0.01] * 6) == pytest.approx(
        0.01 / math.sqrt(6), rel=1e-9,
    )


def test_propagate_mean_esd_mixed_uncertainties():
    """The general formula √(Σσ²)/n."""
    esds = [0.005, 0.010, 0.008]
    expected = math.sqrt(0.005**2 + 0.010**2 + 0.008**2) / 3
    assert propagate_mean_esd(esds) == pytest.approx(expected, rel=1e-12)


def test_propagate_mean_esd_any_missing_returns_none():
    assert propagate_mean_esd([0.01, None, 0.01]) is None
    assert propagate_mean_esd([None]) is None


def test_propagate_mean_esd_empty_returns_none():
    assert propagate_mean_esd([]) is None


def test_format_mean_with_propagated_esd():
    """The end-to-end use case: build the mean's display string."""
    distances = [1.984, 1.992, 2.001]
    esds      = [0.007, 0.008, 0.009]
    mean = sum(distances) / len(distances)              # 1.99233...
    mean_esd = propagate_mean_esd(esds)                 # ~0.00461
    s = format_with_esd(mean, mean_esd)
    assert s == "1.992(5)"
