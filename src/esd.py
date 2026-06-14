"""
Standard-uncertainty (esd) parsing, formatting, and propagation.

Crystallographic CIFs encode bond lengths as e.g. ``1.976(8)``,
meaning *the value 1.976 with an estimated standard deviation of
0.008 Å — that is, the parenthesised digit represents the esd in
the last decimal place(s) of the value.*

This module provides the three small functions that handle the
notation end-to-end:

  parse_value_with_esd("1.976(8)")     →  (1.976, 0.008)
  format_with_esd(1.984, 0.007)        →  "1.984(7)"
  propagate_mean_esd([0.007, 0.008])   →  SEM via √(Σσ²)/n

The functions are deliberately pure — they take and return only
numbers / strings, no chemistry-specific assumptions and no
dependencies. They are the "crystallographer-competence" detail
that turns a generic Fe–N table into something a coordination
chemist trusts at a glance.
"""

from __future__ import annotations
import math
import re


# ----------------------------------------------------------------------
# Parsing  "1.976(8)"  →  (1.976, 0.008)
# ----------------------------------------------------------------------

# Accept:  +/-  digits . digits  optional exponent  optional (esd digits)
_VALUE_ESD_RE = re.compile(
    r"^\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"   # value
    r"(?:\(([0-9]+)\))?\s*$"                   # optional (esd)
)


def parse_value_with_esd(s: str) -> tuple[float, float | None]:
    """Parse a CIF-style number with optional esd suffix.

    Examples
    --------
    >>> parse_value_with_esd("1.976(8)")
    (1.976, 0.008)
    >>> parse_value_with_esd("100.00(12)")
    (100.0, 0.12)
    >>> parse_value_with_esd("2.0")
    (2.0, None)
    >>> parse_value_with_esd(" -3.50(15) ")
    (-3.5, 0.15)

    Raises
    ------
    ValueError
        If `s` doesn't look like a number (possibly with an esd suffix).
    """
    m = _VALUE_ESD_RE.match(s)
    if not m:
        raise ValueError(f"could not parse value+esd from {s!r}")
    value_str, esd_str = m.group(1), m.group(2)
    value = float(value_str)
    if esd_str is None:
        return value, None
    # The esd integer applies to the LAST decimal place(s) of value_str.
    # "1.976(8)"   → value_str "1.976", esd_str "8"  → esd 0.008
    # "100.00(12)" → value_str "100.00", esd_str "12" → esd 0.12
    if "." in value_str:
        decimals = len(value_str.split(".")[1])
    else:
        decimals = 0
    esd = int(esd_str) * 10 ** (-decimals)
    return value, esd


# ----------------------------------------------------------------------
# Formatting  (1.984, 0.007)  →  "1.984(7)"
# ----------------------------------------------------------------------

def format_with_esd(value: float,
                    esd: float | None,
                    *, default_decimals: int = 3) -> str:
    """Render a value with crystallographic-style esd notation.

    The esd is shown in parentheses at the precision of the last
    displayed decimal place of the value. The value is rounded to that
    precision, so the result is internally consistent.

    Examples
    --------
    >>> format_with_esd(1.984, 0.007)
    '1.984(7)'
    >>> format_with_esd(1.984, 0.012)
    '1.98(1)'
    >>> format_with_esd(1.984, None)
    '1.984'
    >>> format_with_esd(1.984, 0.0007)
    '1.9840(7)'
    >>> format_with_esd(2.0, 0.1)
    '2.0(1)'

    Parameters
    ----------
    value : float
    esd : float or None
        If None / non-positive / non-finite, returned without the
        parenthesised esd. The value is shown to `default_decimals`.
    default_decimals : int
        Decimal places to display when esd is missing.
    """
    if esd is None or not math.isfinite(esd) or esd <= 0:
        return f"{value:.{default_decimals}f}"

    # Position of the first significant digit of the esd, e.g.
    #   esd 0.007    → log10 ≈ -2.15 → position -3
    #   esd 0.012    → log10 ≈ -1.92 → position -2
    #   esd 0.0007   → log10 ≈ -3.15 → position -4
    position = math.floor(math.log10(esd))

    # Single-digit esd at that position.
    digit = round(esd * 10 ** (-position))

    # Edge case: rounding may carry — e.g. esd 0.0095 → "10" at
    # position -3 becomes "1" at position -2.
    if digit >= 10:
        position += 1
        digit = round(esd * 10 ** (-position))

    decimal_places = max(0, -position)
    value_str = f"{value:.{decimal_places}f}"
    return f"{value_str}({digit})"


# ----------------------------------------------------------------------
# Propagation — standard error of the mean
# ----------------------------------------------------------------------

def propagate_mean_esd(esds: list[float | None]) -> float | None:
    """Standard error of the mean: σ_mean = √(Σ σᵢ²) / n.

    Returns None when the list is empty or any esd is missing — we
    don't fabricate a mean uncertainty for an incomplete dataset.

    Examples
    --------
    >>> propagate_mean_esd([0.01, 0.01, 0.01])
    0.005773502691896258
    >>> propagate_mean_esd([0.01, None, 0.01]) is None
    True
    >>> propagate_mean_esd([]) is None
    True
    """
    if not esds:
        return None
    if any(e is None for e in esds):
        return None
    n = len(esds)
    return math.sqrt(sum((e or 0.0) ** 2 for e in esds)) / n
