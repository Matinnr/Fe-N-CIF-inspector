"""
Tests for src/plotting.py — the Plotly chart builders.

These are smoke tests: they call each chart function with a small but
realistic input and assert the returned object is a plotly Figure
that Plotly accepts (i.e. validates without raising). The big payoff
is catching xref/yref typos in add_shape calls early — Plotly only
notices those at the renderer step, which previously meant the bug
shipped to the UI and only surfaced when a user hit the panel.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit  # noqa: F401  (macOS import-order rule)

from src.plotting import (
    lollipop,
    sigma_reference_chart,
    summary_cards,
)
from src.data_schema import (
    COL_CIF_FILE, COL_STRUCTURE_ID, COL_TEMPERATURE, COL_FE_LABEL,
    COL_N_LABEL, COL_DISTANCE, COL_DISTANCE_ESD, COL_DETECTION_METHOD,
    COL_SYMMETRY_RELATED, COL_WARNING,
    BOND_TABLE_COLUMNS, DetectionMethod,
)


def _bond_table(distances: list[float]) -> pd.DataFrame:
    """Build a minimal valid bond DataFrame with one Fe centre."""
    return pd.DataFrame([
        {
            COL_CIF_FILE: "demo.cif",
            COL_STRUCTURE_ID: "demo",
            COL_TEMPERATURE: 100.0,
            COL_FE_LABEL: "Fe1",
            COL_N_LABEL: f"N{i + 1}",
            COL_DISTANCE: d,
            COL_DISTANCE_ESD: None,
            COL_DETECTION_METHOD: DetectionMethod.FORMAL_BOND.value,
            COL_SYMMETRY_RELATED: False,
            COL_WARNING: "",
        } for i, d in enumerate(distances)
    ], columns=BOND_TABLE_COLUMNS)


def test_summary_cards_with_data():
    df = _bond_table([1.99, 2.00, 2.01])
    cards = summary_cards(df, n_fe=1)
    assert cards["n_distances"] == "3"
    assert cards["mean_A"] == "2.000"
    assert cards["min_A"]  == "1.990"
    assert cards["max_A"]  == "2.010"


def test_summary_cards_empty_returns_dashes():
    cards = summary_cards(pd.DataFrame(), n_fe=0)
    assert cards["mean_A"] == "—"


def test_lollipop_returns_figure_for_valid_input():
    df = _bond_table([1.99, 2.00, 2.01, 2.02, 2.03, 2.04])
    fig = lollipop(df, show_spin_bands=False)
    assert isinstance(fig, go.Figure)
    # Forces Plotly to validate everything — would raise on a bad
    # xref/yref or out-of-domain enum value.
    fig.to_dict()


def test_lollipop_with_spin_bands_validates_in_plotly():
    df = _bond_table([1.95, 1.96, 1.97, 1.98, 1.99, 2.00])
    fig = lollipop(df, show_spin_bands=True, oxidation_state="Fe(II)")
    fig.to_dict()


def test_lollipop_empty_returns_placeholder():
    fig = lollipop(pd.DataFrame(columns=BOND_TABLE_COLUMNS))
    assert isinstance(fig, go.Figure)
    fig.to_dict()


def test_sigma_reference_chart_validates_in_plotly():
    """Regression test for the xref/yref swap that previously
    raised ValueError at render time."""
    sigma_map = {"Fe1": 12.76, "Fe51": 10.77}
    fig = sigma_reference_chart(sigma_map, "Fe(II)")
    assert isinstance(fig, go.Figure)
    # This is the call that would have failed under the bug:
    fig.to_dict()


def test_sigma_reference_chart_skips_none_values():
    sigma_map = {"Fe1": None, "Fe2": 35.0}
    fig = sigma_reference_chart(sigma_map, "Fe(II)")
    fig.to_dict()


def test_sigma_reference_chart_handles_no_valid_values():
    sigma_map = {"Fe1": None}
    fig = sigma_reference_chart(sigma_map, "Fe(II)")
    fig.to_dict()


def test_sigma_reference_chart_unsupported_oxidation_returns_empty_figure():
    """No bands defined for Fe(IV) — function should not crash."""
    fig = sigma_reference_chart({"Fe1": 12.0}, "Fe(IV)")
    assert isinstance(fig, go.Figure)
    fig.to_dict()
