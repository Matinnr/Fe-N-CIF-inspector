"""
Plotly chart builders for the dashboard.

We only use Plotly here (not matplotlib) — Streamlit + Plotly gives us
interactive charts in the browser with zero extra config, and avoids the
matplotlib/cairo dylib conflict that was hitting the older scripts.

Functions here are pure: they take data in, return a Plotly figure, and
have no side effects (no file writes, no Streamlit calls).
"""

from __future__ import annotations
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from .data_schema import (
    COL_DETECTION_METHOD, COL_DISTANCE,
    COL_FE_LABEL, COL_N_LABEL, COL_WARNING,
    DetectionMethod, select_bands, select_sigma_bands,
)


# Colour map used in every chart so the visual language is consistent.
METHOD_COLOURS: dict[str, str] = {
    DetectionMethod.FORMAL_BOND.value:         "#1f77b4",  # blue
    DetectionMethod.GEOMETRIC_CANDIDATE.value: "#ff7f0e",  # orange
    DetectionMethod.SYMMETRY_CONTACT.value:    "#2ca02c",  # green
    DetectionMethod.UNKNOWN.value:             "#7f7f7f",  # grey
}


# ----------------------------------------------------------------------
# Summary card data
# ----------------------------------------------------------------------

def summary_cards(bonds: pd.DataFrame, n_fe: int) -> dict[str, str]:
    """Numbers for the metric strip at the top of the dashboard.

    Returns a dict of stringified values (already formatted), so the UI
    layer just hands them to st.metric without further math.
    """
    if bonds.empty:
        return {
            "n_fe":         str(n_fe),
            "n_distances":  "0",
            "mean_A":       "—",
            "min_A":        "—",
            "max_A":        "—",
        }
    d = bonds[COL_DISTANCE]
    return {
        "n_fe":         str(n_fe),
        "n_distances":  str(len(d)),
        "mean_A":       f"{d.mean():.3f}",
        "min_A":        f"{d.min():.3f}",
        "max_A":        f"{d.max():.3f}",
    }


# ----------------------------------------------------------------------
# Lollipop chart
# ----------------------------------------------------------------------

def lollipop(
    bonds: pd.DataFrame,
    *,
    show_spin_bands: bool = False,
    oxidation_state: str | None = None,
) -> go.Figure:
    """Lollipop chart: one stem per Fe–N distance, sorted by Fe then N.

    Each lollipop is colour-coded by detection_method. Optional reference
    bands are filtered to the user's oxidation state (or shown as
    combined LS / HS regions when oxidation is unknown) to avoid the
    overlapping-label problem you get when all four bands are plotted.
    """
    fig = go.Figure()

    if bonds.empty:
        fig.update_layout(
            title="No Fe–N distances to plot",
            xaxis_visible=False,
            yaxis_visible=False,
            height=360,
        )
        return fig

    # X axis = bond pair label "FeX – NY", in the order they sit in the
    # (already sorted) DataFrame.
    bonds = bonds.copy()
    bonds["pair_label"] = bonds[COL_FE_LABEL] + "–" + bonds[COL_N_LABEL]

    # Optional reference bands (drawn first so points sit on top).
    # `select_bands` returns 2 well-separated bands so labels never
    # overlap: either Fe(II)/Fe(III)-specific LS+HS, or generic
    # LS region / HS region when oxidation state is unknown.
    if show_spin_bands:
        for label, (lo, hi) in select_bands(oxidation_state).items():
            fig.add_shape(
                type="rect", xref="paper", yref="y",
                x0=0, x1=1, y0=lo, y1=hi,
                fillcolor="lightgrey", opacity=0.20,
                line_width=0, layer="below",
            )
            fig.add_annotation(
                xref="paper", yref="y",
                x=1.005, y=(lo + hi) / 2,
                text=label, showarrow=False,
                xanchor="left", font=dict(size=10, color="#666"),
            )

    # One trace per detection method so the legend is clean.
    for method, sub in bonds.groupby(COL_DETECTION_METHOD):
        # Stems
        for _, row in sub.iterrows():
            fig.add_shape(
                type="line",
                x0=row["pair_label"], x1=row["pair_label"],
                y0=0, y1=row[COL_DISTANCE],
                line=dict(color=METHOD_COLOURS.get(method, "#888"),
                          width=2),
            )
        # Heads
        hover = [
            f"<b>{r[COL_FE_LABEL]} – {r[COL_N_LABEL]}</b><br>"
            f"distance = {r[COL_DISTANCE]:.3f} Å<br>"
            f"method = {r[COL_DETECTION_METHOD]}"
            + (f"<br><i>{r[COL_WARNING]}</i>" if r[COL_WARNING] else "")
            for _, r in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["pair_label"], y=sub[COL_DISTANCE],
            mode="markers",
            name=method,
            marker=dict(size=14, color=METHOD_COLOURS.get(method, "#888"),
                        line=dict(color="white", width=1.5)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    fig.update_layout(
        title="Fe–N distances",
        xaxis_title="Fe–N pair",
        yaxis_title="distance (Å)",
        height=440,
        margin=dict(t=50, r=120 if show_spin_bands else 30, b=60, l=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        font=dict(size=13),
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(rangemode="tozero", gridcolor="#eee")
    return fig


# ----------------------------------------------------------------------
# Σ reference chart — only shown when the user annotates Fe(II)/Fe(III).
# ----------------------------------------------------------------------

def sigma_reference_chart(
    sigma_by_fe: dict[str, float],
    oxidation_state: str,
) -> go.Figure:
    """Show each Fe centre's Σ as a dot against shaded LS / HS bands
    for the chosen oxidation state.

    `sigma_by_fe`: {Fe_label: sigma_degrees}. None values are filtered.
    """
    fig = go.Figure()
    bands = select_sigma_bands(oxidation_state)
    if not bands:
        # Caller should avoid calling this in that case, but be safe.
        return fig

    valid = {k: v for k, v in sigma_by_fe.items() if v is not None}
    if not valid:
        fig.update_layout(
            title="Σ vs reference bands (no Σ values available)",
            height=200,
        )
        return fig

    # Vertical shaded bands: x covers a Σ range (data), y spans the
    # full plotting area (paper). Plotly rejects yref="x" — the right
    # combination for vertical strips is xref="x", yref="paper".
    band_colours = {"LS": "rgba(31,119,180,0.18)",   # blue
                    "HS": "rgba(214, 39, 40,0.18)"}  # red
    for label, (lo, hi) in bands.items():
        spin = "LS" if "LS" in label else "HS"
        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=lo, x1=hi, y0=0, y1=1,
            fillcolor=band_colours[spin], line_width=0, layer="below",
        )
        fig.add_annotation(
            xref="x", yref="paper",
            x=(lo + hi) / 2, y=1.04,
            text=label, showarrow=False,
            font=dict(size=11, color="#555"),
        )

    # Each Fe centre as a labelled dot.
    fig.add_trace(go.Scatter(
        x=list(valid.values()),
        y=list(valid.keys()),
        mode="markers+text",
        marker=dict(size=16, color="#333", symbol="diamond"),
        text=[f"{v:.1f}°" for v in valid.values()],
        textposition="middle right",
        hovertemplate="<b>%{y}</b><br>Σ = %{x:.2f}°<extra></extra>",
    ))

    fig.update_layout(
        title=f"Σ vs typical {oxidation_state} reference bands",
        xaxis_title="Σ (°)",
        yaxis=dict(title="", autorange="reversed"),
        height=120 + 40 * len(valid),
        margin=dict(t=40, r=80, b=40, l=80),
        showlegend=False,
        font=dict(size=12),
    )
    fig.update_xaxes(rangemode="tozero", gridcolor="#eee")
    return fig
