"""
Mode 2 — Batch / cohort analysis.

Drop many CIFs, get a single aggregated table with per-Fe-centre rows
(refcode, T, geometry, BVS, refinement quality, annotations) and four
linked Plotly scatter panels that surface the canonical SCO plots
(mean Fe–N vs T, Σ vs mean Fe–N, Θ vs Σ, BVS vs mean Fe–N).

Auto-grouping by `_chemical_formula_sum` lets users drop a T-series
of the same compound and see the points connected automatically. The
"series" column in the cohort table is user-editable for the cases
where the auto-detection gets it wrong (e.g. solvate differences).

Performance notes:
  - CIF parsing runs through @st.cache_data keyed on a SHA-256 of the
    file content (plus a schema-version constant), so re-uploads or
    edits don't re-parse.
  - Inside each cache miss we hand the file to a worker thread via
    concurrent.futures so 10 CIFs parse in roughly N/4 walltime
    instead of N.
  - The cohort table is rebuilt cheaply on every rerun from the
    cached FeSummary list; BVS recomputes per row on each annotation
    edit without re-parsing.
"""

from __future__ import annotations
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.cif_reader import read_cif
from src.fe_n_analysis import analyse
from src.cohort import (
    COHORT_COLUMNS, COHORT_EDITABLE,
    COL_BVS, COL_FE, COL_FILE, COL_FORMULA, COL_MEAN_D,
    COL_N_FEN, COL_OX, COL_R_FACTOR, COL_REFCODE, COL_SERIES,
    COL_SIGMA, COL_SPIN, COL_TEMP, COL_THETA, COL_ZETA,
    FeSummary,
    auto_assign_series,
    build_cohort_dataframe,
    make_error_summary,
    summaries_for_cif,
)
from src import annotations as ann


# Bump on any change to FeSummary or summaries_for_cif so the cache
# doesn't return stale shapes.
_BATCH_SCHEMA = "v1-batch"


# -----------------------------------------------------------------
# Single-file parse — the expensive step, cached.
# -----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _parse_one_file(
    content_hash: str,
    file_bytes: bytes,
    filename: str,
    schema_version: str = _BATCH_SCHEMA,
) -> list[dict]:
    """Parse one CIF and return a list of FeSummary dicts.

    Returns dicts (not FeSummary objects) because @st.cache_data
    pickles its output and pickling a list of dataclass instances
    works but plain dicts round-trip more reliably across schema
    bumps.

    A parse failure is surfaced as a single error-summary row so
    the user can see *which* file misbehaved.
    """
    del content_hash, schema_version          # part of the cache key only
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        try:
            bundle = read_cif(tmp_path)
            result = analyse(
                bundle, cif_filename=filename, min_occupancy=0.5,
            )
            summaries = summaries_for_cif(filename, bundle, result)
            if not summaries:
                summaries = [make_error_summary(
                    filename, "No N-coordinated Fe centre in this CIF."
                )]
            return [asdict(s) for s in summaries]
        except Exception as exc:                # noqa: BLE001
            return [asdict(make_error_summary(filename, repr(exc)))]
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse_all_in_parallel(uploads) -> list[FeSummary]:
    """Parse every uploaded CIF, reusing the @st.cache_data store.

    ThreadPoolExecutor only helps for cache misses — already-cached
    files come back in microseconds via the @st.cache_data decorator.
    We thread the misses with at most 4 workers (ccdc has process-
    level state we don't want to stress) and show progress.
    """
    items = [(f.name, f.getvalue()) for f in uploads]
    n = len(items)
    progress = st.progress(0.0, text=f"Parsing 0/{n} CIFs…")

    summaries: list[FeSummary] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=min(4, max(1, n))) as ex:
        futures = {
            ex.submit(_parse_one_file, _hash_bytes(b), b, name): name
            for name, b in items
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                dicts = fut.result()
            except Exception as exc:            # noqa: BLE001
                dicts = [asdict(make_error_summary(name, repr(exc)))]
            for d in dicts:
                summaries.append(FeSummary(**d))
            completed += 1
            progress.progress(
                completed / n,
                text=f"Parsing {completed}/{n} CIFs…",
            )
    progress.empty()
    return summaries


# -----------------------------------------------------------------
# Annotation merge from session-state
# -----------------------------------------------------------------
def _annotations_for(filenames: list[str]) -> dict[str, dict[str, str]]:
    """Pull the persisted annotation dict for each uploaded filename.

    Mode 1 may have set oxidation / spin earlier in the same session —
    we honour those as defaults here. The cohort editor overrides them
    when the user makes per-row changes.
    """
    out: dict[str, dict[str, str]] = {}
    for name in filenames:
        saved = ann.get(name)
        if saved:
            out[name] = {
                "oxidation_state": saved.get("oxidation_state", "(unknown)"),
                "spin_state":      saved.get("spin_state",      "(unknown)"),
            }
    return out


# -----------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------
def _empty_plot(title: str, hint: str | None = None) -> go.Figure:
    """A placeholder figure with a useful in-chart hint when no data
    is available for the panel (e.g. BVS before any oxidation state
    has been annotated)."""
    fig = go.Figure()
    if hint:
        fig.add_annotation(
            text=hint, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#888"),
        )
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(t=46, b=40, l=50, r=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _scatter(df: pd.DataFrame, x: str, y: str, title: str,
             color: str | None = None, line_group: str | None = None,
             empty_hint: str | None = None,
             ) -> go.Figure:
    """Standard hoverable scatter; optionally connect by series.

    When the panel is empty (no rows with both x and y populated),
    we render a placeholder with `empty_hint` — typically a one-line
    instruction explaining how to make the panel appear (e.g.
    "Set oxidation states in the table above").
    """
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return _empty_plot(f"{title} — no data", empty_hint)

    hover_cols = [c for c in (
        COL_FILE, COL_REFCODE, COL_FE, COL_N_FEN,
        COL_TEMP, COL_MEAN_D, COL_SIGMA, COL_THETA, COL_BVS,
        COL_OX, COL_SPIN, COL_R_FACTOR,
    ) if c in plot_df.columns]

    if line_group and line_group in plot_df.columns:
        plot_df = plot_df.sort_values([line_group, x])
        fig = px.line(
            plot_df, x=x, y=y, color=color or line_group,
            line_group=line_group, markers=True,
            hover_data=hover_cols, title=title,
        )
    else:
        fig = px.scatter(
            plot_df, x=x, y=y, color=color,
            hover_data=hover_cols, title=title,
        )

    fig.update_layout(
        height=360,
        # Extra right margin reserves space for the vertical legend
        # so it can't collide with the x-axis label when Streamlit
        # compresses the column horizontally.
        margin=dict(t=46, b=44, l=60, r=120),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            font=dict(size=10),
            title_font=dict(size=11),
        ),
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig


def _render_scatter_grid(df: pd.DataFrame) -> None:
    """Four canonical SCO panels in a 2 × 2 layout."""
    # Colour by spin if at least one row has a real annotation,
    # otherwise by oxidation. Falls back to series if neither helps.
    has_spin = (df[COL_SPIN].astype(str) != "(unknown)").any()
    has_ox   = (df[COL_OX].astype(str)   != "(unknown)").any()
    primary_colour = COL_SPIN if has_spin else (COL_OX if has_ox else COL_SERIES)

    row1c1, row1c2 = st.columns(2)
    with row1c1:
        st.plotly_chart(
            _scatter(
                df, x=COL_TEMP, y=COL_MEAN_D,
                color=primary_colour, line_group=COL_SERIES,
                title="⟨Fe–N⟩ vs T (K) — connects same-formula series",
            ),
            use_container_width=True,
        )
    with row1c2:
        st.plotly_chart(
            _scatter(
                df, x=COL_MEAN_D, y=COL_SIGMA, color=primary_colour,
                title="Σ vs ⟨Fe–N⟩ — the classical LS/HS structural shift",
            ),
            use_container_width=True,
        )

    row2c1, row2c2 = st.columns(2)
    with row2c1:
        st.plotly_chart(
            _scatter(
                df, x=COL_SIGMA, y=COL_THETA, color=primary_colour,
                title="Θ vs Σ — does elongation correlate with twist?",
            ),
            use_container_width=True,
        )
    with row2c2:
        st.plotly_chart(
            _scatter(
                df, x=COL_MEAN_D, y=COL_BVS, color=primary_colour,
                title="BVS vs ⟨Fe–N⟩ — cohort sanity check",
                empty_hint=(
                    "Set <b>oxidation_state</b> for at least one row<br>"
                    "in the table above (Fe(II) or Fe(III))<br>"
                    "to populate BVS values."
                ),
            ),
            use_container_width=True,
        )


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------
def render() -> None:
    st.title("Batch / cohort analysis")
    st.caption(
        "Drop many CIFs — typically the same complex measured at "
        "different temperatures, or a small structural series. "
        "Each Fe centre becomes one row; auto-grouping by chemical "
        "formula colours / connects them on the SCO plots below."
    )

    st.sidebar.divider()
    st.sidebar.header("Upload")
    uploads = st.sidebar.file_uploader(
        "Drop multiple .cif files",
        type=["cif"],
        accept_multiple_files=True,
        help="Each file is parsed once and cached on its SHA-256 "
             "content hash — re-uploading the same CIF in a later "
             "session reuses the parsed result.",
    )

    if not uploads:
        st.info(
            "Drop one or more `.cif` files in the sidebar to begin.\n\n"
            "**What you'll get:** a cohort table with one row per Fe "
            "centre across every uploaded CIF, four interactive "
            "scatter plots (⟨Fe–N⟩ vs T, Σ vs ⟨Fe–N⟩, Θ vs Σ, BVS "
            "vs ⟨Fe–N⟩), bulk annotation editing, and a CSV download."
        )
        return

    # ---- Parse everything (cached per-file by SHA-256) ----------
    summaries = _parse_all_in_parallel(uploads)
    if not summaries:
        st.warning("No data extracted from any uploaded file.")
        return

    # ---- Pull existing annotations from session-state ----------
    initial_annotations = _annotations_for(
        [f.name for f in uploads]
    )

    # ---- Build the cohort DataFrame + auto-assign series -------
    df = build_cohort_dataframe(
        summaries, per_file_annotations=initial_annotations,
    )
    df = auto_assign_series(df)

    # ---- Headline counts ---------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CIFs uploaded", len(uploads))
    col2.metric("Fe centres",    len(df))
    col3.metric("series detected",
                df[COL_SERIES].replace({"?": None}).nunique(dropna=True))
    n_errors = int(df[COL_FORMULA].astype(str)
                   .str.startswith("PARSE FAILED").sum())
    col4.metric("parse failures", n_errors)

    # ---- Editable cohort table ---------------------------------
    st.subheader("Cohort table")
    st.caption(
        "Click cells in **oxidation_state / spin_state / series** to "
        "edit. Changes update the BVS column and the scatter plots "
        "below. Per-file annotations also flow back to Mode 1."
    )

    # ---- Auto-fill button ---------------------------------------
    # Runs infer_oxidation_spin() on each row's formal-bond distances
    # and writes the recommended (oxidation, spin) into the session
    # annotation store keyed by filename. The data_editor then picks
    # those up as defaults on the next rerun.
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button(
            "Auto-fill from bond geometry",
            help="For each row, computes BVS at all four (Fe(II)/Fe(III)) "
                 "× (LS/HS) combos and picks the bond-length-consistent "
                 "candidate with the smallest |BVS−Z|. You can still "
                 "edit any row manually afterwards.",
            use_container_width=True,
        ):
            from src.bvs import infer_oxidation_spin as _infer
            applied = 0
            skipped = 0
            for s in summaries:
                if not s.distances_A:
                    skipped += 1
                    continue
                result = _infer(s.distances_A)
                if result is None or result.best_oxidation is None:
                    skipped += 1
                    continue
                ann.update(
                    s.filename,
                    content_hash=ann.hash_bytes(b""),
                    oxidation_state=result.best_oxidation,
                    spin_state=result.best_spin or "(unknown)",
                )
                applied += 1
            st.session_state["batch_autofill_applied"] = applied
            st.session_state["batch_autofill_skipped"] = skipped
            st.rerun()
    with col_b:
        applied = st.session_state.get("batch_autofill_applied")
        skipped = st.session_state.get("batch_autofill_skipped")
        if applied is not None:
            note = (
                f"_Auto-fill applied to **{applied}** Fe centre(s); "
                f"{skipped} skipped (no formal Fe–N bonds). Recommendations "
                "are heuristics — review and edit any row manually. "
                "Watch for the porphyrin caveat in Mode 1 if a structure "
                "lands at Fe(III) but the bond length looks LS Fe(II)-like._"
            )
            st.caption(note)

    edited = st.data_editor(
        df,
        column_config={
            COL_OX: st.column_config.SelectboxColumn(
                COL_OX,
                options=["(unknown)", "Fe(0)", "Fe(II)", "Fe(III)", "Fe(IV)"],
            ),
            COL_SPIN: st.column_config.SelectboxColumn(
                COL_SPIN,
                options=["(unknown)", "LS", "IS", "HS"],
            ),
            COL_SERIES: st.column_config.TextColumn(
                COL_SERIES,
                help="Auto-assigned by chemical formula. Override for "
                     "solvate-only differences, etc.",
            ),
        },
        disabled=[c for c in COHORT_COLUMNS if c not in COHORT_EDITABLE],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="cohort_editor",
    )

    # ---- Sync edits back to the annotation store + recompute BVS
    # The data_editor returns plain strings. We sync per-row to
    # session-state by filename (so re-opens of Mode 1 see them too)
    # and recompute BVS in the displayed DataFrame.
    from src.bvs import compute_bvs
    by_filename: dict[str, FeSummary] = {s.filename: s for s in summaries}
    new_bvs: list[float | None] = []
    for _, row in edited.iterrows():
        fname = row[COL_FILE]
        ox = row[COL_OX]
        spin = row[COL_SPIN]
        s = by_filename.get(fname)
        if s is None or not s.distances_A or ox in ("(unknown)", None, ""):
            new_bvs.append(None)
            continue
        spin_in = spin if spin in ("LS", "HS") else None
        res = compute_bvs(s.distances_A, ox, spin_in)
        new_bvs.append(round(res.bvs, 2) if res is not None else None)
        # Persist to the session-state annotation store so Mode 1
        # picks up the changes when the user re-uploads the file.
        if ox != "(unknown)" or spin != "(unknown)":
            ann.update(
                fname,
                content_hash=ann.hash_bytes(b""),
                oxidation_state=ox,
                spin_state=spin,
            )
    edited[COL_BVS] = new_bvs

    # ---- Scatter plots -----------------------------------------
    st.subheader("Scatter panels")
    _render_scatter_grid(edited)

    # ---- CSV download ------------------------------------------
    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download cohort CSV",
        data=csv_bytes,
        file_name="cohort.csv",
        mime="text/csv",
    )
