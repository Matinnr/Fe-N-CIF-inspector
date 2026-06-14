"""
Single-CIF mode — inspect one structure at a time.

This is the original dashboard, moved out of app.py during the
multi-mode refactor. `render()` is the entry point app.py calls.

What changed vs. the pre-refactor app.py:
  - The sidebar now reads annotations from src.annotations (session_state)
    so they survive mode switches and re-uploads.
  - The CCDC-import-before-matplotlib note no longer applies; we never
    import matplotlib here.
  - Detection-mode defaults are unchanged at the function level
    (include_geometric=False, include_symmetry=False in analyse());
    the sidebar checkboxes now mirror those defaults.
"""

from __future__ import annotations
import tempfile
from pathlib import Path

import streamlit as st

# Project imports — must come AFTER streamlit on macOS (libcrypto load order).
from src.cif_reader import read_cif
from src.fe_n_analysis import analyse, AnalysisResult
from src.plotting import lollipop, summary_cards, sigma_reference_chart
from src.data_schema import (
    COL_DETECTION_METHOD, COL_DISTANCE, COL_DISTANCE_ESD, COL_FE_LABEL,
    suggest_spin_state,
)
from src.esd import format_with_esd, propagate_mean_esd
from src.bvs import (
    R0_LITERATURE, choose_R0, compute_bvs, consistency_status,
    has_literature_R0, infer_oxidation_spin, probe_both_oxidation_states,
)
from src.data_schema import DetectionMethod
from src import annotations as ann


# Stable widget option lists — single source of truth for the
# selectbox values that we also use as annotation defaults.
OXIDATION_OPTIONS = ["(unknown)", "Fe(0)", "Fe(II)", "Fe(III)", "Fe(IV)"]
SPIN_OPTIONS      = ["(unknown)", "LS", "IS", "HS"]


# -----------------------------------------------------------------
# Cached parser — keyed on (bytes, name, _SCHEMA_VERSION).
#
# Why the schema version: @st.cache_resource lives across hot-reloads
# of source files. If we add a new field to CifBundle (Provenance,
# bond_esds, disorder_atoms — all added since this code was first
# written), the cache will silently keep returning old bundles that
# lack the new attribute, and the new render code will hit
# AttributeError. Bumping _SCHEMA_VERSION whenever CifBundle gains
# or loses a field forces a fresh parse on next upload.
# -----------------------------------------------------------------
_SCHEMA_VERSION = "v3-with-provenance"


@st.cache_resource(show_spinner="Parsing CIF…")
def _parse(cif_bytes: bytes, original_name: str, schema_version: str = _SCHEMA_VERSION):
    """Save bytes to a temp file, parse, return (bundle, name).

    @st.cache_resource (not cache_data) because CifBundle holds CCDC
    C++ objects that can't be pickled. The schema_version is part of
    the cache key so any change to CifBundle invalidates old entries.
    """
    del schema_version  # only here to alter the cache key
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(cif_bytes)
        tmp_path = Path(tmp.name)
    try:
        bundle = read_cif(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return bundle, original_name


# -----------------------------------------------------------------
# Sidebar — annotations + detection controls
# -----------------------------------------------------------------
def _sidebar(filename: str | None) -> dict:
    """Render the mode-specific sidebar widgets, return user inputs."""
    # ---- Annotations (oxidation/spin) ----------------------------
    saved = ann.get(filename) if filename else {}
    st.sidebar.divider()
    st.sidebar.header("Annotations (optional)")

    ox_default = saved.get("oxidation_state", OXIDATION_OPTIONS[0])
    user_oxidation = st.sidebar.selectbox(
        "Oxidation state",
        options=OXIDATION_OPTIONS,
        index=OXIDATION_OPTIONS.index(ox_default)
              if ox_default in OXIDATION_OPTIONS else 0,
        help="Recorded for export only — does not affect the analysis. "
             "Saved per file across mode switches.",
    )

    spin_default = saved.get("spin_state", SPIN_OPTIONS[0])
    user_spin = st.sidebar.selectbox(
        "Spin state",
        options=SPIN_OPTIONS,
        index=SPIN_OPTIONS.index(spin_default)
              if spin_default in SPIN_OPTIONS else 0,
    )

    # ---- Detection settings --------------------------------------
    st.sidebar.divider()
    st.sidebar.header("Detection settings")
    cutoff_A = st.sidebar.slider(
        "Distance cutoff (Å)", 1.5, 3.5, 2.7, 0.05,
        help="Fe–N pairs farther than this are ignored.",
    )
    include_geometric = st.sidebar.checkbox(
        "Include geometric candidates",
        value=False,
        help="Pairs within cutoff that the chemistry algorithm does NOT "
             "classify as bonds. Enable for unusual long Fe–N "
             "coordinations; can muddy Σ/Θ if non-coordinating N is "
             "picked up.",
    )
    include_symmetry = st.sidebar.checkbox(
        "Include symmetry-generated contacts",
        value=False,
        help="Catch Fe–N pairs that span asymmetric units. Enable for "
             "polymeric / bridging structures.",
    )
    show_spin_bands = st.sidebar.checkbox(
        "Show LS/HS reference bands on chart",
        value=False,
        help="Overlay typical Fe(II)/Fe(III) LS/HS bond-length ranges.",
    )

    st.sidebar.divider()
    st.sidebar.header("Disorder handling")
    disorder_mode = st.sidebar.radio(
        "How to treat partial-occupancy atoms",
        options=[
            "Use major component only",
            "Use all components",
            "Show both side-by-side",
        ],
        index=0,
        help="'Major component only' (default) drops atoms whose "
             "site occupancy is below 0.5 — the conventional "
             "treatment for crystallographic disorder. 'All "
             "components' includes every refined position. 'Side "
             "by side' computes the mean Fe–N under both treatments "
             "so you can see how disorder affects it.",
    )

    return {
        "oxidation":     user_oxidation,
        "spin":          user_spin,
        "cutoff_A":      cutoff_A,
        "geometric":     include_geometric,
        "symmetry":      include_symmetry,
        "spin_bands":    show_spin_bands,
        "disorder_mode": disorder_mode,
    }


# -----------------------------------------------------------------
# Mode entry point
# -----------------------------------------------------------------
def render() -> None:
    """Draw the single-CIF mode UI."""
    st.title("Single-CIF Fe–N inspector")
    st.caption(
        "One CIF in: every Fe–N distance with esds, octahedral distortion "
        "(ζ, Δ, Σ, Θ), bond-valence sum, refinement quality, and "
        "provenance. The dashboard *suggests* oxidation and spin states "
        "from BVS and bond-length heuristics; your sidebar annotations "
        "override the suggestions when set."
    )

    # ---- Upload widget at the top of the sidebar -----------------
    # (Mode switcher above it is drawn by app.py.)
    st.sidebar.divider()
    st.sidebar.header("Upload")
    uploaded = st.sidebar.file_uploader(
        "Drop a single .cif file",
        type=["cif"],
        help="Only one CIF at a time. The file is parsed once and "
             "cached; sliders re-analyse without re-parsing.",
    )

    # Sidebar widgets that depend on / influence annotations.
    settings = _sidebar(uploaded.name if uploaded is not None else None)

    # ---- No upload yet — show placeholder and stop ---------------
    if uploaded is None:
        st.info(
            "Upload a `.cif` file in the sidebar to begin.\n\n"
            "**What you'll see:** five summary cards (Fe centres / Fe–N "
            "count / mean / min / max), a clean per-bond table, a "
            "hoverable lollipop chart, and a CSV download."
        )
        return

    # ---- Persist annotations BEFORE running analysis -------------
    # The user may have just changed oxidation/spin in the sidebar;
    # record both, plus the SHA-256 content hash so we can detect
    # 'same file uploaded under a different name' downstream.
    cif_bytes = uploaded.getvalue()
    content_hash = ann.hash_bytes(cif_bytes)
    ann.update(
        uploaded.name,
        content_hash=content_hash,
        oxidation_state=settings["oxidation"],
        spin_state=settings["spin"],
    )

    # Rename-detection: did the user upload this same content earlier
    # under a different filename?
    twin = ann.find_by_hash(content_hash, exclude_filename=uploaded.name)
    if twin is not None:
        st.warning(
            f"⚠️ This file's content matches a previously-uploaded "
            f"`{twin}`. Annotations on the two filenames are tracked "
            f"separately — change names to disambiguate if needed."
        )

    # ---- Parse (cached) ------------------------------------------
    try:
        bundle, original_name = _parse(cif_bytes, uploaded.name)
    except RuntimeError as exc:
        st.error("Could not import the CCDC Python API.")
        st.code(str(exc))
        return
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Failed to read CIF: {exc}")
        return

    # ---- Analyse (NOT cached — depends on sliders) ---------------
    # Map the user's disorder-mode choice to a min_occupancy threshold.
    # 'Use all components' (and the unfiltered side of side-by-side) =
    # 0.0; everything else = 0.5 (the conventional "major component
    # only" rule). The side-by-side mode runs the analyser twice and
    # surfaces both means below.
    primary_min_occ = (
        0.0 if settings["disorder_mode"] == "Use all components" else 0.5
    )

    result: AnalysisResult = analyse(
        bundle,
        cif_filename=original_name,
        cutoff_A=settings["cutoff_A"],
        include_geometric=settings["geometric"],
        include_symmetry=settings["symmetry"],
        min_occupancy=primary_min_occ,
    )

    # Side-by-side comparison: run the analyser a second time with the
    # opposite filter so we can report the alternative mean. We don't
    # render its full panel — only the headline mean Fe–N — because the
    # point is to show how disorder affects that single number.
    side_by_side_result: AnalysisResult | None = None
    if settings["disorder_mode"] == "Show both side-by-side":
        side_by_side_result = analyse(
            bundle,
            cif_filename=original_name,
            cutoff_A=settings["cutoff_A"],
            include_geometric=settings["geometric"],
            include_symmetry=settings["symmetry"],
            min_occupancy=0.0,   # the 'all components' counterpart
        )

    # Persist the analysis result to the session-wide store so Mode 3
    # can render this CIF as a comparison card without re-upload.
    # The store keys by filename; re-analysing the same upload
    # refreshes the entry in place. Stored fields are pickleable —
    # no ccdc objects retained.
    #
    # Failures used to be silently swallowed here, which made it
    # impossible to tell whether Mode 3 wasn't seeing a CIF because
    # the record never ran. We now show the error in the UI so the
    # user knows when something went wrong, while still preventing
    # the failure from taking down the rest of Mode 1.
    from src import analysed_store as _astore
    try:
        _astore.record_from_analysis(
            uploaded.name, cif_bytes, bundle, result,
        )
    except Exception as exc:                              # noqa: BLE001
        st.warning(
            f"⚠️ Could not register `{uploaded.name}` for cross-mode "
            f"comparison: `{exc!r}`. The Mode 1 analysis still works; "
            "the only consequence is that this CIF won't appear "
            "automatically in Mode 3."
        )
    else:
        # Tiny non-modal confirmation so the user can see that the
        # cross-mode link is alive. `st.toast` is unobtrusive — top
        # right of the page, auto-dismisses.
        try:
            st.toast(
                f"Registered `{uploaded.name}` for cross-mode "
                "comparison — visible in Reference library mode.",
                icon=None,
            )
        except Exception:                                # noqa: BLE001
            pass

    # Decorate output with user annotations (for CSV export).
    df = result.bonds.copy()
    if not df.empty:
        df["user_oxidation_state"] = settings["oxidation"]
        df["user_spin_state"]      = settings["spin"]

    # ---- Summary cards -------------------------------------------
    # The mean is rendered with a propagated esd (σ_mean = √Σσᵢ²/n)
    # when every formal bond carries an esd; otherwise we fall back
    # to the bare numeric.
    cards = summary_cards(result.bonds, n_fe=result.n_coord_fe)
    mean_display = cards["mean_A"]
    if not result.bonds.empty:
        formal = result.bonds[
            result.bonds[COL_DETECTION_METHOD]
            == DetectionMethod.FORMAL_BOND.value
        ]
        if not formal.empty:
            mean_value = float(formal[COL_DISTANCE].mean())
            esds = formal[COL_DISTANCE_ESD].tolist()
            mean_esd = propagate_mean_esd(esds) if esds else None
            mean_display = format_with_esd(mean_value, mean_esd)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fe centres",      cards["n_fe"])
    c2.metric("Fe–N distances",  cards["n_distances"])
    c3.metric("mean (Å)",        mean_display)
    c4.metric("min (Å)",         cards["min_A"])
    c5.metric("max (Å)",         cards["max_A"])

    caption_parts = [f"Structure: **{bundle.structure_id}**"]
    if bundle.temperature_K is not None:
        caption_parts.append(f"T = **{bundle.temperature_K:.1f} K**")
    else:
        caption_parts.append("T = unknown")
    if result.excluded_fe_labels:
        caption_parts.append(
            f"Excluded {len(result.excluded_fe_labels)} non-N Fe centre(s): "
            f"`{', '.join(result.excluded_fe_labels)}`"
        )
    st.caption(" · ".join(caption_parts))

    # ---- Provenance + refinement-quality badge -------------------
    _render_provenance(bundle)

    # ---- Disorder panel + side-by-side mean comparison ----------
    _render_disorder_panel(
        bundle, result, side_by_side_result, settings["disorder_mode"],
    )

    # ---- Spin-state heuristic hint + BVS cross-check -------------
    _render_spin_and_bvs(
        result=result,
        oxidation=settings["oxidation"],
        spin=settings["spin"],
        filename=uploaded.name,
    )

    # ---- Warnings ------------------------------------------------
    if result.warnings:
        with st.expander(f"Notes ({len(result.warnings)})", expanded=True):
            for w in result.warnings:
                st.write(f"- {w}")

    # ---- Bond table + CSV ----------------------------------------
    st.subheader("Fe–N distances")
    if df.empty:
        st.warning(
            "No Fe–N distances to display — see notes above. "
            "Try widening the cutoff or enabling geometric candidates."
        )
    else:
        # Build a display version with crystallographic-style esd
        # notation ("1.984(7)"). The numeric columns stay in the CSV
        # export so machine consumers still get raw numbers.
        display = _build_display_table(df)
        st.dataframe(display, use_container_width=True, hide_index=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            data=csv_bytes,
            file_name=f"{bundle.structure_id}_fen.csv",
            mime="text/csv",
        )

    # ---- Lollipop chart ------------------------------------------
    st.subheader("Distance chart")
    fig = lollipop(
        result.bonds,
        show_spin_bands=settings["spin_bands"],
        oxidation_state=settings["oxidation"]
                        if settings["oxidation"] != "(unknown)" else None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Geometry panel ------------------------------------------
    _render_geometry(result, settings["oxidation"])

    # ---- Per-Fe summary ------------------------------------------
    if result.n_fe > 1 and not result.bonds.empty:
        st.subheader("Per-Fe-centre summary")
        per_fe = (
            result.bonds
            .groupby(COL_FE_LABEL)[COL_DISTANCE]
            .agg(["count", "mean", "std", "min", "max"])
            .round(3)
            .rename(columns={"count": "n", "mean": "mean (Å)",
                             "std": "std (Å)", "min": "min (Å)",
                             "max": "max (Å)"})
            .reset_index()
        )
        st.dataframe(per_fe, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------
# Provenance — bibliographic + refinement-quality panel
# -----------------------------------------------------------------
def _refinement_quality_badge(
    R_factor_gt: float | None,
    n_reflns: int | None,
    n_parameters: int | None,
) -> tuple[str, str]:
    """Classify refinement quality. Returns (level, message).

    Rules (rules of thumb only):
      green   "Good"      R < 5%  AND  data/parameter > 8
      red     "Check"     R > 10%
      yellow  "Marginal"  anything else (default, including missing R)
    """
    if R_factor_gt is None:
        return "marginal", (
            "No R-factor reported in CIF — unable to assess "
            "refinement quality."
        )
    R_pct = R_factor_gt * 100.0
    dp_ratio: float | None = None
    if n_reflns is not None and n_parameters not in (None, 0):
        dp_ratio = n_reflns / n_parameters
    dp_part = (f"data/parameter = {dp_ratio:.1f}"
               if dp_ratio is not None
               else "data/parameter unknown")
    if R_pct > 10.0:
        return "warning", (
            f"Check refinement — R = {R_pct:.2f}% > 10% "
            f"({dp_part})."
        )
    if R_pct < 5.0 and (dp_ratio is None or dp_ratio > 8.0):
        return "good", (
            f"Good refinement — R = {R_pct:.2f}% < 5%, {dp_part}."
        )
    return "marginal", (
        f"Marginal refinement — R = {R_pct:.2f}%, {dp_part}. "
        "Often acceptable for structures with extensive solvent "
        "disorder; check that the local Fe–N coordination is "
        "fully refined (small esds on bonds is a good sign)."
    )


def _doi_link(doi: str) -> str:
    """Markdown DOI link with URL-safe encoding."""
    # The most common offender is the trailing '+' in older Inorg.
    # Chem. DOIs (e.g. 10.1021/ic049581+).
    safe = doi.replace("+", "%2B")
    return f"[**{doi}**](https://doi.org/{safe})"


def _render_provenance(bundle) -> None:
    """Expander showing CCDC / refinement / publication metadata.

    Sits above the distance chart. Auto-collapsed because it's
    reference info, not headline — but it's the first place a
    crystallographer looks to gauge whether the rest is trustworthy.
    """
    # `getattr` keeps us safe against a CifBundle that was cached
    # before the `provenance` field was added to the dataclass —
    # @st.cache_resource hangs onto old instances across hot-reloads.
    prov = getattr(bundle, "provenance", None)
    if prov is None:
        return

    # Header preview puts the most useful single line up at the
    # expander level so the user can decide whether to open it.
    header_bits: list[str] = []
    if prov.ccdc_refcode:
        header_bits.append(prov.ccdc_refcode)
    if prov.chemical_formula:
        header_bits.append(prov.chemical_formula)
    if prov.R_factor_gt is not None:
        header_bits.append(f"R = {prov.R_factor_gt*100:.2f}%")
    title = "Provenance & refinement quality"
    if header_bits:
        title += "  ·  " + "  ·  ".join(header_bits)

    with st.expander(title, expanded=False):
        col_id, col_ref = st.columns(2)

        # --- Identification side --------------------------------
        with col_id:
            st.markdown("**Identification**")
            st.markdown(
                f"- CCDC refcode: **{prov.ccdc_refcode or '—'}**"
            )
            st.markdown(
                f"- Formula: **{prov.chemical_formula or '—'}**"
            )
            if prov.doi:
                st.markdown(f"- DOI: {_doi_link(prov.doi)}")
            else:
                st.markdown("- DOI: **—**")
            st.markdown(
                f"- Space group: **{prov.space_group or '—'}**"
            )
            z_str = str(prov.Z) if prov.Z is not None else "—"
            zp_str = (f"{prov.Z_prime:.2f}"
                      if prov.Z_prime is not None else "—")
            st.markdown(f"- Z / Z′: **{z_str} / {zp_str}**")
            t_str = (f"{bundle.temperature_K:.1f} K"
                     if bundle.temperature_K is not None else "—")
            st.markdown(f"- Temperature: **{t_str}**")

        # --- Refinement side ------------------------------------
        with col_ref:
            st.markdown("**Refinement**")
            def _pct(v):
                return f"{v*100:.2f}%" if v is not None else "—"

            r_line = f"- R (I > 2σ): **{_pct(prov.R_factor_gt)}**"
            if prov.R_factor_all is not None:
                r_line += f"  (R(all) = {_pct(prov.R_factor_all)})"
            st.markdown(r_line)
            st.markdown(f"- wR₂: **{_pct(prov.wR_factor_ref)}**")
            st.markdown(
                f"- GooF: **"
                f"{prov.goodness_of_fit:.3f}"
                "**" if prov.goodness_of_fit is not None
                else "- GooF: **—**"
            )
            dp_ratio: float | None = None
            if prov.n_reflns is not None and prov.n_parameters not in (None, 0):
                dp_ratio = prov.n_reflns / prov.n_parameters
            dp_str = f"{dp_ratio:.1f}" if dp_ratio is not None else "—"
            counts = (
                f"{prov.n_reflns or '?'} reflns / "
                f"{prov.n_parameters or '?'} parameters"
            )
            st.markdown(
                f"- Data / parameter: **{dp_str}**  ({counts})"
            )

        st.divider()

        # --- Quality badge --------------------------------------
        level, msg = _refinement_quality_badge(
            prov.R_factor_gt, prov.n_reflns, prov.n_parameters,
        )
        if level == "good":
            st.success(f"✅ {msg}")
        elif level == "warning":
            st.error(f"🚫 {msg}")
        else:
            st.warning(f"⚠️ {msg}")
        st.caption(
            "_Rules of thumb: **green** if R < 5% AND data/parameter "
            "> 8; **red** if R > 10%; **yellow** otherwise. "
            "These are guidelines, not absolute standards — "
            "heavily-disordered structures (e.g. lots of crystallisation "
            "solvent) often have higher R-factors without the local "
            "coordination geometry being unreliable. Always cross-check "
            "the bond-level esds._"
        )


# -----------------------------------------------------------------
# Disorder panel — list affected atoms, optional side-by-side mean
# -----------------------------------------------------------------
def _render_disorder_panel(
    bundle,
    result: AnalysisResult,
    side_by_side_result: AnalysisResult | None,
    disorder_mode: str,
) -> None:
    """Disorder atoms expander + (optional) side-by-side mean panel.

    Layout:
      1. If the CIF has any disorder, render an expander listing each
         partial-occupancy atom with its occupancy, assembly, group.
         Auto-collapsed — it's reference info, not headline.
      2. If `disorder_mode == 'Show both side-by-side'`, render a
         small comparison panel reporting the mean Fe–N under
         'major component only' (the primary result here) vs
         'all components' (side_by_side_result).
    """
    import pandas as pd

    # --- 1) Disorder-atoms expander -------------------------------
    if bundle.disorder_atoms:
        with st.expander(
            f"Disorder ({len(bundle.disorder_atoms)} "
            f"partial-occupancy atom(s))",
            expanded=False,
        ):
            df = pd.DataFrame([
                {
                    "atom":       a.label,
                    "occupancy":  round(a.occupancy, 3),
                    "assembly":   a.disorder_assembly or "—",
                    "group":      a.disorder_group or "—",
                }
                for a in bundle.disorder_atoms
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(
                "_Disorder assembly = the group of competing partial-"
                "occupancy models (`A`, `B`, …). Disorder group = the "
                "specific component within an assembly (`1` for major, "
                "`2` for minor, etc.). Dashes mean the CIF didn't "
                "record those tags._"
            )

    # --- 2) Side-by-side comparison -------------------------------
    if disorder_mode != "Show both side-by-side":
        return
    if side_by_side_result is None:
        return

    def _formal_mean_with_esd(r: AnalysisResult) -> tuple[str, int]:
        if r.bonds.empty:
            return "—", 0
        formal = r.bonds[
            r.bonds[COL_DETECTION_METHOD]
            == DetectionMethod.FORMAL_BOND.value
        ]
        if formal.empty:
            return "—", 0
        m = float(formal[COL_DISTANCE].mean())
        esds = formal[COL_DISTANCE_ESD].tolist()
        m_esd = propagate_mean_esd(esds) if esds else None
        return format_with_esd(m, m_esd), len(formal)

    primary_str, primary_n = _formal_mean_with_esd(result)
    other_str,   other_n   = _formal_mean_with_esd(side_by_side_result)

    st.subheader("Disorder: side-by-side mean Fe–N")
    c1, c2 = st.columns(2)
    c1.metric(
        f"Major component only (occ ≥ 0.5)  ·  n = {primary_n}",
        primary_str,
    )
    c2.metric(
        f"All components (occ > 0)  ·  n = {other_n}",
        other_str,
    )
    st.caption(
        "_Same CIF, two disorder treatments. The bond table, "
        "geometry metrics, and BVS shown below all use the "
        "**major-component-only** result; only the means above "
        "compare both._"
    )


# -----------------------------------------------------------------
# Spin-state heuristic + BVS oxidation-state cross-check
# -----------------------------------------------------------------
_STATUS_ICON = {"good": "✅", "caution": "⚠️", "warning": "🚫"}


def _formal_bonds_by_fe(result: AnalysisResult) -> dict[str, list[float]]:
    """Distances grouped by Fe label, using only formal bonds.

    BVS, like the geometry metrics, should be computed over the genuine
    coordination sphere — not over distance-cutoff or symmetry-generated
    candidates that may pull in non-coordinating N atoms.
    """
    if result.bonds.empty:
        return {}
    formal_mask = result.bonds[COL_DETECTION_METHOD] == DetectionMethod.FORMAL_BOND.value
    formal = result.bonds[formal_mask]
    if formal.empty:
        return {}
    return (
        formal.groupby(COL_FE_LABEL)[COL_DISTANCE]
              .apply(list)
              .to_dict()
    )


def _render_spin_and_bvs(*, result: AnalysisResult,
                         oxidation: str, spin: str,
                         filename: str) -> None:
    """Layout for the heuristic spin guess + the BVS card.

    Both panels share the same precondition (result.bonds non-empty)
    and both depend on the annotated oxidation state. When oxidation
    is annotated they're shown side-by-side; when not, we fall back
    to the "probe both" BVS card alone.
    """
    if result.bonds.empty:
        return

    distances_by_fe = _formal_bonds_by_fe(result)
    if not distances_by_fe:
        return

    mean_distance = float(result.bonds[COL_DISTANCE].mean())

    # ---- Branch A: oxidation annotated → two side-by-side cards
    if oxidation != "(unknown)":
        left, right = st.columns(2)
        with left:
            _render_spin_hint(mean_distance, oxidation, spin)
        with right:
            _render_bvs_card(distances_by_fe, oxidation, spin, filename)
    # ---- Branch B: oxidation unknown → probe both Fe(II) and Fe(III)
    else:
        _render_bvs_probe(distances_by_fe)

    # ---- Auto-inference panel — always shown (per-Fe) ----------
    # Replaces the manual trial-and-error of toggling oxidation/spin
    # to see which combination gives the smallest |BVS−Z|. Shows all
    # four candidates ranked, highlights the bond-length-consistent
    # pick, surfaces caveats (especially porphyrin overshoot).
    _render_inference_panel(distances_by_fe, filename)


def _render_spin_hint(mean_d: float, oxidation: str, spin: str) -> None:
    """The original spin-state heuristic card, unchanged in behaviour."""
    suggestion = suggest_spin_state(mean_d, oxidation)
    if not suggestion:
        return
    if spin == "(unknown)":
        st.info(
            f"**Spin-state suggestion**  \n"
            f"Mean Fe–N = **{mean_d:.3f} Å** → {suggestion}.\n\n"
            "Rule of thumb, not a measurement. Confirm with magnetic "
            "susceptibility, Mössbauer, or the original publication "
            "before quoting."
        )
    else:
        st.caption(
            f"_Heuristic check: mean {mean_d:.3f} Å is "
            f"{suggestion}. You annotated **{spin}**._"
        )


def _render_bvs_card(distances_by_fe: dict[str, list[float]],
                     oxidation: str, spin: str, filename: str) -> None:
    """BVS card with status badge + editable R₀ expander.

    Two paths:
      - Literature R₀ exists for (oxidation, spin) → standard card
        with status badge against the literature value.
      - No literature R₀ (Fe(0), Fe(IV)) → 'no-literature' notice,
        and the user can still supply a custom R₀ via the expander to
        compute and judge a BVS themselves.
    """
    spin_for_R0 = spin if spin in ("LS", "HS") else None
    has_lit = has_literature_R0(oxidation, spin_for_R0)
    if has_lit:
        default_R0, default_source = choose_R0(oxidation, spin_for_R0)
    else:
        # Sensible starting point for user tuning: midway between the
        # two generic Fe values (1.769 and 1.815). The user can move
        # it freely via the number input.
        default_R0, default_source = 1.80, f"no published R₀ for {oxidation}–N"

    # Session-state-backed R₀ override. The key includes annotations so
    # changing oxidation / spin resets the field to the new default.
    state_key = f"bvs_R0::{filename}::{oxidation}::{spin}"
    R0_value = float(st.session_state.get(state_key, default_R0))

    # When there's no literature R₀ we treat ANY value (including the
    # 1.80 starting point) as user-supplied for compute_bvs's purposes.
    R0_for_compute = R0_value if (not has_lit
                                  or R0_value != default_R0) else None

    # Compute BVS per Fe centre under the active R₀.
    rows = []
    for fe_label, dists in distances_by_fe.items():
        result = compute_bvs(dists, oxidation,
                             spin_for_R0, R0_override=R0_for_compute)
        if result is None:
            continue
        level, msg = consistency_status(result.bvs, oxidation)
        rows.append((fe_label, result.bvs, level, msg))

    # ---- Fe(0) / Fe(IV) and no override yet → prompt the user ----
    if not has_lit and not rows:
        st.info(
            f"**Bond-valence sum**  \n"
            f"No published R₀ for {oxidation}–N. "
            f"Liebschner 2017 covers Fe(II) and Fe(III) only. "
            f"To compute BVS for {oxidation}, supply a custom R₀ below "
            f"(typical Fe(IV)–N values from the literature are around "
            f"1.65–1.75 Å; for Fe(0)–N around 1.85–1.95 Å — both are "
            f"approximate)."
        )
        _render_R0_expander(state_key, default_R0, default_source,
                            has_literature=False)
        _render_about_bvs_expander()
        return

    if not rows:
        return

    # ---- Headline (single Fe centre) or compact list (multiple) ----
    if len(rows) == 1:
        fe_label, bvs, level, msg = rows[0]
        icon = _STATUS_ICON[level]
        source = default_source if R0_for_compute is None else "user-supplied"
        body = (
            f"**Bond-valence sum**  \n"
            f"{icon} BVS = **{bvs:.2f}** "
            f"({len(distances_by_fe[fe_label])} bonds; R₀ = {R0_value:.3f} Å, "
            f"{source})\n\n_{msg}_"
        )
        {"good": st.success, "caution": st.warning,
         "warning": st.error}[level](body)
    else:
        body_lines = ["**Bond-valence sum**", ""]
        for fe_label, bvs, level, msg in rows:
            icon = _STATUS_ICON[level]
            body_lines.append(f"{icon} **{fe_label}**: BVS = {bvs:.2f} — {msg}")
        st.info("\n".join(body_lines))

    _render_R0_expander(state_key, default_R0, default_source,
                        has_literature=has_lit)
    _render_about_bvs_expander()


# ---- Shared expanders -------------------------------------------------

def _render_R0_expander(state_key: str,
                        default_R0: float,
                        default_source: str,
                        *, has_literature: bool) -> None:
    """The 'Adjust R₀' panel — used by both literature and no-literature
    paths in _render_bvs_card."""
    R0_value = float(st.session_state.get(state_key, default_R0))
    title = ("Adjust R₀" if has_literature
             else "Supply a custom R₀ (required for Fe(0) / Fe(IV))")
    with st.expander(title, expanded=not has_literature):
        if has_literature:
            st.markdown(
                f"**Active R₀:** {R0_value:.3f} Å · "
                f"**source:** "
                f"{default_source if R0_value == default_R0 else 'user-supplied'} · "
                f"**literature default:** {default_R0:.3f} Å"
            )
        else:
            st.markdown(
                f"**Active R₀:** {R0_value:.3f} Å · "
                f"**source:** user-supplied · "
                f"_{default_source}_"
            )

        new_R0 = st.number_input(
            "R₀ (Å)",
            min_value=1.40, max_value=2.20,
            value=R0_value, step=0.01, format="%.3f",
            key=f"input::{state_key}",
            help="Tune to see how BVS responds. Literature R₀ values "
                 "are not exact for every chemistry; chemists routinely "
                 "adjust by ±0.03 Å for their system.",
        )
        if new_R0 != R0_value:
            st.session_state[state_key] = new_R0
            st.rerun()

        reset_label = ("Reset to literature" if has_literature
                       else "Reset to starting value")
        if st.button(reset_label,
                     key=f"reset::{state_key}",
                     help=f"Restore R₀ = {default_R0:.3f} Å."):
            st.session_state.pop(state_key, None)
            st.rerun()

        st.divider()
        st.markdown("**Reference R₀ values (Å):**")
        col_lieb, col_gen = st.columns(2)
        with col_lieb:
            st.markdown(
                "**Liebschner 2017** (spin-specific):\n"
                "- Fe(II) LS: 1.78\n"
                "- Fe(II) HS: 1.91\n"
                "- Fe(III) LS: 1.70\n"
                "- Fe(III) HS: 1.83"
            )
        with col_gen:
            st.markdown(
                "**Generic** (spin unknown):\n"
                "- Fe(II): 1.769 (Brown & Altermatt 1985)\n"
                "- Fe(III): 1.815 (Brese & O'Keeffe 1991)\n"
                "\n_(no published Fe(0) or Fe(IV) R₀ in widely-used "
                "tables — supply yours above if you have a source.)_"
            )


def _render_about_bvs_expander() -> None:
    with st.expander("About BVS"):
        st.markdown(
            "**Bond-valence sum** is the sum over coordinating bonds of "
            "$s_i = \\exp((R_0 - R_i) / B)$ with $B = 0.37$ Å. A "
            "well-behaved coordination sphere yields a BVS close to "
            "the formal oxidation state Z.\n\n"
            "Deviation can mean (a) a wrong oxidation-state annotation, "
            "(b) the wrong R₀ for the chemistry (especially spin-state "
            "mismatch for Fe–N — see Liebschner 2017), or (c) a "
            "coordination sphere missing atoms or padded with secondary "
            "contacts.\n\n"
            "**References:**\n"
            "- Brown & Altermatt (1985) *Acta Cryst.* B**41**, 244.\n"
            "- Brese & O'Keeffe (1991) *Acta Cryst.* B**47**, 192.\n"
            "- Liebschner et al. (2017) *Acta Cryst.* D**73**, 148."
        )


_CONFIDENCE_TINT = {
    "high":   ("✅", st.success),
    "medium": ("⚠️", st.warning),
    "low":    ("🚫", st.error),
}


def _render_inference_panel(distances_by_fe: dict[str, list[float]],
                            filename: str) -> None:
    """Auto-infer (oxidation, spin) from the Fe–N geometry.

    For each Fe centre we run `infer_oxidation_spin()` over its formal
    Fe–N distances. The full 4-combo ranking is shown in a table so
    the user can see the BVS deviation and bond-length plausibility
    side by side; the recommended pick is rendered as a coloured
    info box with an 'Apply' button that writes the annotation into
    the session-state store (which Mode 1's sidebar reads on its
    next rerun).
    """
    if not distances_by_fe:
        return

    st.subheader("Suggested oxidation & spin from bond geometry")
    st.caption(
        "For each Fe centre, computes BVS at all four (Fe(II)/Fe(III)) "
        "× (LS/HS) combinations and cross-checks against the typical "
        "Halcrow bond-length bands. The recommendation is a "
        "*heuristic*, not an assignment — read the caveats."
    )

    for fe_label, distances in distances_by_fe.items():
        result = infer_oxidation_spin(distances)
        if result is None:
            continue
        with st.container(border=True):
            head_ox  = result.best_oxidation
            head_spin = result.best_spin
            icon, banner = _CONFIDENCE_TINT.get(
                result.confidence, ("", st.info)
            )
            banner(
                f"{icon} **{fe_label}** — best fit: "
                f"**{head_ox} {head_spin}** "
                f"(mean Fe–N = {result.mean_distance_A:.3f} Å, "
                f"confidence: **{result.confidence}**)"
            )

            # Full 4-combo table
            import pandas as pd
            rows = []
            for c in result.candidates:
                Z = 2 if c.oxidation == "Fe(II)" else 3
                rows.append({
                    "oxidation": c.oxidation,
                    "spin":      c.spin,
                    "R₀ (Å)":    f"{c.R0:.3f}",
                    "BVS":       f"{c.bvs:.2f}",
                    "|BVS−Z|":   f"{c.deviation:.2f}",
                    "in band?":  "✅ yes" if c.bond_length_consistent else "—",
                    "reason":    c.reason,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            for caveat in result.caveats:
                st.warning(f"_{caveat}_")

            # Apply button — writes to session_state.annotations and
            # forces a rerun. Mode 1's sidebar then defaults to the
            # new oxidation/spin on the next paint.
            apply_key = f"apply_infer_{filename}_{fe_label}"
            if st.button(
                f"Apply '{head_ox} {head_spin}' to the sidebar annotation",
                key=apply_key,
                help="Writes the recommended annotation to the "
                     "session store. The sidebar selectboxes will "
                     "update on the next interaction.",
            ):
                ann.update(
                    filename,
                    content_hash=ann.hash_bytes(b""),
                    oxidation_state=head_ox or "(unknown)",
                    spin_state=head_spin or "(unknown)",
                )
                st.success(
                    f"Annotation set: **{head_ox} {head_spin}**. "
                    f"The BVS card above will refresh on your next "
                    f"sidebar interaction."
                )


def _render_bvs_probe(distances_by_fe: dict[str, list[float]]) -> None:
    """When the user hasn't annotated oxidation, compute BVS for both
    candidates and let them see which is closer to an integer."""
    st.info(
        "**BVS probe — both candidates**  \n"
        "Oxidation state is not annotated. BVS computed against both "
        "generic R₀ values; the one closer to its integer is the more "
        "plausible assignment (with all the usual caveats — see expander)."
    )
    rows = []
    for fe_label, dists in distances_by_fe.items():
        probe = probe_both_oxidation_states(dists)
        if probe is None:
            continue
        rows.append({
            "Fe centre": fe_label,
            "n_FeN":     len(dists),
            "BVS (Fe(II) generic, R₀=1.769)":
                f"{probe['Fe(II)'].bvs:.2f}",
            "BVS (Fe(III) generic, R₀=1.815)":
                f"{probe['Fe(III)'].bvs:.2f}",
        })
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

    with st.expander("About BVS"):
        st.markdown(
            "**Bond-valence sum** uses $s_i = \\exp((R_0 - R_i) / 0.37)$ "
            "with one $R_0$ per (metal, donor, oxidation, spin) combo. "
            "Compare BVS to the formal oxidation state Z; "
            "deviation suggests a different Z, wrong R₀, or "
            "incomplete coordination sphere.\n\n"
            "Annotate oxidation state in the sidebar to enable the "
            "richer status-badged comparison."
        )


# -----------------------------------------------------------------
# Geometry panel — ζ, Δ, Σ, Θ per Fe centre
# -----------------------------------------------------------------
def _fmt(value: float | None, fmt: str = "{:.3f}", unit: str = "") -> str:
    """Format a possibly-None number, falling back to '—' for None."""
    if value is None:
        return "—"
    return fmt.format(value) + (f" {unit}" if unit else "")


def _build_display_table(df):
    """Pure helper: build the per-bond DataFrame the UI shows.

    Replaces the numeric `distance_A` + `distance_esd_A` pair with a
    single human-readable 'distance (Å)' column rendered via
    format_with_esd("1.984(7)" style). Preserves the column order of
    the input and drops the raw numeric distance columns from the
    visible output. Designed for st.dataframe; the CSV export uses
    the original numeric `df` so machine consumers still get raw
    numbers.

    Factored out of render() purely so it's unit-testable — a duplicate
    'distance (Å)' column slipped past the earlier inline version when
    user-annotation columns came after the raw distances.
    """
    import pandas as pd  # local import keeps top-of-module light
    display = df.copy()
    display["distance (Å)"] = display.apply(
        lambda r: format_with_esd(r[COL_DISTANCE],
                                  r.get(COL_DISTANCE_ESD)),
        axis=1,
    )
    # Rebuild the column order from scratch:
    #   - replace COL_DISTANCE with 'distance (Å)' in place,
    #   - drop COL_DISTANCE_ESD,
    #   - keep everything else (including any added user_* cols).
    new_cols: list[str] = []
    for c in df.columns:
        if c == COL_DISTANCE:
            new_cols.append("distance (Å)")
        elif c == COL_DISTANCE_ESD:
            continue
        else:
            new_cols.append(c)
    # Any columns added to display that aren't in df (none right now,
    # but be defensive): append them at the end, skipping the formatted
    # column itself since it's already in new_cols.
    for c in display.columns:
        if c not in new_cols and c not in (COL_DISTANCE, COL_DISTANCE_ESD):
            new_cols.append(c)
    return display[new_cols]


def _coord_badge(label: str, result: AnalysisResult) -> str:
    """Human-readable summary of one Fe centre's coordination geometry."""
    cg = result.coord_geom.get(label)
    if cg is None:
        geom = result.geometry.get(label)
        n = geom.coordination_number if geom else 0
        return f"**{label}** \u00b7 coordination number = **{n}**"

    head = (f"**{label}** \u00b7 **{cg.coordination_number}-coordinate** \u00b7 "
            f"**{cg.classification}**")
    if cg.tau is not None and cg.tau_label is not None:
        head += f" ({cg.tau_label} = {cg.tau:.2f})"
    if cg.trans_angles_deg:
        ta = ", ".join(f"{a:.1f}\u00b0" for a in cg.trans_angles_deg)
        head += f" \u00b7 trans angles {ta}"
    return head


def _render_geometry(result: AnalysisResult, oxidation: str) -> None:
    """Coordination classification + octahedral distortion per Fe centre.

    Single Fe centre  -> badge + four metric cards.
    Multiple centres  -> badges + comparison table including geometry class.
    The Sigma reference-band chart is shown only when oxidation is Fe(II)
    or Fe(III) and at least one Sigma value is present.
    """
    if not result.geometry:
        return

    st.subheader("Geometry \u2014 coordination & octahedral distortion")
    st.caption(
        "\u03b6 = \u03a3 |d \u2212 \u27e8d\u27e9| \u00b7 "
        "\u0394 = (1/n) \u03a3 ((d \u2212 \u27e8d\u27e9)/\u27e8d\u27e9)\u00b2 "
        "\u00b7 \u03a3 = sum of |90\u00b0 \u2212 \u03c6| over 12 cis N\u2013Fe\u2013N "
        "angles \u00b7 \u0398 = sum of |60\u00b0 \u2212 \u03b8| over 24 "
        "projected angles (OctaDist / Ketkaew 2021 convention). "
        "All four are zero for a perfect octahedron. "
        "Non-6-coord centres are classified by \u03c4\u2085 (Addison 1984) "
        "or \u03c4\u2084 (Yang 2007)."
    )

    fe_labels = list(result.geometry.keys())

    # -- Single Fe centre --
    if len(fe_labels) == 1:
        label = fe_labels[0]
        st.markdown(_coord_badge(label, result))
        geom = result.geometry[label]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("\u03b6 (\u00c5)", _fmt(geom.zeta))
        c2.metric(
            "\u0394 (\u00d710\u207b\u2074)",
            _fmt(geom.delta * 1e4 if geom.delta is not None else None,
                 "{:.2f}"),
        )
        c3.metric("\u03a3 (\u00b0)", _fmt(geom.sigma, "{:.2f}"))
        c4.metric("\u0398 (\u00b0)", _fmt(geom.theta, "{:.2f}"))
        for note in geom.notes:
            st.caption(f"_{note}_")
        cg = result.coord_geom.get(label)
        if cg is not None:
            for note in cg.notes:
                st.caption(f"_{note}_")

    # -- Multiple Fe centres --
    else:
        for label in fe_labels:
            st.markdown(_coord_badge(label, result))
        rows = []
        for label in fe_labels:
            g = result.geometry[label]
            cg = result.coord_geom.get(label)
            geom_class = cg.classification if cg else "\u2014"
            rows.append({
                "Fe centre": label,
                "geometry": geom_class,
                "n_FeN": g.coordination_number,
                "\u03b6 (\u00c5)": _fmt(g.zeta),
                "\u0394 (\u00d710\u207b\u2074)": _fmt(
                    g.delta * 1e4 if g.delta is not None else None,
                    "{:.2f}"),
                "\u03a3 (\u00b0)": _fmt(g.sigma, "{:.2f}"),
                "\u0398 (\u00b0)": _fmt(g.theta, "{:.2f}"),
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)
        for label in fe_labels:
            for note in result.geometry[label].notes:
                st.caption(f"_{label}: {note}_")
            cg = result.coord_geom.get(label)
            if cg is not None:
                for note in cg.notes:
                    st.caption(f"_{label}: {note}_")

    # -- Sigma reference chart (gated on Fe(II)/Fe(III)) --
    if oxidation in ("Fe(II)", "Fe(III)"):
        sigma_map = {label: result.geometry[label].sigma
                     for label in fe_labels}
        if any(v is not None for v in sigma_map.values()):
            st.plotly_chart(
                sigma_reference_chart(sigma_map, oxidation),
                use_container_width=True,
            )
            st.caption(
                "_Reference bands are typical ranges from Halcrow "
                "tabulations of non-macrocyclic SCO compounds. \u03a3 "
                "below the LS band typically indicates a macrocyclic or "
                "otherwise geometrically rigid ligand framework "
                "(porphyrins, phthalocyanines, calixarenes)._"
            )
