"""
Mode 3 — Reference library.

A gallery of curated teaching CIFs. Each entry shows its
identification, what to look for, and (when the CIF is available
on disk) a button to load it into Mode 1 with annotations pre-
filled. The "Compare two" view puts two entries side-by-side so
the user can read off the differences directly.

Adding a new entry: drop a CIF into `data/reference/` and append
an object to `library.json`. No code change required.
"""

from __future__ import annotations
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

from src.library import LibraryEntry, LibraryError, load_library
from src import annotations as ann


# Where the library lives on disk. `data/reference/` sits next to
# the `src/` directory (= project root, two parents up from this
# file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIBRARY_DIR = _PROJECT_ROOT / "data" / "reference"


# Session-state keys used by this mode.
_SELECTED_KEY    = "library_selected_id"
_COMPARE_LEFT    = "library_compare_left"
_COMPARE_RIGHT   = "library_compare_right"


# ----------------------------------------------------------------------
# UploadedFeCentre — a card built from a CIF the user dropped *into
# Mode 3 itself*, separate from the persistent library. Same shape as
# LibraryEntry where it matters (display_name, teaching_notes,
# expected_metrics) so the card renderer / compare panel handle both
# without branching.
# ----------------------------------------------------------------------
@dataclass
class UploadedFeCentre:
    """One Fe centre from a user-uploaded CIF, transient (session-scoped)."""
    id: str
    cif_filename: str
    fe_label: str
    display_name: str
    description: str
    teaching_notes: list[str]
    measured_metrics: dict[str, float]
    n_FeN: int
    temperature_K: float | None
    refcode: str | None
    coordination: str | None

    # LibraryEntry-compatible aliases so the same renderer/comparator works
    compound_class: str = "(user upload — this session only)"
    oxidation_state: str | None = None
    spin_state: str | None = None
    cod_id: str | None = None
    doi: str | None = None
    license: str = "User-uploaded CIF; not persisted to disk."
    expected_metrics: dict[str, float] = field(default_factory=dict)
    cif_exists: bool = False           # CIF is in-memory only
    cif_path: Path | None = None
    is_uploaded: bool = True

    def __post_init__(self) -> None:
        # Convenience: ensure expected_metrics dict exists even if
        # not supplied so the card renderer's `if e.expected_metrics`
        # check works.
        if self.expected_metrics is None:
            self.expected_metrics = {}


# Provide an `is_uploaded` attribute on real LibraryEntry as well by
# patching after import, so the renderer can branch cleanly.
LibraryEntry.is_uploaded = False                              # type: ignore[attr-defined]
LibraryEntry.measured_metrics = {}                            # type: ignore[attr-defined]


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@st.cache_data(show_spinner=False)
def _measure_uploaded_cif(
    content_hash: str, file_bytes: bytes, filename: str,
    schema_version: str = "v1-upload",
) -> list[dict]:
    """Parse and analyse one uploaded CIF; return per-Fe-centre dicts.

    Cached by SHA-256 so re-uploads don't re-parse. Returns plain dicts
    (not UploadedFeCentre) because @st.cache_data pickles its output
    and dataclasses with arbitrary fields are fragile across schema
    bumps.

    Parse failures surface as a single error-card dict so the user
    sees which file misbehaved rather than a silent drop.
    """
    del content_hash, schema_version                          # cache key only

    # ccdc imports are lazy inside read_cif so this still works in
    # environments without ccdc, falling through to the error path.
    from src.cif_reader import read_cif
    from src.fe_n_analysis import analyse
    from src.data_schema import (
        COL_DETECTION_METHOD, COL_DISTANCE, COL_FE_LABEL,
        DetectionMethod,
    )

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        try:
            bundle = read_cif(tmp_path)
            result = analyse(bundle, cif_filename=filename,
                             min_occupancy=0.5)
        except Exception as exc:                              # noqa: BLE001
            return [_error_entry_dict(filename, repr(exc))]

        if not result.geometry:
            return [_error_entry_dict(
                filename,
                "No N-coordinated Fe centre found in this CIF. "
                "(All Fe atoms either had zero formal Fe–N bonds, "
                "or there were no Fe atoms at all.)",
            )]

        prov = bundle.provenance
        out: list[dict] = []
        for fe_label, oct_d in result.geometry.items():
            formal = result.bonds[
                (result.bonds[COL_FE_LABEL] == fe_label)
                & (result.bonds[COL_DETECTION_METHOD]
                   == DetectionMethod.FORMAL_BOND.value)
            ]
            if formal.empty:
                continue

            distances = [float(d) for d in formal[COL_DISTANCE].tolist()]
            mean_d = sum(distances) / len(distances)

            measured: dict[str, float] = {
                "mean_FeN_A": round(mean_d, 4),
                "n_FeN":      len(distances),
                "min_FeN_A":  round(min(distances), 4),
                "max_FeN_A":  round(max(distances), 4),
            }
            if oct_d.zeta is not None:
                measured["zeta_A"]    = round(oct_d.zeta, 4)
            if oct_d.delta is not None:
                measured["delta_x1e4"] = round(oct_d.delta * 1e4, 2)
            if oct_d.sigma is not None:
                measured["sigma_deg"]  = round(oct_d.sigma, 2)
            if oct_d.theta is not None:
                measured["theta_deg"]  = round(oct_d.theta, 2)

            cg = result.coord_geom.get(fe_label) if result.coord_geom else None
            coordination = cg.classification if cg else None

            teaching = _autonotes_for_uploaded(
                mean_d, oct_d.sigma, oct_d.theta, len(distances), coordination,
            )

            t_str = (f"{bundle.temperature_K:.0f} K"
                     if bundle.temperature_K is not None else "T = unknown")
            refcode = (prov.ccdc_refcode if prov else None) or filename
            display_name = f"{filename} · {fe_label}"
            description = (
                f"User-uploaded CIF. {len(distances)}-coordinate Fe centre "
                f"at {t_str}. "
                + (f"Refcode {refcode}. " if prov and prov.ccdc_refcode else "")
                + "Measured metrics computed from this file's coordinates."
            )

            out.append(asdict(UploadedFeCentre(
                id=f"upload__{filename}__{fe_label}",
                cif_filename=filename,
                fe_label=fe_label,
                display_name=display_name,
                description=description,
                teaching_notes=teaching,
                measured_metrics=measured,
                n_FeN=len(distances),
                temperature_K=bundle.temperature_K,
                refcode=(prov.ccdc_refcode if prov else None),
                coordination=coordination,
            )))
        if not out:
            return [_error_entry_dict(
                filename,
                "Parsed OK but no Fe centre had formal Fe–N bonds "
                "after the occupancy filter (try Mode 1 for diagnostics).",
            )]
        return out
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _error_entry_dict(filename: str, reason: str) -> dict:
    """A placeholder dict that renders as a failed-upload card."""
    return asdict(UploadedFeCentre(
        id=f"upload__error__{filename}",
        cif_filename=filename,
        fe_label="—",
        display_name=f"⚠ {filename}",
        description=f"**Parse failed:** {reason}",
        teaching_notes=[
            "This file couldn't be processed — try opening it in Mode 1 "
            "(Single CIF) to see the full diagnostic message."
        ],
        measured_metrics={},
        n_FeN=0,
        temperature_K=None,
        refcode=None,
        coordination=None,
    ))


def _autonotes_for_uploaded(
    mean_d: float,
    sigma_deg: float | None,
    theta_deg: float | None,
    n_FeN: int,
    coordination: str | None,
) -> list[str]:
    """Generate teaching notes from an uploaded CIF's measured numbers.

    These are descriptive, not prescriptive — we describe what the
    numbers say structurally without inferring spin state. Anything
    that requires the user's annotation (oxidation, definitive spin
    classification) is deliberately left out.
    """
    notes: list[str] = []
    notes.append(
        f"Mean Fe–N = **{mean_d:.3f} Å** over **{n_FeN}** bond(s)."
    )
    if coordination:
        notes.append(f"Coordination classified as **{coordination}**.")
    if 1.93 <= mean_d <= 2.02:
        notes.append(
            "Mean is in the **low-spin / short** band typical of "
            "Fe(II) LS or Fe(III) LS (1.93–2.02 Å)."
        )
    elif 2.10 <= mean_d <= 2.25:
        notes.append(
            "Mean is in the **high-spin / long** band typical of "
            "Fe(II) HS (2.10–2.25 Å)."
        )
    elif 2.02 < mean_d < 2.10:
        notes.append(
            "Mean is in the **borderline / Fe(III)-HS** region "
            "(2.02–2.10 Å) — could be Fe(III) HS or a "
            "spin-equilibrium sample."
        )
    if sigma_deg is not None and n_FeN == 6:
        if sigma_deg < 25:
            notes.append(
                f"Σ = **{sigma_deg:.1f}°** — very ordered, "
                "below the typical Halcrow LS range. "
                "Suggests a macrocyclic or otherwise rigid ligand "
                "(porphyrin, phthalocyanine, calix-N₄)."
            )
        elif sigma_deg < 60:
            notes.append(
                f"Σ = **{sigma_deg:.1f}°** — within the typical "
                "Fe(II) LS Halcrow band (30–60°)."
            )
        elif sigma_deg < 140:
            notes.append(
                f"Σ = **{sigma_deg:.1f}°** — within the typical "
                "Fe(II) HS Halcrow band (80–140°)."
            )
    if theta_deg is not None and theta_deg > 400:
        notes.append(
            f"Θ = **{theta_deg:.1f}°** — substantial trigonal twist; "
            "the octahedron is significantly distorted from ideal."
        )
    return notes


# -----------------------------------------------------------------
# Cross-mode store integration
# -----------------------------------------------------------------
def _persist_upload_to_store(file_bytes: bytes, filename: str) -> None:
    """Save a Mode-3 upload into the shared analysed-CIF store so the
    same file behaves identically whether the user came in via Mode 1
    or via Mode 3's upload widget.

    Failures are swallowed — recording is an enhancement, not a
    critical path.
    """
    try:
        from src import analysed_store as _astore
        from src.cif_reader import read_cif
        from src.fe_n_analysis import analyse
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            bundle = read_cif(tmp_path)
            result = analyse(bundle, cif_filename=filename, min_occupancy=0.5)
            _astore.record_from_analysis(filename, file_bytes, bundle, result)
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    except Exception:                                      # noqa: BLE001
        pass


def _entries_from_session_store(
    *, already_uploaded: list[str] | None = None,
) -> list[UploadedFeCentre]:
    """Build UploadedFeCentre cards from the cross-mode session store.

    The store key is the filename; we dedupe against `already_uploaded`
    so a CIF the user just dropped into Mode 3 doesn't appear twice
    (once as an upload, once as a session entry).

    Reads per-filename annotations (oxidation, spin) from src.annotations
    so 'Apply' clicks in Mode 1 are reflected here without re-running.
    """
    from src import analysed_store as _astore
    from src import annotations as _ann

    skip = set(already_uploaded or [])
    cards: list[UploadedFeCentre] = []

    for filename, entry in _astore.all_analysed().items():
        if filename in skip:
            continue
        fe_centres = entry.get("fe_centres") or {}
        if not fe_centres:
            # Analysed but no N-coordinated Fe — drop a single
            # informational card so the user sees it was processed.
            cards.append(UploadedFeCentre(
                id=f"session__{filename}__no_fe",
                cif_filename=filename,
                fe_label="—",
                display_name=f"{filename} (no Fe–N coordination)",
                description=(
                    "Analysed in this session but contains no "
                    "N-coordinated Fe centre."
                ),
                teaching_notes=[],
                measured_metrics={},
                n_FeN=0,
                temperature_K=entry.get("temperature_K"),
                refcode=entry.get("refcode"),
                coordination=None,
                doi=entry.get("doi"),
            ))
            continue

        annotations = _ann.get(filename) or {}
        ox = annotations.get("oxidation_state")
        spin = annotations.get("spin_state")
        if ox in (None, "(unknown)", ""):
            ox = None
        if spin in (None, "(unknown)", ""):
            spin = None

        for fe_label, metrics in fe_centres.items():
            # Strip non-numeric / large-list fields from measured_metrics
            # so the diff table renders cleanly.
            visible_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "distances_A"
            }
            mean_d = metrics.get("mean_FeN_A")
            t_str = (f"{entry['temperature_K']:.0f} K"
                     if entry.get("temperature_K") is not None
                     else "T = unknown")
            teaching = _autonotes_for_uploaded(
                mean_d, metrics.get("sigma_deg"),
                metrics.get("theta_deg"), metrics.get("n_FeN", 0),
                metrics.get("coordination"),
            )
            descr_parts = [
                f"Analysed in this session at {t_str}."
            ]
            if entry.get("refcode"):
                descr_parts.append(f"Refcode {entry['refcode']}.")
            if ox or spin:
                pin = ", ".join(p for p in (ox, spin) if p)
                descr_parts.append(f"Current annotation: **{pin}**.")
            cards.append(UploadedFeCentre(
                id=f"session__{filename}__{fe_label}",
                cif_filename=filename,
                fe_label=fe_label,
                display_name=f"{filename} · {fe_label}",
                description=" ".join(descr_parts),
                teaching_notes=teaching,
                measured_metrics=visible_metrics,
                n_FeN=metrics.get("n_FeN", 0),
                temperature_K=entry.get("temperature_K"),
                refcode=entry.get("refcode"),
                coordination=metrics.get("coordination"),
                oxidation_state=ox,
                spin_state=spin,
                doi=entry.get("doi"),
            ))
    return cards


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def _badge(label: str, value: str | None) -> str:
    """Render a small inline badge as markdown."""
    if value is None or value == "":
        return f"`{label}: —`"
    return f"**{label}:** {value}"


def _entry_caption(e: LibraryEntry) -> str:
    """One-line summary suitable for the top of a card."""
    bits: list[str] = []
    if e.oxidation_state:
        bits.append(e.oxidation_state)
    if e.spin_state:
        bits.append(e.spin_state)
    if e.temperature_K is not None:
        bits.append(f"T = {e.temperature_K:.0f} K")
    if e.cod_id:
        bits.append(f"COD {e.cod_id}")
    return "  ·  ".join(bits) if bits else "(no annotation)"


def _doi_link(doi: str) -> str:
    """Markdown DOI link with URL-safe encoding."""
    safe = doi.replace("+", "%2B")
    return f"[{doi}](https://doi.org/{safe})"


# ---------------------------------------------------------------------
# Display names for the per-Fe metric keys.
#
# The session store and the library JSON both use snake_case keys
# (sigma_deg, theta_deg, etc.) because they're robust against pandas
# / json round-trips. The UI deserves the actual Greek letters.
# ---------------------------------------------------------------------
_METRIC_DISPLAY: dict[str, str] = {
    "mean_FeN_A":          "⟨Fe–N⟩ (Å)",
    "min_FeN_A":           "min Fe–N (Å)",
    "max_FeN_A":           "max Fe–N (Å)",
    "n_FeN":               "n Fe–N",
    "sigma_deg":           "Σ (°)",
    "theta_deg":           "Θ (°)",
    "zeta_A":              "ζ (Å)",
    "delta":               "Δ",
    "delta_x1e4":          "Δ (×10⁻⁴)",
    "T_K":                 "T (K)",
    "R_factor_pct":        "R-factor (%)",
    "sigma_deg_approx":    "Σ ≈ (°)",
    "theta_deg_approx":    "Θ ≈ (°)",
    "BVS":                 "BVS",
}


def _pretty_metric(key: str) -> str:
    """Map a raw metric key to a display label with Greek letters."""
    return _METRIC_DISPLAY.get(key, key)


def _entry_metrics_table(e) -> pd.DataFrame:
    """Compact key/value DataFrame for the comparison view.

    Works for both LibraryEntry and UploadedFeCentre. Library entries
    contribute their `expected_metrics`; uploaded entries contribute
    their `measured_metrics`. We label rows with the right prefix so
    the user can tell which is which when both are stacked vertically.
    """
    rows = [
        ("compound",       e.display_name),
        ("class",          getattr(e, "compound_class", None) or "—"),
        ("oxidation",      getattr(e, "oxidation_state", None) or "—"),
        ("spin",           getattr(e, "spin_state", None) or "—"),
        ("T (K)",          f"{e.temperature_K:.0f}"
                            if getattr(e, "temperature_K", None) is not None
                            else "—"),
        ("COD ID",         getattr(e, "cod_id", None) or "—"),
        ("DOI",            getattr(e, "doi", None) or "—"),
    ]
    is_uploaded = getattr(e, "is_uploaded", False)
    if is_uploaded:
        rows.append(("source", "user upload"))
        rows.append(("Fe centre", getattr(e, "fe_label", "—") or "—"))
        rows.append(("coordination",
                     getattr(e, "coordination", None) or "—"))
    else:
        rows.append(("CIF on disk?",
                     "yes" if getattr(e, "cif_exists", False) else "missing"))

    # Expected metrics from the library JSON
    for k, v in (getattr(e, "expected_metrics", {}) or {}).items():
        rows.append((f"expected {_pretty_metric(k)}", f"{v:g}"))
    # Measured metrics from a real parse of an uploaded file
    for k, v in (getattr(e, "measured_metrics", {}) or {}).items():
        rows.append((f"measured {_pretty_metric(k)}", f"{v:g}"))
    return pd.DataFrame(rows, columns=["field", "value"])


# Back-compat alias for the old name used inside the compare panel.
_expected_metrics_table = _entry_metrics_table


# -----------------------------------------------------------------
# Card renderer — used in the gallery
# -----------------------------------------------------------------
def _render_card(e) -> None:
    """Single library or uploaded card.

    Accepts both LibraryEntry and UploadedFeCentre — they have
    compatible attribute surfaces via duck typing.
    """
    try:
        card = st.container(border=True)
    except TypeError:                                    # pragma: no cover
        card = st.container()
    is_uploaded = getattr(e, "is_uploaded", False)
    with card:
        st.markdown(f"#### {e.display_name}")
        st.caption(_entry_caption(e))

        if e.compound_class:
            st.markdown(f"_{e.compound_class}_")

        if e.description:
            st.markdown(e.description)

        if e.teaching_notes:
            with st.expander(
                "What to look for"
                if not is_uploaded
                else "What this CIF's numbers say",
                expanded=is_uploaded,   # auto-open for uploaded so users
            ):                          # don't miss the auto-notes.
                for note in e.teaching_notes:
                    st.markdown(f"- {note}")

        # Expected metrics (from library JSON) ----------------------
        if getattr(e, "expected_metrics", None):
            metrics_inline = "  ·  ".join(
                f"**{_pretty_metric(k)}** ≈ {v:g}"
                for k, v in e.expected_metrics.items()
            )
            st.caption(f"Expected: {metrics_inline}")

        # Measured metrics (from a real parse, uploaded only) -------
        measured = getattr(e, "measured_metrics", None) or {}
        if measured:
            metrics_inline = "  ·  ".join(
                f"**{_pretty_metric(k)}** = {v:g}"
                for k, v in measured.items()
            )
            st.caption(f"Measured: {metrics_inline}")

        # Provenance row --------------------------------------------
        prov_bits: list[str] = []
        if getattr(e, "cod_id", None):
            prov_bits.append(f"COD {e.cod_id}")
        if getattr(e, "doi", None):
            prov_bits.append(f"DOI: {_doi_link(e.doi)}")
        if getattr(e, "license", None):
            prov_bits.append(e.license)
        if prov_bits:
            st.caption("  ·  ".join(prov_bits))

        # Action buttons --------------------------------------------
        col_a, col_b = st.columns([1, 1])
        with col_a:
            # Uploaded entries don't expose a persisted CIF, so the
            # "Use in Single CIF mode" button only applies to library
            # entries. For uploaded ones we surface the filename so
            # the user knows what to re-upload in Mode 1.
            if is_uploaded:
                st.caption(
                    f"_(In-memory only — re-upload "
                    f"`{e.cif_filename}` in Mode 1 to inspect "
                    f"the full per-bond table.)_"
                )
            else:
                disabled = not e.cif_exists
                if st.button(
                    "Inspect in Single CIF mode",
                    key=f"open_{e.id}",
                    disabled=disabled,
                    help=("Pre-fills annotations and writes the CIF "
                          "bytes to session state, then asks you to "
                          "switch to Mode 1 via the sidebar radio."
                          if not disabled else
                          "CIF file is missing from data/reference/."),
                    use_container_width=True,
                ):
                    _stage_for_mode_1(e)
                    st.success(
                        "Loaded. Switch to **Mode 1: Single CIF** "
                        "in the sidebar to inspect."
                    )
        with col_b:
            if st.button(
                "Add to comparison",
                key=f"compare_{e.id}",
                use_container_width=True,
                help="Promote to the 'Compare two' panel below.",
            ):
                _stage_for_comparison(e)


def _stage_for_mode_1(e: LibraryEntry) -> None:
    """Pre-load this entry into Mode 1.

    We:
      - persist its oxidation / spin annotations under the CIF
        filename so Mode 1's sidebar defaults them on next open,
      - stash a 'pending library load' record in session state for
        a future enhancement that auto-uploads the file (the
        current Mode 1 still requires a manual upload — leaving
        this as a hook for Prompt 10's polish pass).
    """
    if e.oxidation_state or e.spin_state:
        try:
            cif_bytes = e.cif_path.read_bytes()
            ann.update(
                e.cif_filename,
                content_hash=ann.hash_bytes(cif_bytes),
                oxidation_state=e.oxidation_state or "(unknown)",
                spin_state=e.spin_state or "(unknown)",
            )
        except OSError:
            pass
    st.session_state["pending_library_entry"] = e.id


def _stage_for_comparison(e: LibraryEntry) -> None:
    """Fill the left then the right comparison slot in turn."""
    if not st.session_state.get(_COMPARE_LEFT):
        st.session_state[_COMPARE_LEFT] = e.id
    else:
        st.session_state[_COMPARE_RIGHT] = e.id


# -----------------------------------------------------------------
# Compare view
# -----------------------------------------------------------------
def _render_compare(entries: list) -> None:
    st.subheader("Compare two references")
    st.caption(
        "Pick two entries to lay their identification, expected "
        "metrics, and teaching notes side-by-side."
    )

    by_id = {e.id: e for e in entries}
    options = ["—"] + [e.id for e in entries]

    col_l, col_r = st.columns(2)
    with col_l:
        left_id = st.selectbox(
            "Left card",
            options=options,
            index=options.index(st.session_state.get(_COMPARE_LEFT, "—"))
                  if st.session_state.get(_COMPARE_LEFT, "—") in options
                  else 0,
            key="compare_left_select",
        )
    with col_r:
        right_id = st.selectbox(
            "Right card",
            options=options,
            index=options.index(st.session_state.get(_COMPARE_RIGHT, "—"))
                  if st.session_state.get(_COMPARE_RIGHT, "—") in options
                  else 0,
            key="compare_right_select",
        )

    if left_id == "—" or right_id == "—":
        st.info(
            "Choose **both** a left and a right entry to see a "
            "side-by-side diff. The 'Add to comparison' buttons on "
            "the cards above also populate these slots — usually "
            "you'd pick your uploaded CIF on one side and a "
            "literature template on the other."
        )
        return
    if left_id == right_id:
        st.warning("Same entry chosen for both slots — pick two different ones.")
        return

    left  = by_id[left_id]
    right = by_id[right_id]

    # Header strip
    h_left, h_right = st.columns(2)
    with h_left:
        st.markdown(f"### {left.display_name}")
        st.caption(_entry_caption(left))
    with h_right:
        st.markdown(f"### {right.display_name}")
        st.caption(_entry_caption(right))

    # Combined diff table — one row per metric, with the value from
    # each side and a numeric delta when both are present.
    st.dataframe(
        _comparison_table(left, right),
        use_container_width=True, hide_index=True,
    )

    # Teaching notes — both, in a 2-column expander row
    notes_left, notes_right = st.columns(2)
    with notes_left:
        if left.teaching_notes:
            with st.expander(f"Notes — {left.display_name}", expanded=True):
                for note in left.teaching_notes:
                    st.markdown(f"- {note}")
    with notes_right:
        if right.teaching_notes:
            with st.expander(f"Notes — {right.display_name}", expanded=True):
                for note in right.teaching_notes:
                    st.markdown(f"- {note}")


# ---------------------------------------------------------------------
# Combined diff table for the compare view
# ---------------------------------------------------------------------

def _entry_numeric_metrics(e) -> dict[str, float]:
    """Return one numeric metrics dict per entry, regardless of source.

    For uploaded entries we surface `measured_metrics`. For library
    entries we surface `expected_metrics`. Names are normalised so
    a measured `mean_FeN_A` can sit next to an expected `mean_FeN_A`
    in the same row of the diff table.
    """
    out: dict[str, float] = {}
    measured = getattr(e, "measured_metrics", None) or {}
    expected = getattr(e, "expected_metrics", None) or {}
    # Measured wins when both present (we're inspecting a real upload).
    for k, v in expected.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    for k, v in measured.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    # Surface temperature too — useful as a row in the diff.
    if getattr(e, "temperature_K", None) is not None:
        out["T_K"] = float(e.temperature_K)
    return out


def _label_for_metric_source(e) -> str:
    """'Measured' for an uploaded entry, 'Expected' for a library entry."""
    return "Measured" if getattr(e, "is_uploaded", False) else "Expected"


def _comparison_table(left, right) -> pd.DataFrame:
    """One row per metric. Columns: metric / left value / right value /
    Δ (right − left) / verdict. The verdict is a small qualitative
    label so the user doesn't have to do mental arithmetic to spot
    matches: close (≤ 10% relative), moderate (10–25%), far (> 25%).
    """
    left_metrics  = _entry_numeric_metrics(left)
    right_metrics = _entry_numeric_metrics(right)

    keys: list[str] = list(dict.fromkeys(
        list(left_metrics) + list(right_metrics)
    ))

    left_kind  = _label_for_metric_source(left)
    right_kind = _label_for_metric_source(right)

    rows = []
    for k in keys:
        lv = left_metrics.get(k)
        rv = right_metrics.get(k)
        lv_str = f"{lv:.3f}" if lv is not None else "—"
        rv_str = f"{rv:.3f}" if rv is not None else "—"
        if lv is not None and rv is not None:
            delta = rv - lv
            delta_str = f"{delta:+.3f}"
            denom = max(abs(lv), abs(rv), 1e-6)
            rel = abs(delta) / denom
            if rel <= 0.10:
                verdict = "close"
            elif rel <= 0.25:
                verdict = "moderate"
            else:
                verdict = "far"
        else:
            delta_str = "—"
            verdict = "—"
        rows.append({
            "metric":              _pretty_metric(k),
            f"left ({left_kind})": lv_str,
            f"right ({right_kind})": rv_str,
            "Δ (right − left)":    delta_str,
            "agreement":           verdict,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------
def render() -> None:
    st.title("Reference library")
    st.caption(
        "A small curated set of CIFs that illustrate the analyses "
        "in this dashboard. Each card explains what the structure "
        "demonstrates and pre-fills oxidation / spin annotations "
        "when promoted to Single-CIF mode."
    )

    try:
        entries = load_library(_LIBRARY_DIR)
    except LibraryError as exc:
        st.error(f"Couldn't load reference library: {exc}")
        return

    if not entries:
        st.info(
            "No reference library configured yet. To set one up:\n\n"
            f"1. Drop one or more `.cif` files into `{_LIBRARY_DIR}`.\n"
            "2. Add a corresponding entry per CIF to "
            f"`{_LIBRARY_DIR / 'library.json'}`.\n\n"
            "See `data/reference/library.json` for the schema."
        )
        return

    # -----------------------------------------------------------------
    # Upload your own CIFs for comparison
    # -----------------------------------------------------------------
    st.subheader("Upload your own CIF(s) to compare")
    st.caption(
        "Drop one or more `.cif` files here to add them as cards "
        "alongside the library. Each Fe centre becomes its own card "
        "with the **measured** ζ / Σ / Θ / ⟨Fe–N⟩ computed from the "
        "file's coordinates. Uploads stay in this browser session "
        "only — nothing is written to `data/reference/`."
    )
    uploads = st.file_uploader(
        "Drop CIFs",
        type=["cif"],
        accept_multiple_files=True,
        help="Same parser as Mode 1. Bond / esd / disorder / "
             "provenance pipelines all run; results land as cards "
             "below.",
        key="library_upload_widget",
    )

    uploaded_entries: list[UploadedFeCentre] = []
    if uploads:
        with st.spinner(f"Parsing {len(uploads)} uploaded CIF(s)…"):
            for f in uploads:
                file_bytes = f.getvalue()
                content_hash = _hash_bytes(file_bytes)
                dicts = _measure_uploaded_cif(content_hash, file_bytes, f.name)
                for d in dicts:
                    uploaded_entries.append(UploadedFeCentre(**d))
                # Also persist into the cross-mode store so a CIF
                # uploaded here is visible the same way a CIF
                # analysed in Mode 1 is — single source of truth.
                _persist_upload_to_store(file_bytes, f.name)

    # -----------------------------------------------------------------
    # Cards that came in from Mode 1 / Mode 2 via the shared store
    # -----------------------------------------------------------------
    # Anything the user analysed in Mode 1 ('Apply' button included)
    # or dropped into the Mode 2 cohort shows up here automatically.
    # The card joins the session-state store with the per-filename
    # annotations (oxidation, spin) so the diff against literature
    # templates reflects whatever's currently pinned.
    session_entries = _entries_from_session_store(
        already_uploaded=[u.cif_filename for u in uploaded_entries],
    )

    st.divider()

    # -----------------------------------------------------------------
    # Combined set + headline counts
    # -----------------------------------------------------------------
    all_entries: list = list(entries) + session_entries + uploaded_entries
    n_library = len(entries)
    n_uploaded = len(uploaded_entries)
    n_session = len(session_entries)
    n_on_disk = sum(1 for e in entries if e.cif_exists)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("library entries",     n_library)
    col_b.metric("from this session",   n_session)
    col_c.metric("uploaded here",       n_uploaded)
    col_d.metric("library CIFs on disk", n_on_disk)

    if n_session:
        st.caption(
            f"_{n_session} Fe centre(s) carried over from CIFs you've "
            "already inspected this session (in Mode 1 or Mode 2). "
            "Their current oxidation / spin annotations are applied — "
            "edit them in Mode 1's sidebar to refresh the diff against "
            "the literature templates below._"
        )

    if n_on_disk < n_library:
        st.caption(
            f"_{n_library - n_on_disk} library entry/entries are "
            "referenced in the manifest but their CIF is missing "
            "from `data/reference/`. The card still renders for "
            "context but the 'Use in Single CIF mode' button is "
            "disabled._"
        )

    # -----------------------------------------------------------------
    # Recently analysed (cross-mode store) — visible diagnostic so the
    # user can confirm what's actually persisted from Mode 1 / Mode 2.
    # If a CIF they inspected in Mode 1 doesn't show up here, the
    # cross-mode link broke and they know to look at the warning the
    # Mode 1 page would have shown.
    # -----------------------------------------------------------------
    from src import analysed_store as _astore_inline
    from src import annotations as _ann_inline
    store_snapshot = _astore_inline.all_analysed()
    with st.expander(
        f"Recently analysed in this session ({len(store_snapshot)})",
        expanded=bool(store_snapshot),
    ):
        if not store_snapshot:
            st.markdown(
                "_No CIFs registered yet. Upload one in **Mode 1** "
                "(or here in Mode 3), run the analysis, and it "
                "will appear in this list. Cards for these files "
                "show up in the gallery below alongside the "
                "library entries._"
            )
        else:
            rows = []
            for fname, entry in store_snapshot.items():
                ann_for = _ann_inline.get(fname) or {}
                fe_keys = list((entry.get("fe_centres") or {}).keys())
                rows.append({
                    "file": fname,
                    "refcode": entry.get("refcode") or "—",
                    "T (K)": (f"{entry['temperature_K']:.0f}"
                               if entry.get("temperature_K") is not None
                               else "—"),
                    "Fe centres": ", ".join(fe_keys) if fe_keys else "(no Fe–N)",
                    "oxidation": ann_for.get("oxidation_state") or "—",
                    "spin":      ann_for.get("spin_state") or "—",
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, hide_index=True,
            )
            # Maintenance: a clear button so the user can reset the
            # cross-mode store if it accumulates unwanted entries.
            if st.button(
                "Clear cross-mode store",
                help="Forget every CIF this Mode-3 page knows from "
                     "other modes. Doesn't touch the library entries "
                     "or your annotations.",
                key="clear_astore_button",
            ):
                _astore_inline.clear()
                st.rerun()

    st.divider()

    # -----------------------------------------------------------------
    # Gallery — 2 columns, library cards first, then uploaded cards
    # -----------------------------------------------------------------
    st.subheader("Gallery")
    cols = st.columns(2)
    for i, entry in enumerate(all_entries):
        with cols[i % 2]:
            _render_card(entry)

    st.divider()
    _render_compare(all_entries)

    st.divider()
    with st.expander("About this library", expanded=False):
        st.markdown(
            "**License model:** entries should be sourced from the "
            "[Crystallography Open Database](https://www.crystallography.net/cod/) "
            "(CC0) — CSD-derived CIFs carry redistribution "
            "restrictions and shouldn't be bundled here.\n\n"
            "**Adding entries:** drop a `.cif` into "
            f"`data/reference/` and add a JSON object to "
            "`library.json` describing it. See the schema "
            "comments in that file for the field set."
        )
