"""
Fe–N CIF inspector — Streamlit entry point.

This file is intentionally thin. All it does is:
  1) Set page config + draw the mode switcher at the top of the sidebar.
  2) Show an Environment panel (versions only — never absolute paths).
  3) Delegate to the chosen mode's render() function.

Run with:

    ~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python -m streamlit \\
        run app.py

Import-order note (macOS):
  - Streamlit MUST be imported before ccdc so Python's _ssl loads with
    the correct libcrypto before ccdc's older one. The mode modules
    import ccdc lazily (inside cif_reader._load_ccdc), keeping that
    order intact even when a mode is switched mid-session.
"""

# 1) Streamlit first.
import streamlit as st

# 2) Stdlib + path setup so `from src.* import ...` resolves no matter
#    where the user launched streamlit from.
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 3) Mode modules — each exposes a render() function.
from src.modes import single_cif, batch, reference_library


# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Fe–N CIF inspector",
    page_icon="🧪",
    layout="wide",
)


# -----------------------------------------------------------------
# Mode registry — add a new mode here, ship the render() function,
# done. No app.py edits beyond this dict.
# -----------------------------------------------------------------
MODES: dict[str, callable] = {
    "Single CIF":        single_cif.render,
    "Batch / cohort":    batch.render,
    "Reference library": reference_library.render,
}


# -----------------------------------------------------------------
# Sidebar — mode switcher at the top, environment versions at the
# bottom. Mode-specific widgets are drawn by each mode's render().
# -----------------------------------------------------------------
mode_name = st.sidebar.radio(
    "Mode",
    options=list(MODES),
    index=0,
    help="Single CIF inspects one structure. Batch handles many. "
         "Library is a curated teaching set.",
)


def _environment_block() -> str:
    """Versions of every library that matters — no absolute paths."""
    lines = [f"Python {sys.version.split()[0]}"]
    for label, modname in [
        ("streamlit", "streamlit"),
        ("pandas",    "pandas"),
        ("plotly",    "plotly"),
        ("ccdc",      "ccdc"),
    ]:
        try:
            module = __import__(modname)
            ver = getattr(module, "__version__", "?")
            lines.append(f"{label} {ver}")
        except Exception:                       # noqa: BLE001
            lines.append(f"{label} not available")
    return "\n".join(lines)


# Mode-specific widgets render BEFORE this block so the env panel
# sits at the bottom of the sidebar regardless of mode.
MODES[mode_name]()

st.sidebar.divider()
with st.sidebar.expander("Environment"):
    st.code(_environment_block())
