"""
Session-state-backed annotation store.

The user assigns oxidation state and spin state to each uploaded CIF.
Those choices must survive:
  - widget changes within the same mode (Streamlit reruns the script),
  - mode switches (single-CIF ↔ batch ↔ reference library),
  - re-uploading the same file (recognised by content hash).

Schema kept in `st.session_state["annotations"]`:

    {
        "myfile.cif": {
            "content_hash": "abc123…",   # SHA-256 of the file bytes
            "oxidation_state": "Fe(II)",
            "spin_state": "LS",
        },
        ...
    }

We key by **filename** because that's what the UI surfaces, but we
also record `content_hash` so a later cohort/library mode can match
a CIF independently of what the user named it on upload.
"""

from __future__ import annotations
import hashlib
from typing import Any

import streamlit as st

# Top-level session_state key. Single source of truth.
_KEY = "annotations"


# --- internal helpers ---------------------------------------------------

def _store() -> dict[str, dict[str, Any]]:
    """Return the annotations dict, initialising it if necessary."""
    return st.session_state.setdefault(_KEY, {})


# --- public API ---------------------------------------------------------

def hash_bytes(b: bytes) -> str:
    """SHA-256 hex digest of `b` — used as the content_hash."""
    return hashlib.sha256(b).hexdigest()


def get(filename: str) -> dict[str, Any]:
    """Return the annotation dict for `filename` (empty if unseen)."""
    return _store().get(filename, {})


def update(filename: str, *, content_hash: str, **fields: Any) -> None:
    """Create or update the annotation for `filename`.

    Always records / refreshes content_hash. Any further keyword
    arguments are written verbatim into the entry — `oxidation_state`
    and `spin_state` today; can grow later.
    """
    entry = _store().setdefault(filename, {})
    entry["content_hash"] = content_hash
    entry.update(fields)


def find_by_hash(content_hash: str,
                 *, exclude_filename: str | None = None) -> str | None:
    """Return the filename of a *different* upload with the same content,
    or None. Useful for the 'you've seen this CIF under another name'
    warning.
    """
    for name, entry in _store().items():
        if name == exclude_filename:
            continue
        if entry.get("content_hash") == content_hash:
            return name
    return None


def all_annotations() -> dict[str, dict[str, Any]]:
    """Read-only snapshot of every stored annotation."""
    return dict(_store())
