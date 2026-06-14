"""
CIF reading layer.

Responsibilities:
  - Open a CIF and return its first crystal/molecule.
  - Pull metadata that lives in the CIF header (temperature, formula).
  - Surface human-readable warnings about disorder, multiple data blocks,
    or missing chemistry — but never crash on them.

Beginner notes:
  - `dataclass` lets us bundle several values together with a name and
    type hints, instead of returning a dict or a tuple. Easier to read.
  - Type hints (`-> CifBundle`) don't change runtime behaviour, they just
    document what a function returns.
  - The only dependency on the CCDC API is in this file. If you ever swap
    in another CIF parser, only this file changes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# We import ccdc lazily inside the function rather than at module top.
# That way:
#   (a) tests that don't actually parse a CIF can still import this module,
#   (b) we don't pay the ccdc startup cost unless we have to.
def _load_ccdc():
    """Try to import the CCDC io module; raise RuntimeError if it fails."""
    try:
        from ccdc import io as ccdc_io
        return ccdc_io
    except Exception as exc:                    # noqa: BLE001
        raise RuntimeError(
            "The CCDC Python API (ccdc) is not importable from this Python.\n"
            "Run with the CCDC interpreter, e.g.:\n"
            "  ~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python -m streamlit ...\n"
            f"Underlying error: {exc!r}"
        ) from exc


@dataclass
class Provenance:
    """Bibliographic + refinement metadata pulled from the CIF header.

    All fields default to None when the corresponding CIF tag is absent
    — the dashboard renders '—' for those so the layout doesn't shift
    between files with rich and sparse metadata.

    Numerical R-factors are stored as fractions (0.0774, not 7.74) so
    consumers can decide their own formatting. Conversion to percent
    happens at the rendering layer.
    """
    # Identification
    ccdc_refcode: str | None = None        # e.g. "CCDC 254204"
    chemical_formula: str | None = None    # e.g. "C56 H62.67 Fe N8 O1.33"
    space_group: str | None = None         # e.g. "P -1"
    Z: int | None = None                   # _cell_formula_units_Z
    # Z′ is intentionally not computed: it depends on the general-
    # position multiplicity of the space group, for which we'd need a
    # full lookup table. Left as None — rendered as '—' in the UI.
    Z_prime: float | None = None
    # Refinement
    R_factor_gt: float | None = None       # _refine_ls_R_factor_gt
    R_factor_all: float | None = None      # _refine_ls_R_factor_all
    wR_factor_ref: float | None = None     # _refine_ls_wR_factor_ref
    wR_factor_gt: float | None = None      # _refine_ls_wR_factor_gt
    goodness_of_fit: float | None = None   # _refine_ls_goodness_of_fit_ref
    n_reflns: int | None = None            # _refine_ls_number_reflns
    n_parameters: int | None = None        # _refine_ls_number_parameters
    # Publication
    doi: str | None = None                 # _journal_paper_doi or _citation_doi


@dataclass
class DisorderAtom:
    """One atom that occupies its crystallographic site only fractionally.

    Captures the three pieces of information a chemist needs to make
    sense of a disorder model:
      label              the atom label as written in the CIF
      occupancy          fractional site occupancy in [0, 1]
      disorder_assembly  the group of competing partial-occupancy
                         models the atom belongs to (e.g. 'A')
      disorder_group     the specific component within the assembly
                         (e.g. '1' for major, '2' for minor)
    """
    label: str
    occupancy: float
    disorder_assembly: str | None = None
    disorder_group: str | None = None


@dataclass
class CifBundle:
    """Everything we need from one CIF, packaged together."""
    crystal: Any                       # ccdc.crystal.Crystal
    molecule: Any                      # ccdc.molecule.Molecule
    structure_id: str                  # crystal.identifier or filename stem
    temperature_K: float | None        # None if not in CIF
    warnings: list[str] = field(default_factory=list)
    # Lookup of esd values from the _geom_bond_distance loop in the CIF
    # text. Key is a frozenset of two atom labels {Fe1, N3}; value is
    # (length, esd) — esd None when the CIF doesn't supply one. Empty
    # when the CIF has no _geom_bond loop. We do the text parse once
    # here so the temp-file path can be freed by the caller before
    # analyse() runs (in the Streamlit upload flow the path is
    # deleted immediately after read_cif returns).
    bond_esds: dict = field(default_factory=dict)
    # Per-atom disorder records — populated only for atoms whose
    # occupancy < 1.0. Empty when the structure has no disorder.
    disorder_atoms: list[DisorderAtom] = field(default_factory=list)
    # Bibliographic / refinement metadata for the Provenance expander.
    # All fields default to None; the parser fills whatever is present.
    provenance: Provenance | None = None


def extract_temperature(cif_path: Path | str) -> float | None:
    """Return the data-collection temperature in Kelvin, or None.

    Reads the raw CIF text and looks for these keys, in order:
      _diffrn_ambient_temperature
      _cell_measurement_temperature

    Returns None if neither is present or parseable. NEVER returns a
    silent default (300 K, room temperature, etc).

    Why we parse the file ourselves rather than asking the CCDC API:
    `crystal.attributes` does not expose raw CIF tags in the version of
    `ccdc` shipped with the CSD Portfolio, and the API path varies
    across releases. A line-by-line text scan is robust and version-free.
    """
    candidates = (
        "_diffrn_ambient_temperature",
        "_cell_measurement_temperature",
    )
    try:
        text = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for line in text.splitlines():
        # CIF tags start with an underscore; we only need the first two
        # whitespace-separated tokens of each line.
        stripped = line.strip()
        if not stripped.startswith("_"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        tag, value = parts[0], parts[1].strip().strip("'\"")
        if tag in candidates:
            # CIF values can be e.g. "100.00(1)" — strip any "(esd)" suffix.
            value = value.split("(")[0]
            try:
                return float(value)
            except ValueError:
                continue
    return None


def parse_geom_bond_distances(
    cif_path: Path | str,
) -> dict[frozenset[str], tuple[float, float | None]]:
    """Extract distances+esds from the `_geom_bond_distance` loop.

    The CCDC API exposes `bond.length` as a plain float — no esd. The
    esd lives in the original CIF text as e.g. ``1.976(8)``. We do a
    minimal targeted scan to pull those tokens out so the bond table
    can show standard uncertainties.

    Only intra-asymmetric-unit bonds are recorded (symmetry tag "."
    or absent). Atom pairs are keyed by frozenset so order doesn't
    matter on lookup.

    Returns an empty dict if the CIF has no `_geom_bond` loop, or
    couldn't be read.
    """
    try:
        text = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    from .esd import parse_value_with_esd

    out: dict[frozenset[str], tuple[float, float | None]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        # Collect tag lines of this loop.
        j = i + 1
        tags: list[str] = []
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("_"):
                tags.append(s.split()[0])
                j += 1
            elif not s or s.startswith("#"):
                j += 1
            else:
                break

        required = (
            "_geom_bond_atom_site_label_1",
            "_geom_bond_atom_site_label_2",
            "_geom_bond_distance",
        )
        if not all(t in tags for t in required):
            i = j
            continue

        col_l1 = tags.index("_geom_bond_atom_site_label_1")
        col_l2 = tags.index("_geom_bond_atom_site_label_2")
        col_d  = tags.index("_geom_bond_distance")
        col_sym = (
            tags.index("_geom_bond_site_symmetry_2")
            if "_geom_bond_site_symmetry_2" in tags else None
        )

        # Parse data rows of this loop until we hit a non-data line.
        k = j
        while k < len(lines):
            row = lines[k].strip()
            if not row or row.startswith("_") or row == "loop_":
                break
            if row.startswith("#"):
                k += 1
                continue
            parts = row.split()
            if len(parts) < len(tags):
                k += 1
                continue
            sym = parts[col_sym] if col_sym is not None else "."
            # Only same-asymmetric-unit bonds (sym is "." or "?").
            if sym not in (".", "?"):
                k += 1
                continue
            try:
                value, esd = parse_value_with_esd(parts[col_d])
                out[frozenset([parts[col_l1], parts[col_l2]])] = (value, esd)
            except (ValueError, IndexError):
                pass
            k += 1
        i = k
    return out


def parse_provenance(cif_path: Path | str) -> Provenance:
    """Extract bibliographic + refinement metadata from a CIF.

    All fields default to None. The DOI priority is:
      1. _journal_paper_doi   (the canonical publication tag)
      2. _citation_doi        (first row of any citation loop)
      3. _audit_block_doi     (fallback — sometimes the CCDC archive
                              DOI rather than the paper; used only if
                              nothing better is present)
    """
    prov = Provenance()
    try:
        text = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return prov

    # --- Single-line tag → (attribute, conversion fn) ---------------
    # The string conversion preserves quoting-stripped raw text.
    # The numeric conversions strip trailing esd parens.
    def _to_int(s: str) -> int:
        return int(float(s.split("(")[0]))

    def _to_float(s: str) -> float:
        return float(s.split("(")[0])

    def _strip_str(s: str) -> str:
        return s.strip().strip("'\"")

    SIMPLE_TAGS: dict[str, tuple[str, callable]] = {
        "_database_code_depnum_ccdc_archive": ("ccdc_refcode",     _strip_str),
        "_database_code_CSD":                 ("ccdc_refcode",     _strip_str),
        "_chemical_formula_sum":              ("chemical_formula", _strip_str),
        "_symmetry_space_group_name_H-M":     ("space_group",      _strip_str),
        "_space_group_name_H-M_alt":          ("space_group",      _strip_str),
        "_cell_formula_units_Z":              ("Z",                _to_int),
        "_refine_ls_R_factor_gt":             ("R_factor_gt",      _to_float),
        "_refine_ls_R_factor_all":            ("R_factor_all",     _to_float),
        "_refine_ls_wR_factor_ref":           ("wR_factor_ref",    _to_float),
        "_refine_ls_wR_factor_gt":            ("wR_factor_gt",     _to_float),
        "_refine_ls_goodness_of_fit_ref":     ("goodness_of_fit",  _to_float),
        "_refine_ls_number_reflns":           ("n_reflns",         _to_int),
        "_refine_ls_number_parameters":       ("n_parameters",     _to_int),
        "_journal_paper_doi":                 ("doi",              _strip_str),
    }

    # Track which fields the user supplied so alt tags / fallbacks
    # don't overwrite better data.
    seen: set[str] = set()
    audit_block_doi: str | None = None

    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("_"):
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        tag, raw = parts[0], parts[1]
        if tag == "_audit_block_doi":
            # Stash separately; we use it only if neither
            # _journal_paper_doi nor _citation_doi shows up.
            try:
                audit_block_doi = _strip_str(raw)
            except Exception:                                # noqa: BLE001
                pass
            continue
        if tag not in SIMPLE_TAGS:
            continue
        attr, conv = SIMPLE_TAGS[tag]
        if attr in seen:
            continue   # earlier tag already populated this field
        try:
            setattr(prov, attr, conv(raw))
            seen.add(attr)
        except (ValueError, TypeError):
            pass

    # --- _citation_doi inside a loop -------------------------------
    # Only used when _journal_paper_doi wasn't present.
    if "doi" not in seen:
        citation_doi = _parse_citation_doi_loop(lines)
        if citation_doi is not None:
            prov.doi = citation_doi
            seen.add("doi")

    # --- Final audit-block fallback --------------------------------
    if "doi" not in seen and audit_block_doi is not None:
        prov.doi = audit_block_doi

    return prov


def _parse_citation_doi_loop(lines: list[str]) -> str | None:
    """Find the first _citation_doi value in any loop_ block."""
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        tags: list[str] = []
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("_"):
                tags.append(s.split()[0])
                j += 1
            elif not s or s.startswith("#"):
                j += 1
            else:
                break
        if "_citation_doi" not in tags:
            i = j
            continue
        col = tags.index("_citation_doi")
        for k in range(j, len(lines)):
            row = lines[k].strip()
            if not row or row.startswith("_") or row == "loop_":
                break
            if row.startswith("#"):
                continue
            parts = row.split()
            if len(parts) > col:
                value = parts[col].strip().strip("'\"")
                # CIFs sometimes use '?' or '.' for missing values.
                if value not in ("", "?", "."):
                    return value
        i = j
    return None


def parse_disorder_atoms(cif_path: Path | str) -> list[DisorderAtom]:
    """Extract per-atom disorder info from the `_atom_site` loop.

    Returns one DisorderAtom record per atom with occupancy < 1.0.
    Atoms at full occupancy (or missing the column entirely) are
    omitted from the list — they don't need disorder handling.

    Disorder assembly / group default to None when the CIF doesn't
    record them (older CIFs frequently omit these tags).
    """
    try:
        text = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    out: list[DisorderAtom] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j = i + 1
        tags: list[str] = []
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith("_"):
                tags.append(s.split()[0])
                j += 1
            elif not s or s.startswith("#"):
                j += 1
            else:
                break

        # We need at least label + occupancy to do anything useful.
        if ("_atom_site_label" not in tags
                or "_atom_site_occupancy" not in tags):
            i = j
            continue

        col_label    = tags.index("_atom_site_label")
        col_occ      = tags.index("_atom_site_occupancy")
        col_assembly = (
            tags.index("_atom_site_disorder_assembly")
            if "_atom_site_disorder_assembly" in tags else None
        )
        col_group    = (
            tags.index("_atom_site_disorder_group")
            if "_atom_site_disorder_group" in tags else None
        )

        k = j
        while k < len(lines):
            row = lines[k].strip()
            if not row or row.startswith("_") or row == "loop_":
                break
            if row.startswith("#"):
                k += 1
                continue
            parts = row.split()
            if len(parts) < len(tags):
                k += 1
                continue
            try:
                occ_str = parts[col_occ].split("(")[0]  # strip esd
                occ = float(occ_str)
            except (ValueError, IndexError):
                k += 1
                continue
            if occ < 1.0:
                def _opt(idx):
                    if idx is None:
                        return None
                    v = parts[idx]
                    return None if v in (".", "?") else v
                out.append(DisorderAtom(
                    label=parts[col_label],
                    occupancy=occ,
                    disorder_assembly=_opt(col_assembly),
                    disorder_group=_opt(col_group),
                ))
            k += 1
        i = k
    return out


def has_disorder(molecule: Any) -> bool:
    """Return True if any atom has occupancy < 1.0 (disordered site)."""
    for atom in getattr(molecule, "atoms", []):
        occ = getattr(atom, "occupancy", 1.0)
        if occ is not None and occ < 1.0:
            return True
    return False


def read_cif(path: Path | str) -> CifBundle:
    """Parse a CIF and return a CifBundle.

    Parameters
    ----------
    path : Path or str
        Filesystem location of the CIF file.

    Returns
    -------
    CifBundle
        Populated with the crystal, molecule, structure id, temperature,
        and any warnings collected during parsing.

    Raises
    ------
    FileNotFoundError
        If the CIF file does not exist on disk.
    RuntimeError
        If the CCDC API can't be imported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CIF not found: {path}")

    ccdc_io = _load_ccdc()
    warnings: list[str] = []

    reader = ccdc_io.CrystalReader(str(path))

    # Pull all data blocks; warn if more than one (rare but happens).
    crystals = list(reader)
    if not crystals:
        raise ValueError(f"No crystal data blocks found in {path.name}")
    if len(crystals) > 1:
        warnings.append(
            f"CIF contains {len(crystals)} data blocks; using only the first."
        )

    crystal = crystals[0]
    molecule = crystal.molecule
    structure_id = crystal.identifier or path.stem

    if has_disorder(molecule):
        warnings.append("Disorder detected (some atom occupancies < 1.0).")

    temperature = extract_temperature(path)
    if temperature is None:
        warnings.append("No temperature found in CIF header.")

    # Pre-parse the _geom_bond_distance loop so analyse() can attach
    # esds to each formal bond. Empty if absent from this CIF.
    bond_esds = parse_geom_bond_distances(path)

    # Pre-parse disorder atoms from the _atom_site loop. Empty when
    # all atoms are at full occupancy.
    disorder_atoms = parse_disorder_atoms(path)

    # Pre-parse bibliographic + refinement metadata. All fields may
    # be None; the Provenance UI shows '—' for missing ones.
    provenance = parse_provenance(path)

    return CifBundle(
        crystal=crystal,
        molecule=molecule,
        structure_id=structure_id,
        temperature_K=temperature,
        warnings=warnings,
        bond_esds=bond_esds,
        disorder_atoms=disorder_atoms,
        provenance=provenance,
    )
