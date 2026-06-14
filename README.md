# Fe–N CIF inspector

An interactive crystallographic-analysis dashboard for iron coordination
compounds, focused on the Fe–N coordination sphere and the structural
fingerprints of spin-crossover (SCO) chemistry. Built on Streamlit +
Plotly + the CCDC CSD Python API.

Drop one CIF or a batch of CIFs, and the tool produces:

- Every Fe–N distance labelled by detection method (formal bond /
  geometric candidate / symmetry-generated contact)
- Octahedral distortion parameters **ζ, Δ, Σ, Θ** in the OctaDist
  (Ketkaew 2021) convention
- Coordination-geometry classification — **octahedral**, **τ₅** for
  5-coordinate, **τ₄** for 4-coordinate
- **Bond-valence sums** with the spin-state-specific Liebschner 2017
  R₀ parameters and editable values for chemists who know their
  ligand class
- **Refinement-quality badge** computed from R-factor + data/parameter
  ratio
- Standard uncertainties (`1.984(7)` notation) parsed from
  `_geom_bond_distance` and propagated to the mean
- Per-structure provenance, disorder handling, and cohort plots

## Design principles

- **Annotations beat suggestions.** The dashboard *will* suggest
  oxidation and spin states from BVS and bond-length heuristics —
  it'd be unhelpful not to, given how informative those numbers are
  — but your sidebar selectboxes are the source of truth. When you
  set them, they override every suggestion downstream. When you
  don't, the heuristic value is shown with its confidence band so
  the guesswork is visible, not hidden.
- **The four BVS combinations are always shown together.** Rather
  than picking one "best" combination silently, the inference panel
  renders all four (Fe(II)/Fe(III)) × (LS/HS) results with their
  bond-length plausibility. The recommended pick is the top row, not
  the only row.
- **BVS isn't an absolute oxidation indicator.** The Liebschner R₀
  values are calibrated against simple hexa-amine Fe systems and
  systematically over-count for porphyrins by ~1.5 valence units.
  The editable-R₀ panel and the porphyrin caveat surface this
  explicitly rather than hiding it.
- **CIFs in the reference library are CC0 only.** Bundled
  structures come from the Crystallography Open Database; CSD-derived
  CIFs aren't redistributed.
- **Everything happens locally.** No network calls.

## Quick start

Requires Python 3.11 and the CCDC Python API (which ships its own
Python interpreter; the dashboard runs against that interpreter).

```bash
# Confirm the CCDC interpreter is available
ls ~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python

# Install the dashboard's pip dependencies into the CCDC env
~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python -m pip install \
    streamlit plotly pytest pytest-cov

# Launch
~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python -m streamlit \
    run app.py
```

Open <http://localhost:8501> in your browser.

Or simply `make demo`.

## The three modes

### Mode 1 — Single CIF

Upload one CIF, inspect it in detail:

- 5-card summary strip (Fe centres, n_FeN, mean, min, max)
- Per-bond table with detection method and esd notation
- Interactive lollipop chart of every Fe–N distance
- Geometry panel with τ classification and the four distortion params
- BVS card with status badge against the annotated oxidation state
- Disorder panel with optional side-by-side mean comparison
- Provenance expander with refinement-quality badge

### Mode 2 — Batch / cohort

Drop many CIFs at once for cross-structure analysis:

- Per-Fe-centre cohort table with auto-grouping by chemical formula
- 2×2 Plotly scatter grid: ⟨Fe–N⟩ vs T, Σ vs ⟨Fe–N⟩, Θ vs Σ, BVS vs ⟨Fe–N⟩
- Inline editor for oxidation, spin, and series
- BVS recomputes per row on annotation change
- CSV download of the cohort

### Mode 3 — Reference library

A small curated set of teaching CIFs:

- Each entry: identification, expected metrics, teaching notes
- "Compare two references" side-by-side view
- Add new entries by dropping CIFs into `data/reference/` and
  registering them in `library.json` — no code change

## Project layout

```
fe-n-cif-inspector/
├── app.py                      ← Streamlit entry point (thin router)
├── Makefile                    ← demo / test / coverage targets
├── src/
│   ├── annotations.py          ← session-state annotation store
│   ├── bvs.py                  ← bond-valence sums (Brown/Liebschner R₀)
│   ├── cif_reader.py           ← CIF I/O, esds, disorder, provenance parser
│   ├── cohort.py               ← cohort table construction
│   ├── data_schema.py          ← canonical column names, spin bands, R₀ refs
│   ├── esd.py                  ← parse/format/propagate standard uncertainties
│   ├── fe_n_analysis.py        ← bond extraction + per-Fe analysis pipeline
│   ├── geometry.py             ← ζ, Δ, Σ, Θ, τ₅, τ₄, trans angles
│   ├── library.py              ← reference library loader + validation
│   ├── plotting.py             ← Plotly figure builders
│   └── modes/
│       ├── single_cif.py       ← Mode 1
│       ├── batch.py            ← Mode 2
│       └── reference_library.py ← Mode 3
├── tests/                      ← 209 tests across the analysis modules
├── data/reference/             ← curated library + library.json
└── docs/methodology.md         ← formulas, references, worked examples
```

## Scientific references

The metrics implemented and the literature they're calibrated against:

| Metric | Source |
|---|---|
| ζ, Δ, Σ definitions | McCusker et al., *Inorg. Chem.* **35** (1996) 2100 |
| Θ definition | Marchivie et al., *Acta Cryst. B* **61** (2005) 25 |
| Θ algorithm + face pairing | Ketkaew et al., *Dalton Trans.* **50** (2021) 1086 (OctaDist tool) |
| τ₅ (Addison parameter) | Addison et al., *J. Chem. Soc. Dalton Trans.* (1984) 1349 |
| τ₄ (Yang parameter) | Yang et al., *Dalton Trans.* (2007) 955 |
| BVS — original R₀ | Brown & Altermatt, *Acta Cryst. B* **41** (1985) 244 |
| BVS — anion R₀ | Brese & O'Keeffe, *Acta Cryst. B* **47** (1991) 192 |
| BVS — Fe spin-state-specific R₀ | Liebschner et al., *Acta Cryst. D* **73** (2017) 148 |
| SCO context, structural bands | Halcrow, M. A., *Chem. Soc. Rev.* **40** (2011) 4119 |

A more detailed methodology walkthrough lives in
[`docs/methodology.md`](docs/methodology.md), with LaTeX formulas and
worked examples for each metric.

## Tests

```bash
make test            # 209 tests, ~10 s
make coverage        # with line-by-line coverage report
```

Coverage on the analysis modules:

| Module | Coverage |
|---|---|
| `bvs.py`, `esd.py`, `plotting.py` | **100%** |
| `cohort.py`, `library.py`, `data_schema.py` | **96–99%** |
| `geometry.py` | **92%** |
| `cif_reader.py` | **85%** |
| `fe_n_analysis.py` | 73% |

The Streamlit UI modules in `src/modes/` aren't unit-tested — Streamlit
widgets need a running session and the value of that coverage is low
compared to the cost. See `TESTING.md` for the full test guide.

## Limitations

- **Designed for Fe–N coordination chemistry.** The codebase will run
  on a Cu–O complex, but the BVS R₀ tables and the LS/HS reference
  bands are Fe-specific. Generalising to other metals is a focused
  ~half-day refactor.
- **Single asymmetric unit only by default.** Symmetry-generated
  contacts can be enabled per-analysis, but the CCDC API's
  symmetry-handling is conservative and won't catch every
  polymeric / bridging interaction.
- **BVS for porphyrin systems systematically overshoots.** This is a
  known limitation of the Liebschner R₀ tables, surfaced by the
  editable-R₀ teaching panel rather than hidden.
- **Θ algorithm assumes octahedral geometry exists.** Wildly distorted
  6-coordinate complexes (very far from octahedral) will produce a
  computed Θ that the algorithm is honest about, but interpreting
  the number requires care.
- **No automatic spin-state assignment.** This is a deliberate choice
  documented in the "What it deliberately does NOT do" section above.

## Citation

If you use this tool in a publication, please cite the underlying
methods (see [`CITATION.cff`](CITATION.cff) for a machine-readable
entry plus the per-method references listed above).

## License

The dashboard source code is released under the MIT License. The
reference library uses Crystallography Open Database CIFs (CC0); the
CCDC Python API itself is proprietary and requires a separate licence
from CCDC.
