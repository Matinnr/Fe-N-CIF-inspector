# Testing the single-CIF dashboard

## How to run the tests

From the project root:

```bash
# from the repo root, using the CCDC interpreter
~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python -m pytest tests/ -v
```

The full suite is 209 tests (~10 s). An excerpt of the run output:

```
tests/test_cif_reader.py::test_reads_valid_cif                        PASSED
tests/test_cif_reader.py::test_temperature_extracted                  PASSED
tests/test_cif_reader.py::test_temperature_missing_does_not_crash     PASSED
tests/test_cif_reader.py::test_missing_file_raises                    PASSED
tests/test_fe_n_analysis.py::test_returns_dataframe                   PASSED
tests/test_fe_n_analysis.py::test_expected_columns_present            PASSED
tests/test_fe_n_analysis.py::test_distances_are_numeric               PASSED
tests/test_fe_n_analysis.py::test_six_distances_for_octahedral        PASSED
tests/test_fe_n_analysis.py::test_detection_method_values_valid       PASSED
tests/test_fe_n_analysis.py::test_no_iron_returns_empty_with_warning  PASSED
tests/test_fe_n_analysis.py::test_iron_no_nitrogen_returns_empty_with_warning  PASSED
tests/test_fe_n_analysis.py::test_no_temperature_does_not_crash       PASSED
tests/test_fe_n_analysis.py::test_atom_finders                        PASSED
tests/test_fe_n_analysis.py::test_no_silent_spin_state_assignment     PASSED
tests/test_fe_n_analysis.py::test_cutoff_filters_geometric_candidates PASSED
```

## What each test checks

| Test | Purpose |
|---|---|
| `test_reads_valid_cif` | The reader returns a `CifBundle` with crystal + molecule populated. |
| `test_temperature_extracted` | `_diffrn_ambient_temperature` from the CIF header is read correctly (100 K). |
| `test_temperature_missing_does_not_crash` | A CIF without a temperature key returns `None` and a warning, never raises. |
| `test_missing_file_raises` | Pointing at a non-existent file raises `FileNotFoundError`. |
| `test_returns_dataframe` | Analyser returns a `pandas.DataFrame`. |
| `test_expected_columns_present` | All canonical columns from `data_schema.BOND_TABLE_COLUMNS` are present. |
| `test_distances_are_numeric` | `distance_A` is a numeric pandas dtype, not a string. |
| `test_six_distances_for_octahedral` | The synthetic FeN6 fixture yields exactly 6 Fe–N rows (no duplication between formal bonds and symmetry contacts). |
| `test_detection_method_values_valid` | Every `detection_method` cell is one of `formal_bond`, `geometric_candidate`, `symmetry_contact`, `unknown`. |
| `test_no_iron_returns_empty_with_warning` | A CIF with no Fe atoms produces an empty table and a "no Fe" warning. |
| `test_iron_no_nitrogen_returns_empty_with_warning` | A CIF with Fe but no N produces an empty table and a "no N" warning. |
| `test_no_temperature_does_not_crash` | Analysis completes even without a temperature in the CIF. |
| `test_atom_finders` | `find_fe_atoms()` and `find_n_atoms()` return the right counts and types. |
| `test_no_silent_spin_state_assignment` | The output DataFrame contains *no* `spin_state` or `oxidation_state` columns — those are user inputs only. |
| `test_cutoff_filters_geometric_candidates` | Tightening the cutoff reduces the number of geometric candidates but doesn't drop formal bonds. |

## How to interpret failures

### `assert None == 100.0` (temperature)
The CIF parser couldn't find `_diffrn_ambient_temperature`. Check:

- The CIF actually contains a temperature line.
- The line isn't broken across multiple physical lines (CIFs sometimes use line continuations; our parser only reads single lines).
- The value parses as a float (e.g. `100(1)` is fine, `room_temperature` is not).

### `assert 12 == 6` or other count mismatches
The deduplication between formal bonds and symmetry contacts may have broken. The dedup key is `(fe_label, n_label, round(distance, 2))` — if the CCDC API starts returning labels with different casing or symmetry-prefixed names, this breaks. Look at the actual contents of `result.bonds` (use `print(result.bonds)` in the test) to diagnose.

### `ImportError: dlopen ... libssl/libcrypto` or `Symbol not found: _png_create_info_struct`
You hit one of the macOS dylib load-order bugs. Make sure:

1. `streamlit` is imported **before** `ccdc` — every test file starts with `import streamlit  # noqa: F401`.
2. You're not also importing `matplotlib` in tests; we use Plotly only.

### `ccdc not importable: ...`
You're running with the wrong Python. Use:

```
~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python
```

A quick check:
```bash
~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python -c "import ccdc; print(ccdc.__version__)"
```

### `DeprecationWarning` from a third-party library
Ignore — those are noise from streamlit / pandas / plotly internals. The tests pass on real failures only, not warnings.

## Adding a new test

1. Create a CIF fixture under `tests/fixtures/` — keep it tiny (just enough atoms to demonstrate the case).
2. Add a fixture function in `tests/conftest.py` returning the path.
3. Write a `test_*` function in `tests/test_fe_n_analysis.py` (or a new file).
4. Re-run pytest.

The tests are intentionally small and offline. They never touch the real CSD database.
