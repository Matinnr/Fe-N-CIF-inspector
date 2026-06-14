# Python learning guide — using this dashboard as a textbook

You can treat this small project as a real-world introduction to:

- **pandas** (tables of data),
- **plotly** (interactive plots),
- **streamlit** (turning Python into a web dashboard),
- **dataclasses & enums** (clean data structures),
- **pytest** (automated tests).

This guide walks through every file, every function, and gives you
small exercises to practise.

---

## 1. The big picture — what each file does

```
single_cif_dashboard/
├── app.py                       ← The Streamlit UI. The user-facing layer.
├── src/
│   ├── data_schema.py           ← Constants. Column names, detection labels.
│   ├── cif_reader.py            ← Reads a CIF, returns a Python object.
│   ├── fe_n_analysis.py         ← Turns the object into a pandas table.
│   └── plotting.py              ← Turns the table into Plotly figures.
├── tests/                       ← Pytest tests for the src/ modules.
├── README.md                    ← How to run.
├── TESTING.md                   ← How to run the tests.
└── PYTHON_LEARNING_GUIDE.md     ← This file.
```

Reading order if you want to learn:
1. `data_schema.py` — easiest, ~50 lines, just constants.
2. `cif_reader.py` — file handling, dataclasses, error handling.
3. `fe_n_analysis.py` — the heart of the project. Build the table.
4. `plotting.py` — turn the table into a chart.
5. `app.py` — wire everything to UI widgets.

---

## 2. What each function does

### `data_schema.py`
- `BOND_TABLE_COLUMNS` — the canonical list of column names. Every
  module that builds or reads the bond table imports this.
- `DetectionMethod` — an `Enum` with four values:
  `FORMAL_BOND, GEOMETRIC_CANDIDATE, SYMMETRY_CONTACT, UNKNOWN`.

### `cif_reader.py`
- `read_cif(path)` — opens a CIF, returns a `CifBundle` (a dataclass).
- `extract_temperature(path)` — returns the temperature in Kelvin or
  `None` if the CIF doesn't have it.
- `has_disorder(molecule)` — returns `True` if any atom has occupancy
  < 1.0.

### `fe_n_analysis.py`
- `find_fe_atoms(molecule)` — list of Fe atoms.
- `find_n_atoms(molecule)` — list of N atoms.
- `analyse(bundle, cif_filename, cutoff_A, …)` — returns an
  `AnalysisResult`:
  - `bonds: pd.DataFrame` — the table.
  - `warnings: list[str]` — messages for the UI.
  - `n_fe`, `n_n` — atom counts.

### `plotting.py`
- `summary_cards(bonds, n_fe)` — returns a dict of pre-formatted strings
  for the metric strip at the top.
- `lollipop(bonds, show_spin_bands)` — returns a Plotly Figure with one
  stem per Fe–N distance.

### `app.py`
- (top-level Streamlit code — runs every time the user moves a widget)
- `cached_read(cif_bytes, original_name)` — wraps `read_cif` with
  Streamlit's `@st.cache_data` so re-running doesn't re-parse the CIF.

---

## 3. What's a pandas DataFrame?

A `DataFrame` is **a table in memory**. Columns have names. Rows are
indexed by integer (or anything else you choose).

```python
import pandas as pd

df = pd.DataFrame([
    {"name": "Alice", "age": 30, "city": "Paris"},
    {"name": "Bob",   "age": 25, "city": "Tokyo"},
])

df["age"]                    # Series (one column)
df["age"].mean()             # 27.5
df[df["age"] > 26]           # filter rows
df.groupby("city")["age"].mean()  # group + aggregate
```

In this project the DataFrame holds one row per Fe–N distance. Each
column is one of the strings in `BOND_TABLE_COLUMNS`.

The pandas idiom we use most:

```python
rows = []                                # build a list of dicts
for ...:
    rows.append({"col_a": ..., "col_b": ...})
df = pd.DataFrame(rows, columns=BOND_TABLE_COLUMNS)
```

This is much faster than appending rows one at a time.

---

## 4. How a CIF becomes a table

1. **Streamlit** receives the upload as raw bytes.
2. **`cached_read`** writes those bytes to a temp file, calls
   **`read_cif`**, deletes the temp file.
3. **`read_cif`** uses **`ccdc.io.CrystalReader`** to parse the CIF and
   returns a `CifBundle` (a dataclass holding the crystal, molecule,
   structure id, temperature, warnings).
4. **`analyse`** iterates `mol.bonds`, `mol.atoms`, and (optionally)
   `crystal.contacts()` — building one Python dict per Fe–N distance.
5. **`pd.DataFrame(rows, columns=...)`** converts the list of dicts
   into a table.

The whole pipeline is: **bytes → file → ccdc objects → list of dicts →
DataFrame**.

---

## 5. How the table becomes a plot

`plotting.lollipop(df)` does this in three steps:

```python
fig = go.Figure()                                # blank canvas

for method, sub in df.groupby("detection_method"):
    fig.add_trace(go.Scatter(
        x=sub["pair_label"],
        y=sub["distance_A"],
        mode="markers",
        name=method,
    ))

fig.update_layout(title="Fe–N distances",
                  yaxis_title="distance (Å)")
return fig
```

A Plotly `Figure` is just a Python object. Streamlit knows how to
render it with `st.plotly_chart(fig)`.

---

## 6. How Streamlit turns Python into a dashboard

Streamlit re-runs `app.py` from top to bottom every time the user
changes a widget. To learn the basics, look at these patterns from
`app.py`:

```python
import streamlit as st

st.title("My app")                       # heading
x = st.slider("Pick", 1, 10, 5)          # widget; returns the value
st.write(f"You picked {x}")              # text output
st.dataframe(df)                         # table widget
st.plotly_chart(fig)                     # chart widget
st.download_button("CSV", df.to_csv())   # download
```

Things you don't need to think about:
- HTML / CSS / JavaScript.
- A web server — `streamlit run app.py` does it.
- State across reruns — Streamlit takes care of widgets; for expensive
  work, decorate with `@st.cache_data`.

---

## 7. Running one function manually in a Python shell

You don't have to launch Streamlit to play with the analysis code. Open
the CCDC Python and import directly:

```bash
cd ~/Python\ /fen_dashboard/single_cif_dashboard
~/CCDC/ccdc-software/csd-python-api/miniconda/bin/python
```

```python
>>> import sys; sys.path.insert(0, ".")
>>> from src.cif_reader import read_cif
>>> from src.fe_n_analysis import analyse
>>> bundle = read_cif("tests/fixtures/fe_octahedral.cif")
>>> bundle.structure_id
'FeN6_test'
>>> bundle.temperature_K
100.0
>>> result = analyse(bundle, cif_filename="fe_octahedral.cif")
>>> result.bonds.head()
            cif_file structure_id  temperature_K Fe_label N_label  distance_A detection_method  symmetry_related warning
0  fe_octahedral.cif     FeN6_test          100.0      Fe1      N1         2.0       formal_bond              False
...
```

Press Ctrl+D (or type `exit()`) to leave the shell.

---

## 8. Things to change to practise

### Easy
1. **Change the lollipop title.** Edit `plotting.py` and find
   `title="Fe–N distances"`. Replace it. Re-launch Streamlit.

2. **Change the cutoff default.** In `app.py`, find `value=2.7` in the
   slider and change it.

3. **Add a custom cell colour.** In `plotting.py`'s `METHOD_COLOURS`
   dict, swap the colours.

### Medium
4. **Add a column.** Track *coordination number* (count of Fe–N bonds
   per Fe centre). In `fe_n_analysis.py`:
   - Add `COL_COORD_NUM = "coordination_number"` to
     `data_schema.py` and append it to `BOND_TABLE_COLUMNS`.
   - In `analyse()`, after the dataframe is built but before returning,
     map each Fe label to the count of rows for that Fe and assign:
     ```python
     df[COL_COORD_NUM] = df[COL_FE_LABEL].map(df[COL_FE_LABEL].value_counts())
     ```

5. **Add a "mean distance per Fe" calculation** to the summary cards.
   Edit `plotting.summary_cards()` and add a key `mean_per_fe` returning
   the average over Fe-grouped means. Wire it into `app.py`.

6. **Reduce noise.** Add a sidebar checkbox "hide warnings about
   geometric candidates" and skip those warnings when ticked.

### Harder
7. **Export a summary CSV** alongside the full bond CSV. Use
   `df.groupby("Fe_label").agg(...)` and offer a second download
   button.

8. **Read multiple CIFs** if the user uploads them. Keep the
   single-CIF default but, when more than one CIF is uploaded,
   concatenate the per-CIF tables and add a `cif_file` column already
   populated.

9. **Support a different metal.** Generalise `find_fe_atoms` /
   `find_n_atoms` to take an element symbol argument, and add a
   sidebar selector for the metal centre.

10. **Write a new test.** Pick one of the changes above and verify it
    in `tests/test_fe_n_analysis.py`.

---

## Cheat sheet

| You want to … | Do … |
|---|---|
| Read a CIF | `from src.cif_reader import read_cif; bundle = read_cif(path)` |
| Analyse it | `from src.fe_n_analysis import analyse; r = analyse(bundle, cif_filename=name)` |
| Show the table | `st.dataframe(r.bonds)` |
| Save as CSV | `r.bonds.to_csv("out.csv", index=False)` |
| Run tests | `python -m pytest tests/ -v` |
| Launch app | `python -m streamlit run app.py` |
