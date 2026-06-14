# Methodology

Formal definitions of every metric the dashboard computes, with the
literature each is calibrated against and one worked example per
metric.

The goal of this document is to make the analysis reproducible: a
reader should be able to take any number the dashboard reports and
re-derive it by hand from the CIF.

---

## 1. Fe–N bond detection

Three detection methods are exposed:

### 1.1 Formal bond — `mol.bonds` after `assign_bond_types(which="all")`

The CCDC Python API runs an internal bond-perception algorithm that
classifies an Fe–N pair as a chemical bond if the distance is within
the expected covalent + coordinative range for the element pair. The
dashboard collects every such pair into the bond table with
`detection_method = formal_bond`.

### 1.2 Geometric candidate — within cutoff, not a formal bond

For every Fe–N pair *not* classified as a formal bond, the dashboard
checks the Euclidean distance:

$$
d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}
$$

(Cartesian coordinates, obtained from `atom.coordinates` after the
CCDC API converts fractional → Cartesian via the cell matrix.)

If $d_{ij} \le \text{cutoff\_A}$ (default 2.7 Å) the pair is recorded
with `detection_method = geometric_candidate`. This catches long
coordinative bonds that the chemistry algorithm classified as
non-bonding.

### 1.3 Symmetry-generated contact — `crystal.contacts()`

Calls the CCDC API's `crystal.contacts(distance_range=(0.1, cutoff_A),
intermolecular="Inter")` to find Fe–N pairs that only exist after
applying space-group symmetry. Filtered with an explicit
`c.length <= cutoff_A` post-check.

---

## 2. Bond-length distortion: ζ and Δ

### 2.1 Zeta — ζ

The total absolute deviation of the six Fe–N bond lengths from their
mean:

$$
\zeta = \sum_{i=1}^{6} \lvert d_i - \langle d \rangle \rvert
\qquad (\text{Å})
$$

Defined for octahedral 6-coordinate Fe centres. The dashboard
generalises to any *n* but applies the formula identically.

**Reference**: McCusker et al., *Inorg. Chem.* **35** (1996) 2100.

**Worked example.** Elongated octahedron with one bond at 2.2 Å and
five at 2.0 Å:

$$
\langle d \rangle = \tfrac{2.2 + 5 \times 2.0}{6} = 2.0333 \text{ Å}
$$

$$
\zeta = \underbrace{\lvert 2.2 - 2.0333 \rvert}_{0.1667}
      + 5 \times \underbrace{\lvert 2.0 - 2.0333 \rvert}_{0.0333}
      = 0.333 \text{ Å}
$$

### 2.2 Delta — Δ

Dimensionless variance of the six Fe–N bond lengths:

$$
\Delta = \frac{1}{n} \sum_{i=1}^{n}
  \left( \frac{d_i - \langle d \rangle}{\langle d \rangle} \right)^2
$$

Often quoted in units of $10^{-4}$ for convenience.

**Reference**: same as ζ (McCusker 1996).

Zero for any structure with all equal Fe–N bonds, irrespective of
$\langle d \rangle$.

---

## 3. Angular distortion: Σ and Θ

### 3.1 Sigma — Σ

The sum of absolute deviations of the **12 cis N–Fe–N angles** from
the ideal 90°:

$$
\Sigma = \sum_{i=1}^{12} \lvert 90^\circ - \varphi_i \rvert
\qquad (\text{degrees})
$$

The 12 cis angles are identified by first finding the three trans
pairs (the three Fe–N pairs whose Fe–N unit vectors have the most
negative dot products), then taking every other pair of the
$\binom{6}{2} = 15$ atom pairs.

**Reference**: McCusker et al., *Inorg. Chem.* **35** (1996) 2100.

**Worked example — perfect octahedron.** All 12 cis angles are
exactly 90°, so $\Sigma = 0$ exactly.

**Worked example — `sr09` Fe1.** From the
`_geom_angle` loop in `254204.cif`:

```
N2-Fe1-N1: 89.19° (×2)   N2-Fe1-N3: 88.97° (×2)
N2-Fe1-N1: 90.81° (×2)   N2-Fe1-N3: 91.03° (×2)
N1-Fe1-N3: 91.34° (×2)   N1-Fe1-N3: 88.66° (×2)
```

$$
\Sigma = 2 \times (0.81 + 0.81 + 1.03 + 1.03 + 1.34 + 1.34)
       = 12.72^\circ
$$

The dashboard reports 12.76°; the small difference is the dashboard
computing angles directly from Cartesian coordinates rather than
reading them off the CIF.

### 3.2 Theta — Θ

The OctaDist (Ketkaew 2021) trigonal-twist parameter:

$$
\Theta = \sum_{i=1}^{24} \lvert 60^\circ - \theta_i \rvert
$$

where the 24 angles are obtained by:

1. Identifying the 8 octahedral faces (3-atom subsets containing no
   trans pair).
2. Pairing them into 4 antipodal face pairs (each pair contains all
   6 ligands between them).
3. For each face pair, projecting all 6 ligands onto the plane
   normal to the line through the two face centroids.
4. Sorting the 6 projections by polar angle around that axis.
5. Computing the 6 consecutive angular gaps (each = 60° in a
   perfect octahedron).
6. Summing $\lvert 60^\circ - \text{gap}_i \rvert$ across the
   $4 \times 6 = 24$ gaps.

**References**:
- Marchivie et al., *Acta Cryst. B* **61** (2005) 25 (definition)
- Ketkaew et al., *Dalton Trans.* **50** (2021) 1086 (algorithm)

Values are directly comparable to OctaDist output for the same
input geometry.

**Worked examples**:
- Perfect octahedron → $\Theta = 0$ exactly
- Bond-elongated octahedron (one bond 2.2 Å, five at 2.0 Å)
  → $\Theta \approx 25^\circ$ — the centroid axes shift slightly,
  even though the angles between Fe–N vectors are unchanged
- Regular trigonal prism → $\Theta \gg 500^\circ$

---

## 4. Coordination-geometry classification: τ₅ and τ₄

### 4.1 Tau-5 — for 5-coordinate centres

$$
\tau_5 = \frac{\beta - \alpha}{60^\circ}
$$

where $\beta$ is the largest L–M–L angle and $\alpha$ the
second-largest.

**Reference**: Addison et al., *J. Chem. Soc. Dalton Trans.* (1984) 1349.

| $\tau_5$ | Geometry |
|---|---|
| 0 | Ideal square pyramidal (two trans-basal angles both 180°) |
| 1 | Ideal trigonal bipyramidal (one ax–M–ax angle 180°, three eq–M–eq angles 120°) |

The dashboard classifies $\tau_5 \le 0.20$ as square pyramidal,
$\tau_5 \ge 0.80$ as trigonal bipyramidal, anything between as
"distorted".

### 4.2 Tau-4 — for 4-coordinate centres

$$
\tau_4 = \frac{360^\circ - (\alpha + \beta)}{141^\circ}
$$

where $\alpha$ and $\beta$ are the two largest L–M–L angles.

**Reference**: Yang et al., *Dalton Trans.* (2007) 955.

| $\tau_4$ | Geometry |
|---|---|
| 0 | Ideal square planar (two trans angles both 180°) |
| 1 | Ideal tetrahedral (all angles 109.47°) |

The denominator $141 = 360 - 2 \times 109.47$ normalises the
tetrahedral case to 1 exactly.

The dashboard classifies $\tau_4 \le 0.10$ as square planar,
$\tau_4 \ge 0.85$ as tetrahedral, anything between as "distorted".

---

## 5. Bond-valence sum (BVS)

$$
\text{BVS} = \sum_{i} s_i,
\qquad
s_i = \exp\!\left( \frac{R_0 - R_i}{B} \right)
$$

where $R_i$ is the *i*-th Fe–N bond length, $R_0$ is a tabulated
reference distance specific to the (metal, donor, spin) combination,
and $B = 0.37$ Å is the universal "soft-shell" parameter.

**References**:
- Brown & Altermatt, *Acta Cryst. B* **41** (1985) 244 (original
  universal $R_0$ table)
- Brese & O'Keeffe, *Acta Cryst. B* **47** (1991) 192 (anion
  extension)
- Liebschner et al., *Acta Cryst. D* **73** (2017) 148 (spin-state-
  specific $R_0$ for Fe)

### 5.1 $R_0$ values exposed in the UI

| (oxidation, spin) | $R_0$ (Å) | Source |
|---|---|---|
| Fe(II), LS | 1.78 | Liebschner 2017 |
| Fe(II), HS | 1.91 | Liebschner 2017 |
| Fe(III), LS | 1.70 | Liebschner 2017 |
| Fe(III), HS | 1.83 | Liebschner 2017 |
| Fe(II), generic | 1.769 | Brown & Altermatt 1985 |
| Fe(III), generic | 1.815 | Brese & O'Keeffe 1991 |

The "generic" values are used when the user has annotated the
oxidation state but not the spin state.

### 5.2 Consistency bands

The dashboard renders BVS with a status badge:

| $\lvert \text{BVS} - Z \rvert$ | Badge | Interpretation |
|---|---|---|
| $\le 0.4$ | ✅ Good | Consistent with the annotated oxidation state |
| $0.4 < \cdot \le 0.8$ | ⚠️ Caution | Marginal; check $R_0$, distances, coordination completeness |
| $> 0.8$ | 🚫 Warning | More consistent with a different integer; suggests mis-annotation or systematic R₀ issue |

### 5.3 Worked example — the teaching case

Fe(II) LS porphyrin with $\langle d \rangle = 1.97$ Å, $R_0 = 1.78$
(Liebschner Fe(II)-LS):

$$
s_i = \exp\!\left(\tfrac{1.78 - 1.97}{0.37}\right)
    = \exp(-0.5135)
    \approx 0.598
$$

$$
\text{BVS} = 6 \times 0.598 = 3.59
$$

This **overshoots** the expected $Z = 2$ by 1.59 — a documented
limitation of the Liebschner R₀ for porphyrin systems. Editing $R_0$
down to ~1.72 in the UI brings BVS to ~2.5; this teaching point is
explicitly built into the editable-R₀ panel.

---

## 6. Refinement-quality badge

A three-band classifier driven by the R-factor and the data /
parameter ratio extracted from the CIF's `_refine_ls_*` tags:

| R(gt) | data / parameter | Badge |
|---|---|---|
| $< 5\%$ AND $> 8$ | ✅ Good |
| anywhere else with R-factor available | ⚠️ Marginal |
| $> 10\%$ | 🚫 Check refinement |

The data / parameter ratio is

$$
\text{data/param} = \frac{\text{\_refine\_ls\_number\_reflns}}{\text{\_refine\_ls\_number\_parameters}}
$$

The "Marginal" caption explicitly notes that heavily-disordered
structures (lots of crystallisation solvent) often land here despite
the local coordination geometry being reliable. The full diagnostic
flow is in [`README.md`'s methodology section].

---

## 7. Standard uncertainties (esds)

Bond lengths are reported with their esd in the crystallographic
1-significant-figure convention:

| Value | esd | Display | Notes |
|---|---|---|---|
| 1.984 | 0.007 | `1.984(7)` | esd at $10^{-3}$ place |
| 1.984 | 0.012 | `1.98(1)` | esd rounds to 0.01, value rounded to 2 dp |
| 1.984 | 0.0007 | `1.9840(7)` | trailing zero needed to indicate precision |
| 1.984 | None | `1.984` | no esd available |

The mean uncertainty is propagated as the standard error of the mean:

$$
\sigma_{\bar d} = \frac{\sqrt{\sum_i \sigma_i^2}}{n}
$$

only when every bond carries an esd; otherwise the mean is shown
without its esd. No silent fabrication.

---

## 8. Disorder treatment

A user-controlled toggle: `min_occupancy` defaults to 0.5
(`"Use major component only"`) or 0.0 (`"Use all components"`). The
filter applies uniformly to:

1. The Fe-atom list
2. The N-atom list
3. The formal-bond detection loop (bonds where either atom fails the
   threshold are skipped)
4. The geometric-candidate detection loop
5. The geometry (ζ, Δ, Σ, Θ) and BVS computations downstream

When a CIF has no disordered atoms the toggle has no effect.

---

## Bibliography (machine-readable form)

See `CITATION.cff` for the BibTeX-style equivalent of the references
listed throughout this document.
