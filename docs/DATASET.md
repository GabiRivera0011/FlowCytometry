# Dataset Provenance and Panel Design

This document records the origin and structure of the flow cytometry data in
`data/`, and its relationship to the panel described in the cited source
publication.

## Source publication

Mair, F., & Liechti, T. (2021). Comprehensive Phenotyping of Human Dendritic
Cells and Monocytes. *Cytometry Part A*, 99(3), 231–242.
https://doi.org/10.1002/cyto.a.24269

The publication addresses the increasingly blurred distinction between human
dendritic cells (DCs) and monocytes under inflammatory conditions, and
proposes an updated high-dimensional flow cytometry gating strategy to
resolve it. Its central contribution is the addition of two marker pairs to
existing DC/monocyte panels:

- **CD88 and CD89** — used to positively identify and exclude monocytes
  (including CD16⁺ nonclassical/intermediate subsets, which show more
  heterogeneous CD89 expression).
- **CD5 and CD163** — used to subset **cDC2** from the newer **DC3**
  population, a distinction earlier panels could not make.

Note: the full text is behind a publisher paywall and was not directly
accessible. The summary above is drawn from the publicly available abstract
(via Europe PMC) and indexed excerpts of the paper's stated conclusions, not
from the complete methods section. The correspondence to `data/`'s file and
column structure below is nonetheless specific enough to identify the panel
with confidence, since CD88, CD89, CD5, and CD163 — the paper's namesake
additions — are all present as columns in `data/pbmc_s1.csv`.

## Panel-to-file correspondence

`data/` contains three categories of `.fcs` file plus one derived `.csv`:

| File pattern | Role |
|---|---|
| `Comp_<Target> <Marker>_<well>.fcs` | Single-stain compensation control — cells stained with only that one antibody, used to measure that fluorophore's spillover into every other detector |
| `Comp_Unstained mouse_*.fcs`, `Comp_Unstained rat_*.fcs` | Unstained negative controls, split by antibody host species (the panel mixes mouse- and rat-clone antibodies, which require species-matched autofluorescence baselines) |
| `PBMC_<date>_<replicate>_DCs_<well>.fcs` | Full-panel acquisition — PBMCs stained with every antibody simultaneously, across two collection dates (`090120`, `091520`) with two replicate tubes each |
| `pbmc_s1.csv` | Exported event table |

`pbmc_s1.csv` was confirmed to be an export of `PBMC_090120_1_DCs_A5_A05_031.fcs`
specifically: both contain exactly 2,088,689 events, and per-event fluorescence
values match to floating-point precision (verified via FlowKit).

## Marker-to-panel role mapping

The 32 columns in `pbmc_s1.csv` correspond to the panel as follows:

| Columns | Role |
|---|---|
| `FSC-A`, `FSC-H`, `SSC-A`, `Time` | Acquisition channels — cell size/granularity and time, not part of the antibody panel |
| `Live Dead UV Blue` | Viability stain — excludes dead cells prior to phenotyping |
| `CD3`, `CD19`, `CD20`, `CD56` | Lineage-exclusion channel — removes T cells, B cells, and NK cells |
| `HLA-DR` | Pan-antigen-presenting-cell marker — gates in DCs/monocytes broadly |
| `CD14`, `CD16`, `CD88`, `CD89` | Monocyte identification (`CD88`/`CD89` are the paper's proposed addition) |
| `CD1c`, `CD172a` | cDC2 markers |
| `CD5`, `CD163` | cDC2-vs-DC3 subsetting (the paper's second proposed addition) |
| `CD141`, `XCR1`, `CADM1` | cDC1 markers |
| `CD303`, `FcER1a` | Plasmacytoid DC (pDC) markers |
| `CD11c`, `CD45RA`, `CD301` | Additional DC/monocyte phenotyping markers |

Column headers follow a `"<target> <marker>"` convention (e.g. `CADM1 FITC`
→ target `CADM1`, fluorophore marker `FITC`), with the exceptions handled in
`notebooks/dev_playground.ipynb`'s EDA section: the acquisition channels
above, blank headers (`Unnamed: N`, columns with no associated
target/marker), and `Live Dead UV Blue`, which names a viability dye rather
than a target/marker pair.

## Compensation status

**The data in `pbmc_s1.csv` (and its source `.fcs` files) has already been
compensated.**

This was verified directly from the FCS metadata rather than assumed:

- Every `.fcs` file checked (`Comp_*` controls and `PBMC_*_DCs` acquisitions
  alike) embeds a `$SPILL` spillover matrix that is a pure **28×28 identity
  matrix** — no off-diagonal spillover coefficients.
- Every file also carries a custom keyword, `apply compensation = TRUE`.
- Loading a sample with FlowKit and applying the embedded matrix explicitly
  (`Sample.apply_compensation`) produces event values identical to the
  uncompensated (`source="raw"`) values, as expected when the matrix being
  applied is the identity — a no-op.

This is the standard signature of an FCS export where compensation was
already applied to the numeric data (e.g., during acquisition or at export
time), with the identity matrix and flag left in place purely as file-format
metadata rather than indicating outstanding correction work.

**Implication for the analysis pipeline:** no additional compensation step
is required before gating, transforming, or clustering `pbmc_s1.csv` or the
`PBMC_*_DCs_*.fcs` files. If `src/flowcyto/compensation` needs to be
exercised meaningfully (e.g., for unit tests), it should either use the
`Comp_*` single-stain files to construct a matrix from scratch, or a
separate, deliberately uncompensated fixture, rather than treating the
existing PBMC files as needing correction.
