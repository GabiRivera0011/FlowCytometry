# FlowCytometry
Dream project to create a automated flowcytometry analysis app

## Application Architecture

A Streamlit app (free/Community Cloud tier) built around an installable,
framework-agnostic core library (`src/flowcyto`) so the analysis logic stays
testable and independent of the UI. Since Community Cloud has no durable
storage, the app is session-only: users upload `.fcs` files each visit and
everything lives in `st.session_state` for that session.

```
Streamlit UI            app/  →  upload · compensate · gate · tabulate
                          │
Workspace state           ▼
(st.session_state)      Workspace  →  loaded sample(s), spillover matrix, gate refs
                          │
Core library              ▼
                         src/flowcyto/
                         ├── io            FCS/CSV parsing (FlowKit), metadata
                         ├── compensation  spillover matrix: controls, diagnostics, search
                         ├── transforms    Logicle display transform (fluorescence channels)
                         └── viz           Plotly figure builders (scatter, bars, heatmap, pairs)
```

Interactive gating itself is Streamlit-coupled (session_state, Plotly select
events) and lives in `app/_gating_widgets.py` rather than `src/flowcyto` —
see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the reasoning. Of the
two modules planned early on, `transforms` now holds the Logicle display
transform (see *Statistical & math approach* below); `clustering` (PCA/UMAP
+ KMeans) is still an empty stub.

Full details, diagrams, and module responsibilities: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
(note: that doc still describes the original phased plan and is not fully
current with what's below).

## Project Organization

```
FlowCytometry/
├── app/            # Streamlit UI (Home.py + pages/) — no analysis logic
├── src/flowcyto/   # core library: io, compensation, transforms, viz
├── tests/          # pytest, with small sample .fcs files in tests/fixtures/
├── data/           # local scratch data (gitignored)
├── docs/           # architecture & project structure docs
└── .streamlit/     # theme + server config
```

`app/` only imports from `src/flowcyto`, never the reverse — the core
library has no Streamlit dependency, so it can be unit-tested and reused
outside the app.

Full layout and conventions: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## Current pages

1. **Parse and View** — load an `.fcs`/`.csv` sample (plus optional
   single-stain controls), view channel metadata, preview raw scatter/histogram.
2. **Compensation** — compare compensation scenarios, edit the spillover
   matrix directly, inspect the most notable channel pairs.
3. **Gating — Live Cells** — acquisition QC, debris/doublet exclusion,
   singlet discrimination, viability gating.
4. **Gating — Broad Cellular Lineage** — lymphocytes vs. myeloid cells.
5. **Gating — Subset Cell Population** — free-form, user-defined gating
   workspaces for further rounds of subsetting, branching from any earlier gate.
6. **Gating Summary** — every gate drawn across the gating pages, tabulated
   with exact event counts resolved against the full (un-subsampled) file.

## Statistical & math approach

### Compensation

- **Control-based spillover row** (`compensation/matrix.py:compute_spillover_row`):
  for each single-stain control, events are split into a "positive"
  (>90th percentile of the primary channel) and "negative" (<10th
  percentile) population. The spillover coefficient into each other
  detector is `(median(positive) − median(negative))` in that detector,
  normalized by the same quantity in the primary channel. This is the
  standard median-based approach used by mainstream flow software. It
  assumes a control with a reasonably separable positive population — a
  control with a small positive fraction could skew the 90th-percentile cut.
- **Theil-Sen over-compensation diagnostic** (`compensation/diagnostics.py:slope_matrix`):
  for each ordered channel pair (i, j), a robust (Theil-Sen, i.e.
  median-of-pairwise-slopes) regression of channel j on channel i is fit,
  restricted to events where i is in its own dimmest 20th percentile. At
  that low intensity, real co-expression biology shouldn't produce a
  relationship, so any slope that remains is attributed to spillover or
  autofluorescence. A **negative** slope is the tell for *over*-compensation:
  genuine spillover only adds signal, it never subtracts it. The score
  shown in the app ("% flagged pairs") is the average, across channels, of
  how many incoming pairs come out negative — lower is better.
  Theil-Sen (not OLS) is used specifically because it's robust to the
  heavy-tailed, outlier-prone distributions typical of flow data.
  `theilslopes` also returns a confidence interval on the slope
  (`slope_matrix_with_ci`); a pair is only flagged as "negative" when the
  CI's **upper bound** is also below zero — i.e. significantly negative, not
  just negative at the point estimate — so the score isn't sensitive to
  noise on the (relatively small) 1,200-event fit subsample. (The diagonal
  exclusion in `pct_negative_slope` uses `np.where(..., np.nan)` rather than
  `np.fill_diagonal` + a `< 0` comparison: `NaN < 0` evaluates to `False`,
  not `NaN`, so the comparison-based version silently counted the diagonal
  as "not negative" instead of excluding it — a real bug, fixed.)
- **Scenario comparison & default selection**: each candidate matrix
  (already-compensated/identity, control-derived, automatic search) is
  scored with the same diagnostic (smaller `fit_size=400` for responsiveness
  on page load) and the minimum-score scenario is selected as the default.
  `ScenarioScore.score_se` is an **empirical (bootstrap) standard error**:
  the diagnostic is re-run a few times (`n_bootstrap=3`, at a smaller
  `bootstrap_fit_size`) with different random seeds, and the spread of the
  resulting scores is used directly — not a formula that assumes each
  channel's flagged-pair fraction is independent. They aren't: every target
  channel in one source channel's row is fit from the *same* sampled/
  dim-gated events (`_fit_slope_matrix`'s `pick`), so an independence-based
  standard error would understate the real variability. Resampling the
  whole procedure captures that correlation automatically, without needing
  to model it analytically. `scenarios_are_close` compares two scenarios'
  scores against their combined SE, and the Compensation page shows a note
  when the picked default isn't meaningfully better than the runner-up.
- **Automatic (no-controls) search** (`diagnostics.py:run_validated_search`):
  greedily proposes a per-pair coefficient (bounded scalar minimization of
  % negative events, bounds `[0, 2]`) on a training sample, in random pair
  order, but only **keeps** it if it also improves (or doesn't worsen) a
  held-out validation sample. This train/validation split is deliberate and
  load-bearing — an earlier, unguarded version overfit to sampling noise
  and made the full dataset worse.

### Gating

- All interactive gating plots render from a **fixed 20,000-event random
  subsample** (`np.random.default_rng(0)`, seeded for reproducibility), for
  responsiveness — not the full file.
- **Fluorescence channels are displayed on a Logicle transform**
  (Parks, Moore & Herzenberg 2006; GatingML 2.0 spec; `flowcyto.transforms`,
  via FlowKit's `LogicleTransform`), applied in every gating page's
  `channel_values` and in the Compensation page's pair-detail/notable-pairs
  plots. A linear axis compresses the dim/negative population into a sliver
  near zero and makes population separation genuinely harder to judge;
  Logicle (unlike plain log) handles zero and negative values natively,
  which matters because compensated fluorescence routinely has both.
  Forward/side scatter and Time stay linear, which is standard for scatter
  channels. The compensation **matrix fit/diagnostics themselves
  intentionally stay on raw linear values** — only display and gating use
  the transform, matching standard practice (FlowJo/Cytobank compute
  spillover on linear data but display and gate on a transformed scale).
  - **Parameters are data-driven per channel**, not one fixed cofactor
    shared by every channel (`estimate_logicle_params`): `T` (top of scale)
    is the channel's own 99.9th percentile; `W` (the linear region's width,
    in decades) follows the Logicle paper's relationship to the negative
    population's spread (`W = (M - log10(T/|r|)) / 2`, `r` = 5th percentile
    of the channel's negative values), so a channel with little negative
    signal gets a narrower linear region than one with wide
    compensation-driven negative spread. `M=4.5`, `A=0` are fixed, standard
    defaults.
  - **Parameters are estimated once per channel and cached**
    (`workspace.transforms`, read/written by `app/_gating_widgets.py:
    transform_channel`), not re-estimated on every plot — a gate is drawn
    against one specific transformed scale, and resolving it against the
    full dataset later has to replay that *exact* same scale, not a freshly
    re-estimated one that could differ slightly. The cache is invalidated
    (`workspace.transforms = None`) whenever what "compensated" means for a
    channel changes: a new sample, new/changed controls, or an applied/reset
    spillover matrix edit.
- Each axis **autoscales to a central percentile band** of its own channel
  (default central 90%, adjustable via "Axis zoom" up to "All events"),
  rather than full min–max, so a few extreme outliers don't stretch the
  visible range past where the bulk of the data sits. This operates on
  whichever scale (linear or Logicle) the channel is already displayed on.
- Four gate shapes — box, lasso (point-in-polygon via vectorized even-odd
  ray casting), point-click, and quadrant — are stored as their literal
  drawn geometry (`app/_gating_widgets.py:_mask_from_shape`), not as a
  ratio or extrapolation.
- **Exact full-dataset resolution**: because gates are drawn against the
  20K subsample, "how many cells are actually in this gate" is answered by
  replaying each gate's *exact* geometric boundary against the full,
  un-subsampled file (`resolve_full_mask`), recursively walking back
  through whatever chain of upstream gates/picker combinations fed it. This
  is not a scaled estimate — every one of the full file's events is tested
  against the same literal boundary a user drew. The full file's channel
  values go through the identical cached Logicle transform before that
  test, so box/lasso/quadrant boundaries compare like-for-like against what
  was actually drawn on screen.
  - Point-click gates are the one shape that **can't** be replayed by
    coordinate comparison against the full file: raw channel values are
    often quantized, so an exact-equality match against ~2M events could
    hit far more than the one event actually clicked in the 20k-event
    subsample. Instead, each clicked point's **original full-file event
    index** is captured at draw time (via the page's own subsample→full-file
    index mapping) and stored on the gate; `resolve_full_mask` looks those
    indices up directly, with zero ambiguity, rather than re-matching
    coordinates.
- **Subsample-vs-full-file divergence check** (Gating Summary page): each
  gate's `% of subsample` and `% of full file` are compared, and an
  adjustable-threshold warning flags gates where the two diverge — a sign
  the boundary was drawn against a sparse or misleading view of the
  population and may need redrawing, e.g. in Pseudocolor/Contour view.

### Remaining known gaps

- **Logicle's own parameter-estimation heuristic is one reasonable choice
  among several in the literature.** `estimate_logicle_params`'s 99.9th
  percentile for `T` and 5th-percentile-of-negatives for `r` are defensible,
  documented defaults, but flow software varies in exactly which percentiles
  it uses for auto-estimation (e.g. FlowJo/flowCore's choices aren't
  identical to these). Not wrong, just not the one single universal
  standard — a channel that looks slightly over- or under-spread on-screen
  can still be adjusted by redrawing after a different compensation, since
  the cache recomputes from whatever the channel's data looks like then.
- **`score_se`'s bootstrap is small (`n_bootstrap=3`) and uses a reduced
  `fit_size` for speed** — enough to catch "these scores aren't really
  different" (see the Compensation section above and
  `test_score_se_is_a_nonzero_empirical_bootstrap_not_a_fixed_formula`),
  but a small bootstrap is itself a noisy estimate of variability; it isn't
  a rigorous confidence interval, and could disagree with a larger/slower
  bootstrap run at full `fit_size`.
- **Fixed 90th/10th percentile split for control positive/negative
  populations** (`compensation/matrix.py:compute_spillover_row`) —
  reasonable default, but can misfire on a control with a small positive
  fraction. Lower priority: the resulting matrix is directly, cell-by-cell
  editable in the app as a manual override.

## References

- [FlowRepository](http://flowrepository.org/) — public repository of flow cytometry datasets, used as the source of sample `.fcs` files for development and testing
- Mair, F., & Liechti, T. (2021). Comprehensive Phenotyping of Human Dendritic Cells and Monocytes. *Cytometry Part A*, 99(3), 231–242. https://doi.org/10.1002/cyto.a.24269 — source of the dendritic cell/monocyte panel data (`data/pbmc_s1.csv`, `data/*.fcs`) used in this project
