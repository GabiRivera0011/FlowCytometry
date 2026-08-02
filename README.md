# FlowCytometry
Dream project to create a automated flowcytometry analysis app

## Application Architecture

A Streamlit app (free/Community Cloud tier) built around an installable,
framework-agnostic core library (`src/flowcyto`) so the analysis logic stays
testable and independent of the UI. Since Community Cloud has no durable
storage, v1 is session-only: users upload `.fcs` files each visit and
everything lives in `st.session_state` for that session.

```
Streamlit UI            app/  →  upload · compensate · gate · transform · cluster · export
                          │
Workspace state           ▼
(st.session_state)      Workspace  →  loaded sample(s), gates, transforms, cluster result
                          │
Core library              ▼
                         src/flowcyto/
                         ├── io            FCS parsing (FlowKit)
                         ├── compensation  spillover matrix apply/edit
                         ├── gating        gate tree, polygon/rect/quad gates
                         ├── transforms    logicle / arcsinh / biexponential
                         ├── clustering    PCA/UMAP + KMeans
                         └── viz           Plotly figure builders
```

Build is staged in three phases so there's a working end-to-end app at
each step, rather than one big-bang release:

```
Phase 1              Phase 2                  Phase 3
Parse & view    ──▶  Compensate & gate   ──▶   Transform & cluster
(upload, view        (spillover matrix,        (logicle/arcsinh,
 metadata, raw        interactive gating,       PCA/UMAP + KMeans
 scatter/histogram)   gate tree)                on gated events)
```

Full details, diagrams, and module responsibilities: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Project Organization

```
FlowCytometry/
├── app/            # Streamlit UI (Home.py + pages/) — no analysis logic
├── src/flowcyto/   # core library: io, compensation, gating, transforms, clustering, viz
├── tests/          # pytest, with small sample .fcs files in tests/fixtures/
├── data/           # local scratch data (gitignored)
├── docs/           # architecture & project structure docs
└── .streamlit/     # theme + server config
```

`app/` only imports from `src/flowcyto`, never the reverse — the core
library has no Streamlit dependency, so it can be unit-tested and reused
outside the app.

Full layout and conventions: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## Phase 1 Checklist — Parse & View

- [ ] Add sample small `.fcs` file(s) to `tests/fixtures/` for local/offline testing
- [ ] `src/flowcyto/io`: load an `.fcs` file with FlowKit, return a `Sample` (metadata + event `DataFrame`)
- [ ] `tests/`: unit tests for `io` against the fixture files
- [ ] `src/flowcyto/state.py`: `Workspace` dataclass to hold the loaded `Sample` in `st.session_state`
- [ ] `src/flowcyto/viz`: basic Plotly scatter + histogram figure builders for raw channels
- [ ] `app/Home.py`: file upload widget → parse via `flowcyto.io` → store in `Workspace`
- [ ] `app/Home.py`: display metadata (channel list, event count, file info)
- [ ] `app/Home.py`: render raw scatter/histogram using `flowcyto.viz`
- [ ] Run locally (`streamlit run app/Home.py`) and verify with a real `.fcs` file
- [ ] Deploy to Streamlit Community Cloud and confirm `requirements.txt` is picked up
- [ ] Confirm session-only behavior (data clears on refresh / new session, as expected for v1)

## References

- [FlowRepository](http://flowrepository.org/) — public repository of flow cytometry datasets, used as the source of sample `.fcs` files for development and testing
