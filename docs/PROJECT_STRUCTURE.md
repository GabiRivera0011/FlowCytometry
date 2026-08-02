# Project structure

```
FlowCytometry/
├── app/                        # Streamlit UI — no analysis logic lives here
│   ├── Home.py                 # entrypoint: `streamlit run app/Home.py`
│   └── pages/                  # multipage app, numbered for sidebar order
│       ├── 1_Upload.py
│       ├── 2_Compensation.py
│       ├── 3_Gating.py
│       ├── 4_Transform_Cluster.py
│       └── 5_Export.py
│
├── src/flowcyto/               # installable core library (framework-agnostic)
│   ├── io/                     # FCS parsing → Sample objects
│   ├── compensation/           # spillover matrix apply/edit
│   ├── gating/                 # gate tree, polygon/rect/quad gates
│   ├── transforms/             # logicle / arcsinh / biexponential
│   ├── clustering/             # PCA/UMAP + KMeans/FlowSOM-lite
│   ├── viz/                    # Plotly figure builders
│   └── state.py                # Workspace dataclass (session state model)
│
├── tests/
│   ├── fixtures/                # small sample .fcs files for offline tests
│   └── ...                      # test_*.py mirroring src/flowcyto layout
│
├── data/                        # local scratch data, gitignored
├── docs/
│   ├── ARCHITECTURE.md
│   └── PROJECT_STRUCTURE.md
├── .streamlit/
│   └── config.toml              # theme + server config
├── requirements.txt
├── .gitignore
└── README.md
```

## Conventions

- `app/` imports from `src/flowcyto`, never the reverse.
- Every `src/flowcyto/<module>` is a package (`__init__.py`) so it can be
  unit-tested and imported independently of Streamlit.
- Page filenames are prefixed with numbers to control Streamlit's sidebar
  ordering; the prefix is cosmetic and not part of the module path.
