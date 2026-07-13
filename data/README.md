# Data directory

**No data files are committed to this repository.** Only paths live here — the
CSVs, parquet artifacts and pattern files are pulled at runtime and are excluded
by `.gitignore` (`data/raw/`, `data/processed/`, `*.csv`, `*.parquet`).

## Layout

```
data/
├── raw/                     # source files pulled from Kaggle (gitignored)
│   ├── HI-Small_Trans.csv       # IBM AMLworld transactions
│   └── HI-Small_Patterns.txt    # ground-truth laundering motifs
└── processed/               # pipeline artifacts (gitignored)
    └── edges_scored.parquet     # per-edge risk scores for the demo
```

## How to obtain the data

### Option A — Kaggle (source of truth)

The download is automated in `src/data_io.py`. It requires Kaggle credentials
provided **only via environment variables** (never committed):

```bash
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...            # or KAGGLE_API_TOKEN
python -c "from src import data_io; data_io.download_hi_small()"
```

Dataset: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`.

### Option B — DVC (reproducible)

Raw data and processed artifacts are DVC-tracked. Once a remote is configured
(see `.dvc/config`):

```bash
dvc pull        # fetch tracked data from the remote
dvc repro       # rebuild the pipeline (see dvc.yaml)
```
