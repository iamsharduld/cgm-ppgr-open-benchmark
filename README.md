# CGM PPGR Benchmark — code

Reproducible **code** for a pre-registered, multi-cohort benchmark that predicts the postprandial glucose
response (PPGR; 2-hour incremental AUC) to a meal in **non-diabetic adults** — with honest baselines,
disjoint-subject cross-validation, cross-cohort transfer, a feature-tier ablation, and an empirical
dual-CGM noise ceiling.

This repository contains the **pipeline only**. Given the public source datasets, the scripts regenerate
all metrics locally. (The manuscript and results are released separately.)

License: MIT (code) · Python 3.12 · Datasets: CGMacros + BIG IDEAs Lab (PhysioNet)

## Repository layout
```
src/ppgr/            # the pipeline (dataset-agnostic core + per-cohort adapters)
  iauc.py            # iAUC_0-120 outcome (Wolever/ISO 26642), baseline, resampling
  inclusion.py       # per-meal inclusion funnel
  features.py        # nested feature tiers (macros, context, anthro/labs, history, microbiome)
  evaluate.py        # LOSO + subject-grouped nested CV, baselines, models, bootstrap CIs
  loader.py          # shared Cohort interface
  adapters/          # cgmacros.py, bigideas.py
tests/test_iauc.py   # unit tests for the iAUC computation
run_*.py             # one script per experiment (see "Run")
docs/                # data dictionaries for the two source datasets
```

## Setup
```bash
python3 -m pip install -r requirements.txt      # Python 3.12; use a venv as needed
```

## Data (not redistributed — download from PhysioNet)
Raw data is **not** included (gitignored; governed by the source licenses). Download both open-access
datasets and place them under `data/`:

1. **CGMacros** (CC BY-NC-SA 4.0) — https://physionet.org/content/cgmacros/1.0.0/
   Extract the per-subject CSVs to:
   ```
   data/cgmacros/extracted/CGMacros/CGMacros-0XX/CGMacros-0XX.csv   (XX = 001..049)
   data/cgmacros/extracted/CGMacros/bio.csv  microbes.csv  gut_health_test.csv
   ```
2. **BIG IDEAs Lab Glycemic Variability** (ODC-By 1.0) —
   https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/
   ```bash
   wget -r -N -c -np -A 'Dexcom_*.csv,Food_Log_*.csv' \
     https://physionet.org/files/big-ideas-glycemic-wearable/1.1.2/   # -> data/physionet.org/files/...
   ```
The exact paths the scripts expect are at the top of each `run_*.py` / `src/ppgr/adapters/*.py`. See the
data dictionaries in [`docs/`](docs/).

## Run
```bash
PYTHONPATH=src python3 -m pytest tests/ -q     # unit tests

PYTHONPATH=src python3 run_cgmacros.py             # anchor benchmark (CGMacros, non-diabetic)
PYTHONPATH=src python3 run_bigideas.py             # second cohort
PYTHONPATH=src python3 run_transfer_matched.py     # cross-cohort transfer (matched)
PYTHONPATH=src python3 run_ablation.py             # feature-tier ablation
PYTHONPATH=src python3 run_noise_ceiling.py        # dual-CGM noise ceiling
PYTHONPATH=src python3 run_cgmacros_sensitivity.py # sensitivities
```
Runs are seeded and write their metrics tables to a local `results/` directory.

## License & citation
- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Data & artifacts you generate** (e.g. a local `results/`): respect the source-data licenses — CGMacros
  is CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution). Raw datasets are not redistributed here.
- Please cite via [`CITATION.cff`](CITATION.cff) and cite the underlying CGMacros and BIG IDEAs datasets.

## Transparency (AI use)
This code was developed with the assistance of an AI coding tool; the author reviewed and verified the
implementation and takes full responsibility for it.

> ⚠️ Research/education project — **not medical advice**.
