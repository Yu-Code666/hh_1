# Repository Guidelines

## Project Structure & Module Organization
- Root notebooks: `xgboost-竞争-2023-2025-Copy1.ipynb`, `pre2023-2025.ipynb`, `express2023-2025.ipynb`, `补充竞品航线价格.ipynb` (data prep, training, analysis).
- Python modules: `xgb_bayes_opt.py` (XGBoost Bayes tuning). New LightGBM/CatBoost utilities should live beside it.
- Data: `data_hh/` (raw CSVs); results in `data_hh/result/` (e.g., `pre_2023-2025_with_comp_train.csv`, `..._test.csv`, `pred_*.csv`, `model/`).
- Docs: `PLAN.md`, `AGENTS.md`.

## Build, Test, and Development Commands
- Environment (Python 3.9+):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -U pip`
  - `pip install pandas numpy scikit-learn xgboost lightgbm catboost bayesian-optimization jupyter`
- Run notebooks: `jupyter lab` (open `xgboost-竞争-2023-2025-Copy1.ipynb`).
- Optional lint/format (recommended): `pip install ruff black` then `ruff check .` and `black .`.

## Coding Style & Naming Conventions
- Python: 4‑space indentation, type hints where practical, docstrings for public functions, `snake_case` for functions/variables, `CapWords` for classes, UPPER_CASE for constants.
- Notebooks: keep cells deterministic (`random_state=42`), avoid hard‑coding absolute paths; write outputs under `data_hh/result/`.
- Filenames: English where possible; keep meaningful, short, and consistent. Do not rename large data files without discussion.

## Testing Guidelines
- Framework: `pytest` (if adding Python modules). Place tests in `tests/` mirroring module paths.
- Run: `pytest -q`.
- Cover feature engineering utilities and metric calculations (e.g., SMAPE) with small unit tests.

## Commit & Pull Request Guidelines
- Commits: small, focused, imperative subject line (<=72 chars). Prefer Conventional Commit style when helpful.
  - Examples: `feat(train): add LightGBM bayes tuning`, `fix(feat): guard comp_price_ratio div-by-zero`.
- PRs: include summary, motivation, key files/paths, how to reproduce (commands/notebook cells), before/after metrics (SMAPE), and screenshots/tables if relevant.

## Security & Configuration Tips
- Large/sensitive data: keep under `data_hh/`. Avoid committing new large binaries; store derived artifacts in `data_hh/result/` and commit only small CSV/JSON as needed.
- Reproducibility: keep `random_state=42`, persist split indices when introducing new models, and document any external assumptions.

## Overview of Usage Rules

Responses must be written in Simplified Chinese.

After receiving a request, first organize a detailed to-do list and send it to the user for confirmation; if the user proposes any revisions, the list must be reorganized and reconfirmed.