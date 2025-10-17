# Repository Guidelines

## Project Structure & Module Organization
Exploratory modeling lives in Jupyter notebooks such as `express2023-2025.ipynb`, `lightbgm-竞争-2023-2025.ipynb`, and `补充竞品航线价格.ipynb` in the repository root; give new analysis notebooks a descriptive, date-stamped name so they sort naturally. Source-like scripts belong beside the notebooks; the current `补充cap.py` script shows the expected structure: top-level pandas workflow with relative paths into the shared data vault. Raw and derived datasets reside in `data_hh/`; keep raw encrypted inputs in the directory root and place any generated artifacts in `data_hh/result/`, mirroring the existing naming pattern (`pre_*.csv`, `cap_补全后.csv`, etc.).

## Build, Test, and Development Commands
- `python 补充cap.py`: recomputes the seat-capacity backfill and writes results to `data_hh/result/`.
- `jupyter lab` (or `jupyter notebook`): launches the notebook workspace for iterative modeling; run from this directory to preserve relative paths.
- `python -m venv .venv && source .venv/bin/activate`: create and activate an isolated environment before installing pandas, xgboost, and other experiment libraries.

## Coding Style & Naming Conventions
Stick with PEP 8 defaults: 4-space indentation, snake_case variables, and descriptive notebook filenames combining topic and date (`xgboost-竞争-测试2024-7-10数据-origin.ipynb` is a good template). Data columns should remain in lowercase with underscores, matching the CSV schema. When adding scripts, prefer pure functions for transformations and keep I/O paths centralized at the top of the file for easier reconfiguration.

## Testing Guidelines
There is no automated test harness yet; validate outputs by re-running the relevant notebook cells and diffing generated CSVs against baselines in `data_hh/result/`. For scripts, add quick assertions after the main dataframe transformations (e.g., `assert df.cap.ge(0).all()`). Before committing, open the downstream notebook that consumes the refreshed files to ensure charts and metrics still load.

## Commit & Pull Request Guidelines
The history favors short, Chinese-language summaries (see `git log` entries like “将数据放在同一路径”); keep that style and lead with the functional change. Bundle related notebook, data, and script updates in one commit with a concise body explaining the workflow and any data dependencies. Pull requests should outline the scenario, attach sample output screenshots when charts changed, and link to any issue or runbook entry.

## Security & Data Handling
Encrypted CSVs under `data_hh/` must stay encrypted at rest; do not commit decrypted exports. Scrub notebooks of credentials and strip large intermediate outputs (`jupyter nbconvert --ClearOutputPreprocessor.enabled=True`) before sharing. Document any new data sources in `执行顺序.txt` so reruns remain reproducible.
