# LightGBM + CatBoost + Ensemble Plan

This plan extends the existing XGBoost pax workflow by adding LightGBM and CatBoost, supporting both manual and Bayesian tuning for all three, enforcing identical splits, and producing an ensemble prediction with final SMAPE computed on the original (de‑standardized) scale.

## Steps
1. Create unified preprocessing pipeline and lock split indices (random_state=42)
2. Retrain XGBoost using saved best params from `xgb_bayes_best_params.json` (no re-tuning needed)
3. Implement and train LightGBM (manual + bayes modes, native categorical)
4. Implement and train CatBoost (manual + bayes modes, native categorical)
5. Generate predictions from all three models on validation and test sets
6. Ensemble averaging on original scale + SMAPE evaluation (metrics saved as JSON)
7. CLI wrappers and docs updates

## Agreed Constraints
- Final evaluation uses SMAPE on the original scale. SMAPE on standardized scale is only for training/monitoring.
- **All three models use identical train/val split indices** with `random_state=42`, persisted to disk for reproducibility.
- **XGBoost retraining strategy**: Load existing best parameters from `xgb_bayes_best_params.json` (if available) and retrain on the new locked splits. Handle key mapping (e.g., `lambda` vs `lambda_`); if `num_boost_round` is missing, default to a safe value (e.g., 500). This avoids costly re-tuning (~hours) while ensuring fair comparison (~5–10 min retrain).
- LightGBM and CatBoost leverage native categorical handling:
  - LightGBM: set `categorical_feature` for columns converted to pandas `category` dtype.
  - CatBoost: pass `cat_features` indices to `Pool`, using raw string or category columns.
- All three models support two modes:
  - **bayes**: Bayesian Optimization via `bayesian-optimization` with early stopping.
  - **manual**: user-specified params with early stopping.

## Data, Features, and Preprocessing
- Inputs:
  - Train: `data_hh/result/pre_2023-2025_with_comp_train.csv`.
  - Final test (holdout): `data_hh/result/pre_2023-2025_with_comp_test.csv` (not used in splitting; only for final evaluation).
- Feature engineering mirrors the XGBoost notebook (time features, competitor deltas/ratios). For leakage safety, compute `from_avg_pax_month`, `to_avg_pax_month`, and `seg_avg_pax` on the TRAIN split only, then merge into VAL/TEST/HOLDOUT by keys.
- Feature types and handling:
  - Numeric (standardize):
    - Base: `cap, legs, leg_no, duration, year, month, day, weekday, hour, minute, quarter, is_weekend, is_holiday_season`.
    - Price: `unit_price, competitor_price, comp_price_diff_abs, comp_price_ratio` (and `comp_price_ratio_abs` if present).
    - Stats: `from_avg_pax_month, to_avg_pax_month, seg_avg_pax`.
    - Cyclic: `hour_sin, hour_cos, month_sin, month_cos`.
    - Embeddings (if present): `a_embedding_1, a_embedding_2, b_embedding_1, b_embedding_2, c_embedding_1, c_embedding_2, from_embedding_1, from_embedding_2, to_embedding_1, to_embedding_2`.
    - Graph embedding labels (numeric codes, existing features): `a_label, b_label, c_label, from_label, to_label` derived from city graph clustering (values 0/1/2). Standardize with other numerics for all models.
  - Categorical (strings; native for LGBM/CatBoost): `flt_no, aircraft, a, b, c, from, to`.
  - XGBoost categorical (label-encoded, create new columns): `flt_no_encoded, aircraft_encoded` (use LabelEncoder on `flt_no, aircraft`). For `a, b, c, from, to`, XGBoost reuses the existing `*_label` columns instead of creating duplicates.
- Splits:
  - Create a deterministic train/val split on the TRAIN file only with `random_state=42`. Persist indices to `data_hh/result/splits/train_val_indices.json` and reuse across all three models.
  - The TEST file is a separate holdout used solely for final evaluation.
- Scaling:
  - y: one shared `StandardScaler` for all models at `data_hh/result/encoder/standard_scaler_y.pkl`.
  - X (numeric only): `StandardScaler` saved to `data_hh/result/encoder/standard_scaler_X_numeric.pkl`. LightGBM/CatBoost use unscaled categorical columns (ensure `category` dtype), while XGBoost uses label-encoded categorical columns alongside the standardized numeric block.

## Tuning
- LightGBM (bayes): search `num_leaves`, `max_depth`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, `learning_rate`, `n_estimators`; objective `regression` with custom SMAPE eval for monitoring.
- CatBoost (bayes): search `depth`, `l2_leaf_reg`, `bagging_temperature`, `learning_rate`, `iterations` with `loss_function='RMSE'` and SMAPE monitoring; use `early_stopping_rounds` on validation.
- Manual modes: accept explicit parameter dicts for each model.

## Outputs
- Predictions (per split):
  - Individual models:
    * Validation: `data_hh/result/pred_xgb_retrain_val.csv`, `data_hh/result/pred_lgbm_val.csv`, `data_hh/result/pred_cat_val.csv`
    * Test (holdout): `data_hh/result/pred_xgb_retrain_test.csv`, `data_hh/result/pred_lgbm_test.csv`, `data_hh/result/pred_cat_test.csv`
    * Columns: `y_true_orig, y_pred_orig, y_pred_standardized`
  - Ensemble:
    * Validation: `data_hh/result/pred_ensemble_val.csv`
    * Test (holdout): `data_hh/result/pred_ensemble_test.csv`
    * Columns: `y_true_orig, y_pred_xgb_orig, y_pred_lgbm_orig, y_pred_cat_orig, y_pred_ensemble_orig`
- Models (saved under `data_hh/result/model/`):
  - XGBoost: `data_hh/result/model/xgb_model_retrain.json`.
  - LightGBM: `data_hh/result/model/lgbm_bayes.txt` or `data_hh/result/model/lgbm_manual.txt`.
  - CatBoost: `data_hh/result/model/cat_bayes.cbm` or `data_hh/result/model/cat_manual.cbm`.
- Metrics (scalar SMAPE, not per-row):
  - Validation: `data_hh/result/metrics_val.json` with `smape_xgb`, `smape_lgbm`, `smape_cat`, `smape_ensemble`.
  - Test: `data_hh/result/metrics_test.json` with the same keys.
- Tuning logs:
  - XGBoost: reuse existing `xgb_bayes_best_params.json` (no new tuning).
  - LightGBM: `data_hh/result/lgbm_bayes_trials.csv`, `data_hh/result/lgbm_bayes_best_params.json`.
  - CatBoost: `data_hh/result/cat_bayes_trials.csv`, `data_hh/result/cat_bayes_best_params.json`.
- Splits: `data_hh/result/splits/train_val_indices.json` (format: `{ "train": [...], "val": [...], "uids": [...] }` — `uids` optional, for alignment validation).
- Scalers: 
  - `data_hh/result/encoder/standard_scaler_y.pkl` (shared by all models).
  - `data_hh/result/encoder/standard_scaler_X_numeric.pkl` (for numeric features only).

## Notes
- **Retraining rationale**: All three models are trained from scratch using identical locked splits to ensure fair comparison and reliable ensemble. XGBoost uses its previously optimized hyperparameters (no re-tuning), saving significant time while maintaining consistency.
- **SMAPE calculation**: Unified across models; final score is always reported on original scale. The ensemble uses the arithmetic mean of the three original-scale predictions.
- **Feature pipeline**: Categorical columns (`flt_no`, `aircraft`, `a`, `b`, `c`, `from`, `to`) are preserved as strings/categories for LightGBM/CatBoost native handling, while numeric features are standardized. For XGBoost, use `flt_no_encoded, aircraft_encoded` (newly created) and reuse existing `a_label, b_label, c_label, from_label, to_label`; do not feed raw string categories to XGBoost.
- **High-cardinality features**: If `flt_no` exceeds 10k unique values and causes CatBoost training slowdown (>1 hour), consider excluding it or using a label-encoded version consistently across models.
- **Library-specific reproducibility**: For CatBoost, set `allow_writing_files=False` and `random_state=42`. For LightGBM, ensure categorical columns are `category` dtype and not altered by scaling.
- **Reproducibility**: Same indices, same scalers, same random_state=42 across all models to guarantee comparability and stable ensembling.

## Usage
- Environment
  - python -m venv .venv && source .venv/bin/activate
  - pip install -U pip
  - pip install pandas numpy scikit-learn xgboost lightgbm catboost bayesian-optimization
- Train + Ensemble
  - python train_ensemble.py \
    --train-csv data_hh/result/pre_2023-2025_with_comp_train.csv \
    --test-csv  data_hh/result/pre_2023-2025_with_comp_test.csv \
    --lgbm-mode bayes --cat-mode bayes
  - Manual modes 示例：
    - python train_ensemble.py --lgbm-mode manual --lgbm-params my_lgbm.json
    - python train_ensemble.py --cat-mode manual --cat-params my_cat.json
- Outputs
  - Predictions: data_hh/result/pred_*_{val,test}.csv
  - Models: data_hh/result/model/
  - Metrics: data_hh/result/metrics_{val,test}.json
