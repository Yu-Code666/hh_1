from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from metrics import save_json, smape_np
from preprocess_pipeline import (
    RESULT_DIR,
    TARGET_COL,
    prepare_datasets,
)
from xgb_bayes_opt import smape_eval as xgb_smape_eval
from lgb_bayes_opt import bayesian_tune_lgbm, train_lgbm_manual
from cat_bayes_opt import bayesian_tune_catboost, train_cat_manual


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrain_xgb(
    X_train: pd.DataFrame,
    y_train_std: np.ndarray,
    X_val: pd.DataFrame,
    y_val_std: np.ndarray,
    best_params_path: str,
    results_dir: str,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
):
    # Load best params
    best_cfg = _load_json_if_exists(best_params_path)
    if best_cfg and "best_params" in best_cfg:
        best_params = dict(best_cfg["best_params"])  # type: ignore[index]
    else:
        best_params = {}

    # Map keys and defaults
    final_params = {
        "objective": "reg:squarederror",
        "seed": 42,
        "tree_method": "hist",
    }
    for k in [
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "alpha",
    ]:
        if k in best_params:
            final_params[k] = best_params[k]
    # lambda mapping
    if "lambda" in best_params:
        final_params["lambda"] = best_params["lambda"]
    elif "lambda_" in best_params:
        final_params["lambda"] = best_params["lambda_"]
    num_boost_round = int(best_params.get("num_boost_round", 500))

    dtrain = xgb.DMatrix(X_train, label=y_train_std)
    dval = xgb.DMatrix(X_val, label=y_val_std)
    evals = [(dtrain, "train"), (dval, "validation")]

    booster = xgb.train(
        final_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        feval=xgb_smape_eval,
        maximize=False,
        verbose_eval=verbose_eval,
    )

    _ensure_dir(os.path.join(results_dir, "model"))
    model_path = os.path.join(results_dir, "model", "xgb_model_retrain.json")
    try:
        booster.save_model(model_path)
    except Exception:
        pass
    return booster


def _save_preds_csv(path: str, y_true_orig: Optional[np.ndarray], y_pred_orig: np.ndarray, y_pred_std: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    df = pd.DataFrame({
        "y_pred_orig": y_pred_orig,
        "y_pred_standardized": y_pred_std,
    })
    if y_true_orig is not None:
        df.insert(0, "y_true_orig", y_true_orig)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Train XGB+LGBM+CatBoost and ensemble predictions.")
    parser.add_argument("--train-csv", default=os.path.join(RESULT_DIR, "pre_2023-2025_with_comp_train.csv"))
    parser.add_argument("--test-csv", default=os.path.join(RESULT_DIR, "pre_2023-2025_with_comp_test.csv"))
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--exclude-fltno", action="store_true", help="Exclude flt_no as categorical for LGBM/CatBoost")
    parser.add_argument("--lgbm-mode", choices=["bayes", "manual"], default="bayes")
    parser.add_argument("--cat-mode", choices=["bayes", "manual"], default="bayes")
    parser.add_argument("--lgbm-params", default=None, help="Path to JSON with LightGBM manual params")
    parser.add_argument("--cat-params", default=None, help="Path to JSON with CatBoost manual params")
    parser.add_argument("--results-dir", default=RESULT_DIR)
    parser.add_argument("--xgb-best-params", default=os.path.join(RESULT_DIR, "xgb_bayes_best_params.json"))
    args = parser.parse_args()

    # Prepare data and features (80:10:10 split)
    pdata = prepare_datasets(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        val_size=args.val_size,
        test_size=args.test_size,
        exclude_flt_no=args.exclude_fltno,
    )

    # 1) XGBoost retrain
    xgb_model = retrain_xgb(
        pdata.X_train_xgb,
        pdata.y_train_std,
        pdata.X_val_xgb,
        pdata.y_val_std,
        best_params_path=args.xgb_best_params,
        results_dir=args.results_dir,
    )
    # Predict on val, internal test, and external test
    dval = xgb.DMatrix(pdata.X_val_xgb)
    dtest_internal = xgb.DMatrix(pdata.X_test_internal_xgb)
    dtest_external = xgb.DMatrix(pdata.X_test_external_xgb)
    xgb_pred_val_std = xgb_model.predict(dval)
    xgb_pred_test_internal_std = xgb_model.predict(dtest_internal)
    xgb_pred_test_external_std = xgb_model.predict(dtest_external)
    # Inverse to original scale
    import joblib
    y_scaler = joblib.load(os.path.join(args.results_dir, "encoder", "standard_scaler_y.pkl"))
    xgb_pred_val_orig = y_scaler.inverse_transform(xgb_pred_val_std.reshape(-1, 1)).ravel()
    xgb_pred_test_internal_orig = y_scaler.inverse_transform(xgb_pred_test_internal_std.reshape(-1, 1)).ravel()
    xgb_pred_test_external_orig = y_scaler.inverse_transform(xgb_pred_test_external_std.reshape(-1, 1)).ravel()
    _save_preds_csv(os.path.join(args.results_dir, "pred_xgb_retrain_val.csv"), pdata.y_val_orig, xgb_pred_val_orig, xgb_pred_val_std)
    _save_preds_csv(os.path.join(args.results_dir, "pred_xgb_retrain_test_internal.csv"), pdata.y_test_internal_orig, xgb_pred_test_internal_orig, xgb_pred_test_internal_std)
    _save_preds_csv(os.path.join(args.results_dir, "pred_xgb_retrain_test_external.csv"), pdata.y_test_external_orig, xgb_pred_test_external_orig, xgb_pred_test_external_std)

    # 2) LightGBM
    if args.lgbm_mode == "bayes":
        lgb_model, lgb_best_params, _ = bayesian_tune_lgbm(
            pdata.X_train_lgb,
            pdata.y_train_std,
            pdata.X_val_lgb,
            pdata.y_val_std,
            categorical_feature=pdata.categorical_cols,
            results_dir=args.results_dir,
        )
        lgb_model_name = "lgbm_bayes.txt"
    else:
        params = {}
        if args.lgbm_params:
            obj = _load_json_if_exists(args.lgbm_params)
            if obj:
                params = obj
        lgb_model = train_lgbm_manual(
            pdata.X_train_lgb,
            pdata.y_train_std,
            pdata.X_val_lgb,
            pdata.y_val_std,
            categorical_feature=pdata.categorical_cols,
            params=params,
            results_dir=args.results_dir,
        )
        lgb_model_name = "lgbm_manual.txt"

    lgb_pred_val_std = lgb_model.predict(pdata.X_val_lgb, num_iteration=getattr(lgb_model, "best_iteration", None))
    lgb_pred_test_internal_std = lgb_model.predict(pdata.X_test_internal_lgb, num_iteration=getattr(lgb_model, "best_iteration", None))
    lgb_pred_test_external_std = lgb_model.predict(pdata.X_test_external_lgb, num_iteration=getattr(lgb_model, "best_iteration", None))
    lgb_pred_val_orig = y_scaler.inverse_transform(np.array(lgb_pred_val_std).reshape(-1, 1)).ravel()
    lgb_pred_test_internal_orig = y_scaler.inverse_transform(np.array(lgb_pred_test_internal_std).reshape(-1, 1)).ravel()
    lgb_pred_test_external_orig = y_scaler.inverse_transform(np.array(lgb_pred_test_external_std).reshape(-1, 1)).ravel()
    _save_preds_csv(os.path.join(args.results_dir, "pred_lgbm_val.csv"), pdata.y_val_orig, lgb_pred_val_orig, np.array(lgb_pred_val_std))
    _save_preds_csv(os.path.join(args.results_dir, "pred_lgbm_test_internal.csv"), pdata.y_test_internal_orig, lgb_pred_test_internal_orig, np.array(lgb_pred_test_internal_std))
    _save_preds_csv(os.path.join(args.results_dir, "pred_lgbm_test_external.csv"), pdata.y_test_external_orig, lgb_pred_test_external_orig, np.array(lgb_pred_test_external_std))

    # 3) CatBoost
    if args.cat_mode == "bayes":
        cat_model, cat_best_params, _ = bayesian_tune_catboost(
            pdata.X_train_lgb,
            pdata.y_train_std,
            pdata.X_val_lgb,
            pdata.y_val_std,
            categorical_cols=pdata.categorical_cols,
            results_dir=args.results_dir,
        )
        cat_model_name = "cat_bayes.cbm"
    else:
        params = {}
        if args.cat_params:
            obj = _load_json_if_exists(args.cat_params)
            if obj:
                params = obj
        cat_model = train_cat_manual(
            pdata.X_train_lgb,
            pdata.y_train_std,
            pdata.X_val_lgb,
            pdata.y_val_std,
            categorical_cols=pdata.categorical_cols,
            params=params,
            results_dir=args.results_dir,
        )
        cat_model_name = "cat_manual.cbm"

    cat_pred_val_std = cat_model.predict(pdata.X_val_lgb)
    cat_pred_test_internal_std = cat_model.predict(pdata.X_test_internal_lgb)
    cat_pred_test_external_std = cat_model.predict(pdata.X_test_external_lgb)
    cat_pred_val_orig = y_scaler.inverse_transform(np.array(cat_pred_val_std).reshape(-1, 1)).ravel()
    cat_pred_test_internal_orig = y_scaler.inverse_transform(np.array(cat_pred_test_internal_std).reshape(-1, 1)).ravel()
    cat_pred_test_external_orig = y_scaler.inverse_transform(np.array(cat_pred_test_external_std).reshape(-1, 1)).ravel()
    _save_preds_csv(os.path.join(args.results_dir, "pred_cat_val.csv"), pdata.y_val_orig, cat_pred_val_orig, np.array(cat_pred_val_std))
    _save_preds_csv(os.path.join(args.results_dir, "pred_cat_test_internal.csv"), pdata.y_test_internal_orig, cat_pred_test_internal_orig, np.array(cat_pred_test_internal_std))
    _save_preds_csv(os.path.join(args.results_dir, "pred_cat_test_external.csv"), pdata.y_test_external_orig, cat_pred_test_external_orig, np.array(cat_pred_test_external_std))

    # 4) Ensemble on original scale (simple mean)
    # Validation
    ens_val = (xgb_pred_val_orig + lgb_pred_val_orig + cat_pred_val_orig) / 3.0
    df_ens_val = pd.DataFrame({
        "y_true_orig": pdata.y_val_orig,
        "y_pred_xgb_orig": xgb_pred_val_orig,
        "y_pred_lgbm_orig": lgb_pred_val_orig,
        "y_pred_cat_orig": cat_pred_val_orig,
        "y_pred_ensemble_orig": ens_val,
    })
    df_ens_val.to_csv(os.path.join(args.results_dir, "pred_ensemble_val.csv"), index=False)

    # Internal Test
    ens_test_internal = (xgb_pred_test_internal_orig + lgb_pred_test_internal_orig + cat_pred_test_internal_orig) / 3.0
    df_ens_test_internal = pd.DataFrame({
        "y_true_orig": pdata.y_test_internal_orig,
        "y_pred_xgb_orig": xgb_pred_test_internal_orig,
        "y_pred_lgbm_orig": lgb_pred_test_internal_orig,
        "y_pred_cat_orig": cat_pred_test_internal_orig,
        "y_pred_ensemble_orig": ens_test_internal,
    })
    df_ens_test_internal.to_csv(os.path.join(args.results_dir, "pred_ensemble_test_internal.csv"), index=False)

    # External Test
    ens_test_external = (xgb_pred_test_external_orig + lgb_pred_test_external_orig + cat_pred_test_external_orig) / 3.0
    df_ens_test_external = pd.DataFrame({
        "y_pred_xgb_orig": xgb_pred_test_external_orig,
        "y_pred_lgbm_orig": lgb_pred_test_external_orig,
        "y_pred_cat_orig": cat_pred_test_external_orig,
        "y_pred_ensemble_orig": ens_test_external,
    })
    if pdata.y_test_external_orig is not None:
        df_ens_test_external.insert(0, "y_true_orig", pdata.y_test_external_orig)
    df_ens_test_external.to_csv(os.path.join(args.results_dir, "pred_ensemble_test_external.csv"), index=False)

    # 5) Metrics (SMAPE on original scale)
    metrics_val = {
        "smape_xgb": smape_np(pdata.y_val_orig, xgb_pred_val_orig),
        "smape_lgbm": smape_np(pdata.y_val_orig, lgb_pred_val_orig),
        "smape_cat": smape_np(pdata.y_val_orig, cat_pred_val_orig),
        "smape_ensemble": smape_np(pdata.y_val_orig, ens_val),
    }
    save_json(os.path.join(args.results_dir, "metrics_val.json"), metrics_val)

    # Internal test metrics
    metrics_test_internal = {
        "smape_xgb": smape_np(pdata.y_test_internal_orig, xgb_pred_test_internal_orig),
        "smape_lgbm": smape_np(pdata.y_test_internal_orig, lgb_pred_test_internal_orig),
        "smape_cat": smape_np(pdata.y_test_internal_orig, cat_pred_test_internal_orig),
        "smape_ensemble": smape_np(pdata.y_test_internal_orig, ens_test_internal),
    }
    save_json(os.path.join(args.results_dir, "metrics_test_internal.json"), metrics_test_internal)

    # External test metrics (if available)
    if pdata.y_test_external_orig is not None:
        metrics_test_external = {
            "smape_xgb": smape_np(pdata.y_test_external_orig, xgb_pred_test_external_orig),
            "smape_lgbm": smape_np(pdata.y_test_external_orig, lgb_pred_test_external_orig),
            "smape_cat": smape_np(pdata.y_test_external_orig, cat_pred_test_external_orig),
            "smape_ensemble": smape_np(pdata.y_test_external_orig, ens_test_external),
        }
        save_json(os.path.join(args.results_dir, "metrics_test_external.json"), metrics_test_external)


if __name__ == "__main__":
    main()

