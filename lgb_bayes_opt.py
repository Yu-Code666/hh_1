from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'lightgbm'. Install with: \n"
        "  pip install lightgbm\n"
        "Then re-run the tuning."
    ) from exc

try:
    from bayes_opt import BayesianOptimization
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'bayesian-optimization'. Install with: \n"
        "  pip install bayesian-optimization\n"
        "Then re-run the tuning."
    ) from exc


def lgbm_smape_eval(y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = dataset.get_label()
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)
    return "SMAPE", float(smape), False  # False -> lower is better


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bayesian_tune_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    categorical_feature: Optional[List[str]] = None,
    base_params: Optional[Dict] = None,
    init_points: int = 8,
    n_iter: int = 25,
    random_state: int = 42,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
    results_dir: str = "data_hh/result",
    save_model_name: Optional[str] = None,
):
    if base_params is None:
        base_params = {
            "objective": "regression",
            "metric": ["rmse"],
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,
        }

    default_bounds = {
        "num_leaves": (16, 512),  # int
        "max_depth": (3, 16),  # int; can map 16->-1 if desired
        "feature_fraction": (0.6, 1.0),
        "bagging_fraction": (0.6, 1.0),
        "lambda_l1": (0.0, 10.0),
        "lambda_l2": (0.0, 10.0),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (200, 1200),  # int
    }
    if bounds:
        default_bounds.update(bounds)

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature, free_raw_data=False)

    def train_evaluate(
        num_leaves,
        max_depth,
        feature_fraction,
        bagging_fraction,
        lambda_l1,
        lambda_l2,
        learning_rate,
        n_estimators,
    ):
        params = dict(base_params)
        params.update(
            {
                "num_leaves": int(round(num_leaves)),
                "max_depth": int(round(max_depth)),
                "feature_fraction": float(np.clip(feature_fraction, 0.0, 1.0)),
                "bagging_fraction": float(np.clip(bagging_fraction, 0.0, 1.0)),
                "lambda_l1": float(lambda_l1),
                "lambda_l2": float(lambda_l2),
                "learning_rate": float(learning_rate),
            }
        )

        num_boost_round = int(round(n_estimators))
        evals_result = {}
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "validation"],
            feval=lgbm_smape_eval,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0 if not verbose_eval else verbose_eval),
                lgb.record_evaluation(evals_result),
            ],
        )

        # Prefer booster.best_score if available; key: dataset name -> metric
        if booster.best_score and "validation" in booster.best_score and "SMAPE" in booster.best_score["validation"]:
            smape = float(booster.best_score["validation"]["SMAPE"])
        else:
            smape_series = evals_result.get("validation", {}).get("SMAPE")
            if not smape_series:
                raise RuntimeError("SMAPE not found in evals_result; check feval setup")
            smape = float(smape_series[-1])

        return -smape  # BayesianOptimization maximizes

    optimizer = BayesianOptimization(
        f=train_evaluate,
        pbounds=default_bounds,
        random_state=random_state,
        verbose=2,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = optimizer.max
    best_params = dict(best["params"])  # type: ignore[index]
    # Coerce types
    best_params["num_leaves"] = int(round(best_params["num_leaves"]))
    best_params["max_depth"] = int(round(best_params["max_depth"]))
    best_params["n_estimators"] = int(round(best_params["n_estimators"]))

    # Train final model with best params
    final_params = dict(base_params)
    final_params.update(
        {
            "num_leaves": best_params["num_leaves"],
            "max_depth": best_params["max_depth"],
            "feature_fraction": float(best_params["feature_fraction"]),
            "bagging_fraction": float(best_params["bagging_fraction"]),
            "lambda_l1": float(best_params["lambda_l1"]),
            "lambda_l2": float(best_params["lambda_l2"]),
            "learning_rate": float(best_params["learning_rate"]),
        }
    )

    booster = lgb.train(
        final_params,
        dtrain,
        num_boost_round=int(best_params["n_estimators"]),
        valid_sets=[dtrain, dval],
        valid_names=["train", "validation"],
        feval=lgbm_smape_eval,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=verbose_eval),
        ],
    )

    # Persist
    _ensure_dir(results_dir)
    _ensure_dir(os.path.join(results_dir, "model"))
    trials_csv = os.path.join(results_dir, "lgbm_bayes_trials.csv")
    best_json = os.path.join(results_dir, "lgbm_bayes_best_params.json")
    model_name = save_model_name or "lgbm_bayes.txt"
    model_path = os.path.join(results_dir, "model", model_name)

    # Save trials
    try:
        fieldnames = [
            "num_leaves",
            "max_depth",
            "feature_fraction",
            "bagging_fraction",
            "lambda_l1",
            "lambda_l2",
            "learning_rate",
            "n_estimators",
            "smape",
        ]
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in optimizer.res:  # type: ignore[attr-defined]
                row = dict(res["params"])  # type: ignore[index]
                row["smape"] = -float(res["target"])  # back to SMAPE
                writer.writerow(row)
    except Exception:
        pass

    # Save best params
    try:
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump({"best_params": best_params, "best_smape": -float(best["target"])}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Save model
    try:
        booster.save_model(model_path)
    except Exception:
        pass

    return booster, best_params, -float(best["target"])  # model, params, smape


def train_lgbm_manual(
    X_train,
    y_train,
    X_val,
    y_val,
    categorical_feature: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
    results_dir: str = "data_hh/result",
    save_model_name: Optional[str] = None,
):
    base_params = {
        "objective": "regression",
        "metric": ["rmse"],
        "verbosity": -1,
        "seed": 42,
        "feature_pre_filter": False,
    }
    if params:
        base_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feature, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feature, free_raw_data=False)

    booster = lgb.train(
        base_params,
        dtrain,
        num_boost_round=int(n_estimators),
        valid_sets=[dtrain, dval],
        valid_names=["train", "validation"],
        feval=lgbm_smape_eval,
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=verbose_eval),
        ],
    )

    _ensure_dir(results_dir)
    _ensure_dir(os.path.join(results_dir, "model"))
    model_name = save_model_name or "lgbm_manual.txt"
    model_path = os.path.join(results_dir, "model", model_name)
    try:
        booster.save_model(model_path)
    except Exception:
        pass
    return booster


__all__ = [
    "lgbm_smape_eval",
    "bayesian_tune_lgbm",
    "train_lgbm_manual",
]

