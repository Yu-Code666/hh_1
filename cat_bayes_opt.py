from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


try:
    from catboost import CatBoostRegressor, Pool
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'catboost'. Install with: \n"
        "  pip install catboost\n"
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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _cat_feature_indices(df, categorical_cols: Optional[List[str]]) -> List[int]:
    if categorical_cols is None:
        # Infer: category dtype or object
        return [i for i, c in enumerate(df.columns) if str(df[c].dtype) in ("category", "object")]
    col_to_idx = {c: i for i, c in enumerate(df.columns)}
    return [col_to_idx[c] for c in categorical_cols if c in col_to_idx]


def bayesian_tune_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    categorical_cols: Optional[List[str]] = None,
    base_params: Optional[Dict] = None,
    init_points: int = 8,
    n_iter: int = 25,
    random_state: int = 42,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
    results_dir: str = "data_hh/result",
    save_model_name: Optional[str] = None,
):
    if base_params is None:
        base_params = {
            "loss_function": "RMSE",
            "random_state": random_state,
            "allow_writing_files": False,
            "thread_count": -1,
        }

    default_bounds = {
        "depth": (4, 10),  # int
        "l2_leaf_reg": (0.0, 10.0),
        "bagging_temperature": (0.0, 10.0),
        "learning_rate": (0.01, 0.3),
        "iterations": (200, 1200),  # int
    }
    if bounds:
        default_bounds.update(bounds)

    cat_features_idx = _cat_feature_indices(X_train, categorical_cols)
    dtrain = Pool(X_train, label=y_train, cat_features=cat_features_idx)
    dval = Pool(X_val, label=y_val, cat_features=cat_features_idx)

    def train_evaluate(
        depth,
        l2_leaf_reg,
        bagging_temperature,
        learning_rate,
        iterations,
    ):
        params = dict(base_params)
        params.update(
            {
                "depth": int(round(depth)),
                "l2_leaf_reg": float(l2_leaf_reg),
                "bagging_temperature": float(bagging_temperature),
                "learning_rate": float(learning_rate),
                "iterations": int(round(iterations)),
                "od_type": "Iter",
                "od_wait": int(early_stopping_rounds),
            }
        )

        model = CatBoostRegressor(**params)
        model.fit(
            dtrain,
            eval_set=dval,
            verbose=False if verbose is None or verbose == 0 else verbose,
            use_best_model=True,
        )
        preds = model.predict(dval)
        smape = _smape_np(y_val, preds)
        return -smape

    optimizer = BayesianOptimization(
        f=train_evaluate,
        pbounds=default_bounds,
        random_state=random_state,
        verbose=2,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = optimizer.max
    best_params = dict(best["params"])  # type: ignore[index]
    best_params["depth"] = int(round(best_params["depth"]))
    best_params["iterations"] = int(round(best_params["iterations"]))

    final_params = dict(base_params)
    final_params.update(best_params)
    final_params.update({"od_type": "Iter", "od_wait": int(early_stopping_rounds)})

    model = CatBoostRegressor(**final_params)
    model.fit(
        dtrain,
        eval_set=dval,
        verbose=verbose,
        use_best_model=True,
    )

    # Persist
    _ensure_dir(results_dir)
    _ensure_dir(os.path.join(results_dir, "model"))
    trials_csv = os.path.join(results_dir, "cat_bayes_trials.csv")
    best_json = os.path.join(results_dir, "cat_bayes_best_params.json")
    model_name = save_model_name or "cat_bayes.cbm"
    model_path = os.path.join(results_dir, "model", model_name)

    try:
        fieldnames = ["depth", "l2_leaf_reg", "bagging_temperature", "learning_rate", "iterations", "smape"]
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in optimizer.res:  # type: ignore[attr-defined]
                row = dict(res["params"])  # type: ignore[index]
                row["smape"] = -float(res["target"])  # back to SMAPE
                writer.writerow(row)
    except Exception:
        pass

    try:
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump({"best_params": best_params, "best_smape": -float(best["target"])}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        model.save_model(model_path)
    except Exception:
        pass

    return model, best_params, -float(best["target"])  # model, params, smape


def train_cat_manual(
    X_train,
    y_train,
    X_val,
    y_val,
    categorical_cols: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    iterations: int = 500,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
    results_dir: str = "data_hh/result",
    save_model_name: Optional[str] = None,
):
    base_params = {
        "loss_function": "RMSE",
        "random_state": 42,
        "allow_writing_files": False,
        "thread_count": -1,
        "iterations": int(iterations),
    }
    if params:
        base_params.update(params)

    cat_features_idx = _cat_feature_indices(X_train, categorical_cols)
    dtrain = Pool(X_train, label=y_train, cat_features=cat_features_idx)
    dval = Pool(X_val, label=y_val, cat_features=cat_features_idx)

    model = CatBoostRegressor(**base_params)
    model.fit(
        dtrain,
        eval_set=dval,
        verbose=verbose,
        use_best_model=True,
        early_stopping_rounds=early_stopping_rounds,
    )

    _ensure_dir(results_dir)
    _ensure_dir(os.path.join(results_dir, "model"))
    model_name = save_model_name or "cat_manual.cbm"
    model_path = os.path.join(results_dir, "model", model_name)
    try:
        model.save_model(model_path)
    except Exception:
        pass
    return model


__all__ = [
    "bayesian_tune_catboost",
    "train_cat_manual",
]

