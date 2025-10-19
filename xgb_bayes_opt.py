import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb


try:
    # pip install bayesian-optimization
    from bayes_opt import BayesianOptimization
except Exception as exc:  # pragma: no cover - import-time guidance only
    raise ImportError(
        "Missing dependency 'bayesian-optimization'. Install with: \n"
        "  pip install bayesian-optimization\n"
        "Then re-run the tuning."
    ) from exc


def smape_eval(y_pred: np.ndarray, dtrain: xgb.DMatrix):
    """Custom SMAPE evaluation for XGBoost (feval API).

    XGBoost expects a 2-tuple (name, value) for feval. We minimize SMAPE by
    passing maximize=False in xgb.train.
    """
    y_true = dtrain.get_label()
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    smape = 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)
    return "SMAPE", float(smape)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bayesian_tune_xgb(
    dtrain: xgb.DMatrix,
    evals: List[Tuple[xgb.DMatrix, str]],
    base_params: Optional[Dict] = None,
    init_points: int = 8,
    n_iter: int = 25,
    random_state: int = 42,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    early_stopping_rounds: int = 10,
    verbose_eval: int = 50,
    results_dir: str = "data_hh/result",
    save_model_name: Optional[str] = None,
):
    """Tune key XGBoost hyperparameters using Bayesian Optimization.

    Parameters
    - dtrain: training DMatrix
    - evals: list of (DMatrix, name) watched for early stopping; first is used for score
    - base_params: fixed XGBoost params (e.g., {"objective": "reg:squarederror"})
    - init_points: random init samples
    - n_iter: Bayesian iterations
    - random_state: RNG seed
    - bounds: override search bounds
    - early_stopping_rounds: early stop rounds during evaluation
    - verbose_eval: xgboost training verbose interval (or Falsey for quiet)
    - results_dir: directory to persist best params and trials CSV
    - save_model_name: final model filename (under results_dir/model)

    Returns
    - best_model: trained xgb.Booster with best params
    - best_params: dict of best hyperparameters (including num_boost_round)
    - best_smape: float SMAPE on the chosen eval set
    """

    if base_params is None:
        base_params = {
            "objective": "reg:squarederror",
        }

    # Default search space; int-cast where noted below
    default_bounds = {
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 12),  # int
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "alpha": (0.0, 10.0),
        "lambda_": (0.0, 10.0),  # use lambda_ here; map to 'lambda' for xgboost
        "num_boost_round": (200, 1200),  # int
    }
    if bounds:
        # allow partial override
        default_bounds.update(bounds)

    # Use validation set if provided (second item), otherwise the first
    if not evals:
        raise ValueError("'evals' must be a non-empty list of (DMatrix, name)")
    score_eval_idx = 1 if len(evals) > 1 else 0

    def train_evaluate(
        learning_rate,
        max_depth,
        subsample,
        colsample_bytree,
        alpha,
        lambda_,
        num_boost_round,
    ):
        params = dict(base_params)
        params.update(
            {
                "learning_rate": float(learning_rate),
                "max_depth": int(round(max_depth)),
                "subsample": float(np.clip(subsample, 0.0, 1.0)),
                "colsample_bytree": float(np.clip(colsample_bytree, 0.0, 1.0)),
                "alpha": float(alpha),
                "lambda": float(lambda_),
            }
        )

        n_rounds = int(round(num_boost_round))
        evals_result = {}

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            feval=smape_eval,
            maximize=False,  # minimize SMAPE
            evals_result=evals_result,
            verbose_eval=False if not verbose_eval else verbose_eval,
        )

        # Prefer Booster.best_score when available (requires early stopping)
        if getattr(booster, "best_score", None) is not None:
            smape = float(booster.best_score)
        else:
            # Fallback to last SMAPE from the chosen eval set
            chosen_name = evals[score_eval_idx][1]
            smape_series = evals_result.get(chosen_name, {}).get("SMAPE")
            if not smape_series:
                raise RuntimeError("SMAPE not found in evals_result; check feval setup")
            smape = float(smape_series[-1])

        # BayesianOptimization maximizes; we want to minimize SMAPE -> return negative
        return -smape

    optimizer = BayesianOptimization(
        f=train_evaluate,
        pbounds=default_bounds,
        random_state=random_state,
        verbose=2,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_entry = optimizer.max
    best_target = float(best_entry["target"])  # negative SMAPE
    best_smape = -best_target

    # Extract and coerce best params
    best_params = dict(best_entry["params"])  # type: ignore[index]
    best_params["max_depth"] = int(round(best_params["max_depth"]))
    best_params["num_boost_round"] = int(round(best_params["num_boost_round"]))
    # Rename lambda_
    best_params["lambda"] = float(best_params.pop("lambda_"))

    # Compose final xgboost params
    final_params = dict(base_params)
    final_params.update(
        {
            "learning_rate": float(best_params["learning_rate"]),
            "max_depth": int(best_params["max_depth"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "alpha": float(best_params["alpha"]),
            "lambda": float(best_params["lambda"]),
        }
    )

    # Train final model with the tuned configuration
    final_model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=int(best_params["num_boost_round"]),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        feval=smape_eval,
        maximize=False,
        verbose_eval=verbose_eval,
    )

    # Persist outputs under results_dir
    _ensure_dir(results_dir)
    trials_csv = os.path.join(results_dir, "xgb_bayes_trials.csv")
    best_json = os.path.join(results_dir, "xgb_bayes_best_params.json")
    model_dir = os.path.join(results_dir, "model")
    _ensure_dir(model_dir)

    # Save trials
    try:
        import csv

        fieldnames = [
            "learning_rate",
            "max_depth",
            "subsample",
            "colsample_bytree",
            "alpha",
            "lambda_",
            "num_boost_round",
            "smape",
        ]
        with open(trials_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in optimizer.res:
                row = dict(res["params"])  # type: ignore[index]
                row["smape"] = -float(res["target"])  # convert back to SMAPE
                writer.writerow(row)
    except Exception:
        # Non-fatal if writing trials fails
        pass

    # Save best params JSON
    try:
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump({"best_params": best_params, "best_smape": best_smape}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Save final model if requested
    if save_model_name is None:
        save_model_name = "xgb_bayes_model.json"
    model_path = os.path.join(model_dir, save_model_name)
    try:
        final_model.save_model(model_path)
    except Exception:
        pass

    return final_model, best_params, best_smape


__all__ = [
    "smape_eval",
    "bayesian_tune_xgb",
]
