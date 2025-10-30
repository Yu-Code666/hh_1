from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


RESULT_DIR = "data_hh/result"
SPLIT_PATH = os.path.join(RESULT_DIR, "splits", "train_val_indices.json")
ENCODER_DIR = os.path.join(RESULT_DIR, "encoder")
SCALER_Y_PATH = os.path.join(ENCODER_DIR, "standard_scaler_y.pkl")
SCALER_X_NUM_PATH = os.path.join(ENCODER_DIR, "standard_scaler_X_numeric.pkl")


NUMERIC_CANDIDATES: List[str] = [
    # Base
    "cap",
    "legs",
    "leg_no",
    "duration",
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "minute",
    "quarter",
    "is_weekend",
    "is_holiday_season",
    # Price
    "unit_price",
    "competitor_price",
    "comp_price_diff_abs",
    "comp_price_ratio",
    "comp_price_ratio_abs",
    # Stats computed on train only, then merged
    "from_avg_pax_month",
    "to_avg_pax_month",
    "seg_avg_pax",
    # Cyclic
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Embeddings (optional)
    "a_embedding_1",
    "a_embedding_2",
    "b_embedding_1",
    "b_embedding_2",
    "c_embedding_1",
    "c_embedding_2",
    "from_embedding_1",
    "from_embedding_2",
    "to_embedding_1",
    "to_embedding_2",
    # Graph labels (existing numeric codes)
    "a_label",
    "b_label",
    "c_label",
    "from_label",
    "to_label",
]


CATEGORICAL_CANDIDATES: List[str] = [
    "flt_no",
    "aircraft",
    "a",
    "b",
    "c",
    "from",
    "to",
]


TARGET_COL = "pax"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _compute_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add simple cyclic features if base columns exist and cyclic not yet present.
    out = df.copy()
    if "hour" in out.columns and "hour_sin" not in out.columns:
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"].astype(float) / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"].astype(float) / 24.0)
    if "month" in out.columns and "month_sin" not in out.columns:
        out["month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
        out["month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    return out


def load_train_test(train_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    # Optional cyclic features
    train_df = _compute_cyclic_features(train_df)
    test_df = _compute_cyclic_features(test_df)
    return train_df, test_df


def persist_splits(index_train: List[int], index_val: List[int], index_test: List[int], uids: Optional[List] = None) -> None:
    _ensure_dir(os.path.dirname(SPLIT_PATH))
    payload = {"train": list(map(int, index_train)), "val": list(map(int, index_val)), "test": list(map(int, index_test))}
    if uids is not None:
        payload["uids"] = list(uids)
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_splits() -> Optional[Dict[str, List[int]]]:
    if not os.path.exists(SPLIT_PATH):
        return None
    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not ("train" in data and "val" in data and "test" in data):
        return None
    return {"train": data["train"], "val": data["val"], "test": data["test"]}


def create_or_load_splits(df_train: pd.DataFrame, val_size: float = 0.1, test_size: float = 0.1, random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = load_splits()
    if loaded is not None:
        train_idx = np.array(loaded["train"], dtype=int)
        val_idx = np.array(loaded["val"], dtype=int)
        test_idx = np.array(loaded["test"], dtype=int)
        return train_idx, val_idx, test_idx

    # Deterministic 80:10:10 split using index positions
    # First split: 80% train, 20% temp
    all_idx = np.arange(len(df_train))
    temp_size = val_size + test_size
    idx_train, idx_temp = train_test_split(
        all_idx, test_size=temp_size, random_state=random_state, shuffle=True
    )
    # Second split: split temp into val and test (50:50 of temp = 10:10 of total)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=random_state, shuffle=True
    )
    persist_splits(idx_train.tolist(), idx_val.tolist(), idx_test.tolist())
    return np.array(idx_train, dtype=int), np.array(idx_val, dtype=int), np.array(idx_test, dtype=int)


def _add_group_stat(
    base: pd.DataFrame,
    ref: pd.DataFrame,
    keys: List[str],
    value_col: str,
    new_col: str,
) -> pd.DataFrame:
    # Compute only on ref (usually train split), then merge into base by keys
    if not all(k in ref.columns for k in keys):
        return base
    if value_col not in ref.columns:
        return base
    grp = ref.groupby(keys)[value_col].mean().reset_index().rename(columns={value_col: new_col})
    return base.merge(grp, on=keys, how="left")


def add_leakage_safe_stats(
    train_df: pd.DataFrame, other_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Compute on train_df only, then merge to both train_df and other_df
    t_keys = []
    out_train = train_df.copy()
    out_other = other_df.copy()

    # from_avg_pax_month: (from, month)
    if all(k in train_df.columns for k in ["from", "month", TARGET_COL]):
        grp = train_df.groupby(["from", "month"])[TARGET_COL].mean().reset_index().rename(
            columns={TARGET_COL: "from_avg_pax_month"}
        )
        out_train = out_train.merge(grp, on=["from", "month"], how="left")
        out_other = out_other.merge(grp, on=["from", "month"], how="left")

    # to_avg_pax_month: (to, month)
    if all(k in train_df.columns for k in ["to", "month", TARGET_COL]):
        grp = train_df.groupby(["to", "month"])[TARGET_COL].mean().reset_index().rename(
            columns={TARGET_COL: "to_avg_pax_month"}
        )
        out_train = out_train.merge(grp, on=["to", "month"], how="left")
        out_other = out_other.merge(grp, on=["to", "month"], how="left")

    # seg_avg_pax: (from, to)
    if all(k in train_df.columns for k in ["from", "to", TARGET_COL]):
        grp = train_df.groupby(["from", "to"])[TARGET_COL].mean().reset_index().rename(
            columns={TARGET_COL: "seg_avg_pax"}
        )
        out_train = out_train.merge(grp, on=["from", "to"], how="left")
        out_other = out_other.merge(grp, on=["from", "to"], how="left")

    return out_train, out_other


@dataclass
class PreparedData:
    X_train_lgb: pd.DataFrame
    X_val_lgb: pd.DataFrame
    X_test_internal_lgb: pd.DataFrame
    X_test_external_lgb: pd.DataFrame
    X_train_xgb: pd.DataFrame
    X_val_xgb: pd.DataFrame
    X_test_internal_xgb: pd.DataFrame
    X_test_external_xgb: pd.DataFrame
    y_train_std: np.ndarray
    y_val_std: np.ndarray
    y_test_internal_std: np.ndarray
    y_test_external_std: Optional[np.ndarray]
    y_train_orig: np.ndarray
    y_val_orig: np.ndarray
    y_test_internal_orig: np.ndarray
    y_test_external_orig: Optional[np.ndarray]
    numeric_cols: List[str]
    categorical_cols: List[str]
    xgb_categorical_cols: List[str]


def _fit_scalers(
    y_train: np.ndarray, X_train_num: pd.DataFrame
) -> Tuple[StandardScaler, StandardScaler]:
    y_scaler = StandardScaler()
    X_scaler = StandardScaler()
    y_scaler.fit(y_train.reshape(-1, 1))
    if not X_train_num.empty:
        X_scaler.fit(X_train_num.values)
    else:
        # Fit on a dummy vector to allow transform calls later
        X_scaler.fit(np.zeros((1, 1)))
    _ensure_dir(ENCODER_DIR)
    import joblib

    joblib.dump(y_scaler, SCALER_Y_PATH)
    joblib.dump(X_scaler, SCALER_X_NUM_PATH)
    return y_scaler, X_scaler


def _load_scalers() -> Tuple[Optional[StandardScaler], Optional[StandardScaler]]:
    try:
        import joblib

        y_scaler = joblib.load(SCALER_Y_PATH) if os.path.exists(SCALER_Y_PATH) else None
        X_scaler = joblib.load(SCALER_X_NUM_PATH) if os.path.exists(SCALER_X_NUM_PATH) else None
        return y_scaler, X_scaler
    except Exception:
        return None, None


def _factorize_series(train_s: pd.Series, full_s: pd.Series) -> Tuple[pd.Series, Dict]:
    # Fit on full series categories to reduce unknowns, but return codes aligned to full_s
    cats = pd.Series(full_s.astype(str).unique())
    cat_to_code = {cat: i for i, cat in enumerate(cats)}
    def map_codes(s: pd.Series) -> pd.Series:
        return s.astype(str).map(cat_to_code).fillna(-1).astype(int)
    return map_codes(full_s), cat_to_code


def prepare_datasets(
    train_csv: str,
    test_csv: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = RANDOM_STATE,
    exclude_flt_no: bool = False,
) -> PreparedData:
    """Load CSVs, create deterministic 80:10:10 split, compute features, and return
    model-ready DataFrames and arrays.
    """
    _ensure_dir(RESULT_DIR)
    _ensure_dir(ENCODER_DIR)
    train_df_raw, test_external_df_raw = load_train_test(train_csv, test_csv)

    # Make 80:10:10 splits on the TRAIN file
    idx_train, idx_val, idx_test_internal = create_or_load_splits(train_df_raw, val_size=val_size, test_size=test_size, random_state=random_state)
    df_train_split = train_df_raw.iloc[idx_train].reset_index(drop=True)
    df_val_split = train_df_raw.iloc[idx_val].reset_index(drop=True)
    df_test_internal_split = train_df_raw.iloc[idx_test_internal].reset_index(drop=True)

    # Compute leakage-safe stats on TRAIN split only and merge into others
    df_train_feat, df_val_feat = add_leakage_safe_stats(df_train_split, df_val_split)
    _, df_test_internal_feat = add_leakage_safe_stats(df_train_split, df_test_internal_split)
    _, df_test_external_feat = add_leakage_safe_stats(df_train_split, test_external_df_raw)

    # Ensure graph label columns exist before determining numeric cols
    for raw_col, lbl_col in [("a", "a_label"), ("b", "b_label"), ("c", "c_label"), ("from", "from_label"), ("to", "to_label")]:
        if lbl_col not in df_train_feat.columns and raw_col in df_train_feat.columns:
            combined = pd.concat([df_train_feat[raw_col], df_val_feat[raw_col], df_test_internal_feat[raw_col], df_test_external_feat[raw_col]], axis=0)
            codes, uniques = pd.factorize(combined.astype(str), sort=True)
            df_train_feat[lbl_col] = codes[: len(df_train_feat)]
            df_val_feat[lbl_col] = codes[len(df_train_feat) : len(df_train_feat) + len(df_val_feat)]
            df_test_internal_feat[lbl_col] = codes[len(df_train_feat) + len(df_val_feat) : len(df_train_feat) + len(df_val_feat) + len(df_test_internal_feat)]
            df_test_external_feat[lbl_col] = codes[len(df_train_feat) + len(df_val_feat) + len(df_test_internal_feat) :]

    # Determine column sets actually present (after creating label cols)
    numeric_cols = _safe_cols(pd.concat([df_train_feat, df_val_feat, df_test_internal_feat, df_test_external_feat], axis=0), NUMERIC_CANDIDATES)
    categorical_cols = _safe_cols(pd.concat([df_train_feat, df_val_feat, df_test_internal_feat, df_test_external_feat], axis=0), CATEGORICAL_CANDIDATES)
    if exclude_flt_no and "flt_no" in categorical_cols:
        categorical_cols.remove("flt_no")

    # Build LGB/Cat features: numeric + raw categories (category dtype)
    def cast_categories(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in categorical_cols:
            if c in out.columns:
                out[c] = out[c].astype("category")
        return out

    df_train_lgb = cast_categories(df_train_feat)
    df_val_lgb = cast_categories(df_val_feat)
    df_test_internal_lgb = cast_categories(df_test_internal_feat)
    df_test_external_lgb = cast_categories(df_test_external_feat)

    # Targets
    if TARGET_COL not in df_train_lgb.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in train CSV")
    y_train = df_train_lgb[TARGET_COL].to_numpy(dtype=float)
    y_val = df_val_lgb[TARGET_COL].to_numpy(dtype=float)
    y_test_internal = df_test_internal_lgb[TARGET_COL].to_numpy(dtype=float)
    y_test_external = None
    if TARGET_COL in df_test_external_lgb.columns:
        y_test_external = df_test_external_lgb[TARGET_COL].to_numpy(dtype=float)

    # Standardize y and numeric X (fit on TRAIN split only)
    X_train_num = df_train_lgb[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df_train_lgb.index)
    y_scaler, X_scaler = _fit_scalers(y_train, X_train_num)

    def transform_numeric(df: pd.DataFrame) -> pd.DataFrame:
        if not numeric_cols:
            return pd.DataFrame(index=df.index)
        Xn = df[numeric_cols].copy()
        Xn[:] = X_scaler.transform(Xn.values)
        Xn.columns = numeric_cols
        return Xn

    X_train_num_std = transform_numeric(df_train_lgb)
    X_val_num_std = transform_numeric(df_val_lgb)
    X_test_internal_num_std = transform_numeric(df_test_internal_lgb)
    X_test_external_num_std = transform_numeric(df_test_external_lgb)

    y_train_std = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_std = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_internal_std = y_scaler.transform(y_test_internal.reshape(-1, 1)).ravel()
    y_test_external_std = y_scaler.transform(y_test_external.reshape(-1, 1)).ravel() if y_test_external is not None else None

    # Combine numeric (scaled) + categorical for LGBM/CatBoost
    X_train_lgb = pd.concat([X_train_num_std, df_train_lgb[categorical_cols]], axis=1)
    X_val_lgb = pd.concat([X_val_num_std, df_val_lgb[categorical_cols]], axis=1)
    X_test_internal_lgb = pd.concat([X_test_internal_num_std, df_test_internal_lgb[categorical_cols]], axis=1)
    X_test_external_lgb = pd.concat([X_test_external_num_std, df_test_external_lgb[categorical_cols]], axis=1)

    # XGBoost features: numeric (scaled) + encoded categorical
    xgb_extra_cols: List[str] = []

    # Encode flt_no and aircraft to new columns
    full = pd.concat([df_train_feat, df_val_feat, df_test_internal_feat, df_test_external_feat], axis=0)
    for col, new_col in [("flt_no", "flt_no_encoded"), ("aircraft", "aircraft_encoded")]:
        if col in full.columns:
            cats = full[col].astype(str)
            cat_to_code = {cat: i for i, cat in enumerate(cats.unique())}
            def map_one(df: pd.DataFrame) -> pd.Series:
                return df[col].astype(str).map(cat_to_code).fillna(-1).astype(int)
            df_train_feat[new_col] = map_one(df_train_feat)
            df_val_feat[new_col] = map_one(df_val_feat)
            df_test_internal_feat[new_col] = map_one(df_test_internal_feat)
            df_test_external_feat[new_col] = map_one(df_test_external_feat)
            xgb_extra_cols.append(new_col)

    # XGB feature columns are numeric scaled + extra encoded columns
    xgb_numeric_cols = list(dict.fromkeys(numeric_cols))  # stable unique
    X_train_xgb = pd.concat([X_train_num_std, df_train_feat[xgb_extra_cols]], axis=1)
    X_val_xgb = pd.concat([X_val_num_std, df_val_feat[xgb_extra_cols]], axis=1)
    X_test_internal_xgb = pd.concat([X_test_internal_num_std, df_test_internal_feat[xgb_extra_cols]], axis=1)
    X_test_external_xgb = pd.concat([X_test_external_num_std, df_test_external_feat[xgb_extra_cols]], axis=1)

    return PreparedData(
        X_train_lgb=X_train_lgb,
        X_val_lgb=X_val_lgb,
        X_test_internal_lgb=X_test_internal_lgb,
        X_test_external_lgb=X_test_external_lgb,
        X_train_xgb=X_train_xgb,
        X_val_xgb=X_val_xgb,
        X_test_internal_xgb=X_test_internal_xgb,
        X_test_external_xgb=X_test_external_xgb,
        y_train_std=y_train_std,
        y_val_std=y_val_std,
        y_test_internal_std=y_test_internal_std,
        y_test_external_std=y_test_external_std,
        y_train_orig=y_train,
        y_val_orig=y_val,
        y_test_internal_orig=y_test_internal,
        y_test_external_orig=y_test_external,
        numeric_cols=xgb_numeric_cols,
        categorical_cols=categorical_cols,
        xgb_categorical_cols=xgb_extra_cols,
    )


__all__ = [
    "RESULT_DIR",
    "SPLIT_PATH",
    "ENCODER_DIR",
    "SCALER_Y_PATH",
    "SCALER_X_NUM_PATH",
    "prepare_datasets",
    "PreparedData",
    "NUMERIC_CANDIDATES",
    "CATEGORICAL_CANDIDATES",
    "TARGET_COL",
]
