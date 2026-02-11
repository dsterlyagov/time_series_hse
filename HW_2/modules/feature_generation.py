from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .index_slicing import get_cols_idx, get_slice


def get_features_df_and_targets(
    df: pd.DataFrame,
    features_ids,
    targets_ids,
    id_column: Union[str, Sequence[str]] = "id",
    date_column: Union[str, Sequence[str]] = "datetime",
    target_column: str = "target",
):
    df = df.copy()

    # календарные признаки
    df["minute"] = df[date_column].dt.minute
    df["hour"] = df[date_column].dt.hour
    df["day"] = df[date_column].dt.day
    df["month"] = df[date_column].dt.month
    df["quarter"] = df[date_column].dt.quarter
    df["year"] = df[date_column].dt.year
    date_columns = ["minute", "hour", "day", "month", "quarter", "year"]

    features_df_id = get_slice(df, (targets_ids, get_cols_idx(df, id_column)))
    features_time = get_slice(df, (targets_ids, get_cols_idx(df, date_columns)))
    features_lags = get_slice(df, (features_ids, get_cols_idx(df, target_column)))
    lags = features_lags
    window_feats = []

    for w in [3, 6, 12]:
        w = min(w, lags.shape[1])
        tail = lags[:, -w:]  # последние w лагов
        window_feats.append(tail.mean(axis=1, keepdims=True))  # roll_mean_w
        window_feats.append(tail.std(axis=1, keepdims=True))   # roll_std_w

    features_windows = np.hstack(window_feats)  # shape: (n_samples, 2*len(windows))
    features = np.hstack([features_df_id, features_time, features_lags, features_windows])
    categorical_features_idx = np.arange(features_df_id.shape[1] + features_time.shape[1])
    features_obj = features.astype(object)
    for j in categorical_features_idx:
        features_obj[:, j] = features_obj[:, j].astype(str)

    targets = get_slice(df, (targets_ids, get_cols_idx(df, target_column)))
    return features_obj, targets, categorical_features_idx

