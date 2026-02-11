from typing import Tuple

import numpy as np
import pandas as pd


def expanding_window_validation(
    data: pd.DataFrame,
    model,
    horizon: int,
    history: int,
    start_train_size: int,
    step_size: int,
    id_col: str = "ts_id",
    timestamp_col: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Валидация с расширяющимся окном обучения.

    Args:
    - data: DataFrame с временными рядами.
    - model: не обученная модель для прогнозирования.
    - horizon: длина выходного окна (горизонт прогнозирования).
    - history: длина входного окна (история).
    - start_train_size: начальный размер обучающего набора.
    - step_size: шаг расширения окна обучения.
    - id_col: название столбца с идентификатором ряда.
    - timestamp_col: название столбца с временной меткой.
    - value_col: название столбца с значением ряда.

    Returns: DataFrame с истинными и предсказанными значениями c столбцами
        ['ts_id', 'fold', 'timestamp', 'true_value', 'predicted_value'].

    """
    res_df_list = []

    unique_timestamps = data[timestamp_col].sort_values().unique()
    n_timestamps = len(unique_timestamps)
    train_start_idx = 0
    tr_e_index = start_train_size
    val_start_idx = tr_e_index - history
    validate_e_index = tr_e_index + horizon
    test_start_idx = validate_e_index - history
    test_end_idx = validate_e_index + horizon

    while test_end_idx <= n_timestamps:
        data_masked = data.copy()
        data_masked[value_col] = data_masked[value_col].where(
            data_masked[timestamp_col] < unique_timestamps[test_start_idx + history], np.nan
        )

        train_mask = data_masked[timestamp_col] < unique_timestamps[tr_e_index]
        val_mask = (data_masked[timestamp_col] >= unique_timestamps[val_start_idx]) & (
                data_masked[timestamp_col] < unique_timestamps[validate_e_index]
        )
        test_mask = (data_masked[timestamp_col] >= unique_timestamps[test_start_idx]) & (
                data_masked[timestamp_col] < unique_timestamps[test_end_idx]
        )

        train_data = data_masked[train_mask]
        val_data = data_masked[val_mask]
        test_data = data_masked[test_mask]

        model.fit(train_data, val_data)
        predictions = model.predict(test_data)
        test_data_unmasked = data[test_mask]
        test_data = test_data[test_data[timestamp_col] >= unique_timestamps[validate_e_index]]
        test_data_unmasked = test_data_unmasked[
            test_data_unmasked[timestamp_col] >= unique_timestamps[validate_e_index]
        ]

        res_df = pd.DataFrame(
            {
                id_col: test_data[id_col].values,
                "fold": np.repeat(len(res_df_list), len(test_data)),
                timestamp_col: test_data[timestamp_col].values,
                "true_value": test_data_unmasked[value_col].values,
                "predicted_value": predictions["predicted_value"].values,
            }
        )
        res_df_list.append(res_df)
        tr_e_index += step_size
        val_start_idx = tr_e_index - history
        validate_e_index = tr_e_index + horizon
        test_start_idx = validate_e_index - history
        test_end_idx = validate_e_index + horizon


    res_df = pd.concat(res_df_list).reset_index(drop=True)

    return res_df
