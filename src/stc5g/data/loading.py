"""Loading of the windowed throughput CSV files, train/val/test split, scaling and sequencing.

The CSV files are those produced by the dataset stage (one sub-directory per class) with the
columns ``frame.time, frame.len, frame.time_delta_displayed, frame.time_epoch,
throughput_Mbps, class``.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from stc5g.config import (
    BATCH_SIZE,
    CSV_THROUGHPUT_DIR,
    DATA_DIR,
    DROPPED_COLUMNS,
    LOOKBACK,
    MODELS_DIR,
    TAIL_ROWS_DROPPED,
)


def make_sequences(
    features: np.ndarray, target: np.ndarray, lookback: int = LOOKBACK
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a time series into sliding windows for supervised learning.

    Args:
        features: Array of shape ``(n_rows, n_features)``.
        target: Array of shape ``(n_rows,)`` with the class of every row.
        lookback: Window length.

    Returns:
        ``(x, y)`` where ``x`` has shape ``(n_rows - lookback, lookback, n_features)`` and
        ``y[k]`` is the target of the row that immediately follows window ``k``. Both are
        ``float32``.
    """
    x = []
    y = []
    for i in range(lookback, len(features)):
        x.append(features[i - lookback : i])
        y.append(target[i])

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return x, y


def _make_loader(x, y, shuffle: bool) -> DataLoader:
    """Wrap ``(x, y)`` arrays in a ``float32`` TensorDataset and a DataLoader."""
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=shuffle)


def transform_data_loader(
    x_train, y_train, x_val, y_val, x_test, y_test
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Build the train/val/test DataLoaders (batch size 32).

    The training loader is shuffled; validation and test loaders are not. An empty
    validation or test split yields ``None`` instead of a loader.

    Returns:
        ``(train_loader, val_loader, test_loader)``.
    """
    train_loader = _make_loader(x_train, y_train, shuffle=True)
    val_loader = _make_loader(x_val, y_val, shuffle=False) if len(x_val) > 0 else None
    test_loader = _make_loader(x_test, y_test, shuffle=False) if len(x_test) > 0 else None
    return train_loader, val_loader, test_loader


def _read_capture(filepath: str, ignore_missing_columns: bool) -> pd.DataFrame:
    """Read one capture CSV, drop its last 30 rows and the unused columns."""
    df = pd.read_csv(filepath)
    df = df.iloc[:-TAIL_ROWS_DROPPED] if len(df) > TAIL_ROWS_DROPPED else df
    return df.drop(columns=DROPPED_COLUMNS, errors="ignore" if ignore_missing_columns else "raise")


def get_data(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    norm: int,
    throughput_dir: str = CSV_THROUGHPUT_DIR,
    test_dir: Optional[str] = None,
    *,
    splits_dir: Optional[str] = DATA_DIR,
    models_dir: str = MODELS_DIR,
):
    """Load every capture CSV, split each one chronologically and scale the features.

    Every CSV under ``throughput_dir`` is split *per file* in temporal order:
    ``train_end = int(n * train_ratio)`` and ``val_end = train_end + int(n * val_ratio)``;
    the remainder is the test part. When ``test_dir`` is given, ``throughput_dir`` is split
    only into train and validation (``train_ratio / (train_ratio + val_ratio)``) and every
    CSV under ``test_dir`` becomes test data. The per-file parts are then concatenated.

    Files are visited in ``os.walk`` order (not sorted), as in the original pipeline; the
    concatenation order therefore depends on the file system.

    Side effects:
        * Each per-file part is written to ``{splits_dir}/{train,val,test}/<file name>``
          (empty parts are skipped; the test part only without ``test_dir``). Pass
          ``splits_dir=None`` to disable the export.
        * If ``norm == 1``, the fitted scaler is pickled to
          ``{models_dir}/scaler_{train_ratio}-{val_ratio}-{test_ratio}.pkl``.

    Args:
        train_ratio: Fraction of every capture used for training.
        val_ratio: Fraction of every capture used for validation.
        test_ratio: Fraction used for testing (only used in the scaler file name; the test
            part is the remainder after train and validation).
        norm: ``1`` to fit a ``MinMaxScaler`` on the training split only and apply it to
            all splits; any other value leaves the features unscaled.
        throughput_dir: Root directory of the capture CSV files.
        test_dir: Optional independent directory with test CSV files.
        splits_dir: Destination of the exported splits, or ``None``.
        models_dir: Destination directory of the scaler pickle.

    Returns:
        ``(x_train, x_val, x_test, y_train, y_val, y_test)``. With ``norm == 1`` the ``x``
        values are NumPy arrays, otherwise DataFrames. ``y`` values are ``int`` arrays.
    """
    list_df_train: List[pd.DataFrame] = []
    list_df_val: List[pd.DataFrame] = []
    list_df_test: List[pd.DataFrame] = []
    for root, _, files in os.walk(throughput_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            df = _read_capture(os.path.join(root, file), ignore_missing_columns=False)
            n_rows = len(df)

            # With an independent test_dir, throughput_dir is shared between train and val.
            df_test = None
            if test_dir:
                total_train_val = train_ratio + val_ratio
                eff_train_ratio = train_ratio / total_train_val if total_train_val > 0 else 1.0
                train_end = int(n_rows * eff_train_ratio)

                df_train = df.iloc[:train_end]
                df_val = df.iloc[train_end:]
                list_df_train.append(df_train)
                list_df_val.append(df_val)
            else:
                train_end = int(n_rows * train_ratio)
                val_end = train_end + int(n_rows * val_ratio)

                df_train = df.iloc[:train_end]
                df_val = df.iloc[train_end:val_end]
                df_test = df.iloc[val_end:]

                list_df_train.append(df_train)
                list_df_val.append(df_val)
                list_df_test.append(df_test)

            if splits_dir is not None:
                os.makedirs(os.path.join(splits_dir, "train"), exist_ok=True)
                df_train.to_csv(os.path.join(splits_dir, "train", file), index=False)
                if len(df_val) > 0:
                    os.makedirs(os.path.join(splits_dir, "val"), exist_ok=True)
                    df_val.to_csv(os.path.join(splits_dir, "val", file), index=False)
                if df_test is not None and len(df_test) > 0:
                    os.makedirs(os.path.join(splits_dir, "test"), exist_ok=True)
                    df_test.to_csv(os.path.join(splits_dir, "test", file), index=False)

    # Independent test set.
    if test_dir and os.path.isdir(test_dir):
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.endswith(".csv"):
                    list_df_test.append(_read_capture(os.path.join(root, file), ignore_missing_columns=True))

    x_train = pd.concat(list_df_train)
    x_val = pd.concat(list_df_val)
    x_test = pd.concat(list_df_test)

    y_train = x_train["class"].values.astype(int)
    y_val = x_val["class"].values.astype(int)
    y_test = x_test["class"].values.astype(int)

    x_train = x_train.drop(columns="class")
    x_val = x_val.drop(columns="class")
    x_test = x_test.drop(columns="class")

    if norm == 1:
        mms = MinMaxScaler()
        mms.fit(x_train.values)
        x_train = mms.transform(x_train.values)
        x_val = mms.transform(x_val.values) if len(x_val) > 0 else x_val.values
        x_test = mms.transform(x_test.values) if len(x_test) > 0 else x_test.values

        # Keep the scaler for inference.
        scaler_path = os.path.join(models_dir, f"scaler_{train_ratio}-{val_ratio}-{test_ratio}.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(mms, f)

    return x_train, x_val, x_test, y_train, y_val, y_test
