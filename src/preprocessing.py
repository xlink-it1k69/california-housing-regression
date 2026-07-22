"""
preprocessing.py
================

Module chịu trách nhiệm:

1. Chia dữ liệu train/test
2. Chuẩn hóa feature bằng StandardScaler

Không thực hiện:
- Visualization
- Huấn luyện model
- Đánh giá model
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_data


TARGET_COLUMN = "MedHouseVal"


def split_features_target(df: pd.DataFrame):
    """
    Tách feature và target.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
):
    """
    Chia train/test.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )


def scale_features(
    X_train,
    X_test
):
    """
    Chuẩn hóa feature bằng StandardScaler.

    Chỉ fit trên train.
    Test chỉ transform.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


def preprocess_data():
    """
    Pipeline tiền xử lý hoàn chỉnh.

    Returns
    -------
    X_train
    X_test
    y_train
    y_test
    scaler
    """

    df = load_data()

    X, y = split_features_target(df)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_data()

    print("=" * 60)
    print("Preprocessing Completed")
    print("=" * 60)

    print()

    print("Train Feature Shape :", X_train.shape)
    print("Test Feature Shape  :", X_test.shape)

    print()

    print("Train Target Shape  :", y_train.shape)
    print("Test Target Shape   :", y_test.shape)

    print()

    print("Scaler Mean")
    print(scaler.mean_)

    print()

    print("Scaler Std")
    print(scaler.scale_)