"""
evaluate.py
===========

Module đánh giá các mô hình Regression.

Bao gồm:

- MSE
- RMSE
- MAE
- R²

Ngoài ra còn có hàm:

- print_metrics()
- compare_models()
"""

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import numpy as np


# ----------------------------------------------------
# Metric Functions
# ----------------------------------------------------

def calculate_mse(y_true, y_pred):
    """
    Mean Squared Error
    """

    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """

    mse = mean_squared_error(y_true, y_pred)

    return np.sqrt(mse)


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error
    """

    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true, y_pred):
    """
    R² Score
    """

    return r2_score(y_true, y_pred)


# ----------------------------------------------------
# Evaluate
# ----------------------------------------------------

def evaluate_model(y_true, y_pred):
    """
    Đánh giá model.

    Returns
    -------
    dict
    """

    metrics = {

        "MSE": calculate_mse(
            y_true,
            y_pred
        ),

        "RMSE": calculate_rmse(
            y_true,
            y_pred
        ),

        "MAE": calculate_mae(
            y_true,
            y_pred
        ),

        "R2": calculate_r2(
            y_true,
            y_pred
        )

    }

    return metrics


# ----------------------------------------------------
# Print Metric
# ----------------------------------------------------

def print_metrics(metrics):
    """
    In metric đẹp ra terminal.
    """

    print("=" * 40)

    print("Evaluation Result")

    print("=" * 40)

    print(f"MSE  : {metrics['MSE']:.6f}")
    print(f"RMSE : {metrics['RMSE']:.6f}")
    print(f"MAE  : {metrics['MAE']:.6f}")
    print(f"R²   : {metrics['R2']:.6f}")

    print("=" * 40)


# ----------------------------------------------------
# Compare Models
# ----------------------------------------------------

def compare_models(results):
    """
    So sánh nhiều model.

    Parameters
    ----------
    results : dict

    Example

    {
        "Linear": {...},
        "Ridge": {...},
        "Lasso": {...}
    }
    """

    table = pd.DataFrame(results).T

    table = table[
        [
            "MSE",
            "RMSE",
            "MAE",
            "R2"
        ]
    ]

    return table


# ----------------------------------------------------
# Save Result
# ----------------------------------------------------

def save_results(
    table,
    filename="results/model_comparison.csv"
):
    """
    Lưu bảng so sánh.
    """

    import os

    os.makedirs(
        "results",
        exist_ok=True
    )

    table.to_csv(
        filename,
        index=True
    )

    print()

    print("Saved to")

    print(filename)


# ----------------------------------------------------
# Example
# ----------------------------------------------------

if __name__ == "__main__":

    y_true = np.array(
        [
            3,
            5,
            2,
            7,
            9
        ]
    )

    y_pred = np.array(
        [
            2.9,
            4.8,
            2.2,
            6.7,
            9.1
        ]
    )

    metrics = evaluate_model(
        y_true,
        y_pred
    )

    print_metrics(metrics)

    models = {

        "Linear Regression": metrics,

        "Dummy Model": {

            "MSE": 2.5,
            "RMSE": 1.58,
            "MAE": 1.31,
            "R2": 0.65

        }

    }

    table = compare_models(models)

    print()

    print(table)

    save_results(table)

if __name__ == "__main__":

    y_true = ...
    y_pred = ...

    metrics = evaluate_model(y_true, y_pred)

    print_metrics(metrics)