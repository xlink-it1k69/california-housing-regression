"""
model_compare.py

So sánh Linear Regression tự cài đặt và sklearn.
"""

import pandas as pd

from data_loader import load_data
from preprocessing import prepare_data

from linear_numpy import LinearRegressionGD
from linear_sklearn import SklearnLinearRegression

from evaluate import (
    mean_absolute_error_numpy,
    root_mean_squared_error,
    r2_score_numpy
)


def evaluate_model(name, model, X_test, y_test):

    prediction = model.predict(X_test)

    mae = mean_absolute_error_numpy(y_test, prediction)
    rmse = root_mean_squared_error(y_test, prediction)
    r2 = r2_score_numpy(y_test, prediction)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def main():

    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    df = load_data()

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    #######################################
    # NumPy
    #######################################

    numpy_model = LinearRegressionGD(
        learning_rate=0.01,
        epochs=2000
    )

    numpy_model.fit(X_train, y_train)

    #######################################
    # Sklearn
    #######################################

    sklearn_model = SklearnLinearRegression()

    sklearn_model.fit(X_train, y_train)

    #######################################
    # Evaluate
    #######################################

    results = []

    results.append(
        evaluate_model(
            "NumPy Gradient Descent",
            numpy_model,
            X_test,
            y_test
        )
    )

    results.append(
        evaluate_model(
            "Sklearn LinearRegression",
            sklearn_model,
            X_test,
            y_test
        )
    )

    results = pd.DataFrame(results)

    print("\nComparison Result\n")

    print(results)

    print("\nBest model (RMSE)")

    best = results.sort_values("RMSE").iloc[0]

    print(best)


if __name__ == "__main__":
    main()