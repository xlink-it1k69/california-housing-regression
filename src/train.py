"""
train.py

Huấn luyện mô hình Linear Regression viết bằng NumPy.
"""

import os
import pickle

from data_loader import load_data
from preprocessing import prepare_data
from linear_numpy import LinearRegressionGD
from evaluate import regression_report


def save_model(model, path="../models/linear_numpy.pkl"):
    """
    Lưu model bằng pickle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)


def main():
    print("=" * 60)
    print("California Housing Regression")
    print("=" * 60)

    # ---------------------------------------------------
    # Load data
    # ---------------------------------------------------
    print("\nLoading dataset...")

    df = load_data()

    print(df.head())

    # ---------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------
    print("\nPreparing data...")

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------
    print("\nTraining model...")

    model = LinearRegressionGD(
        learning_rate=0.01,
        epochs=2000
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------
    # Predict
    # ---------------------------------------------------
    y_pred = model.predict(X_test)

    # ---------------------------------------------------
    # Evaluate
    # ---------------------------------------------------
    print("\nEvaluation")

    regression_report(
        y_test,
        y_pred
    )

    # ---------------------------------------------------
    # Save
    # ---------------------------------------------------
    save_model(model)

    print("\nModel saved to models/linear_numpy.pkl")

    print("\nDone!")


if __name__ == "__main__":
    main()