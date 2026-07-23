"""
linear_numpy.py
===============

Linear Regression from Scratch using NumPy.

Không sử dụng:
    sklearn.linear_model.LinearRegression

Có sử dụng:
    NumPy

Tác giả: Xuân Linh
"""

import numpy as np


class LinearRegressionGD:
    """
    Linear Regression sử dụng Gradient Descent.
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=1000,
        random_state=42
    ):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.weights = None
        self.bias = None

        self.loss_history = []

    # --------------------------------------------------

    def initialize_parameters(self, n_features):
        """
        Khởi tạo weight và bias.
        """

        np.random.seed(self.random_state)

        self.weights = np.zeros(n_features)

        self.bias = 0.0

    # --------------------------------------------------

    def predict(self, X):
        """
        Dự đoán.
        """

        return np.dot(X, self.weights) + self.bias

    # --------------------------------------------------

    def compute_loss(self, y_true, y_pred):
        """
        Mean Squared Error.
        """

        return np.mean((y_true - y_pred) ** 2)

    # --------------------------------------------------

    def fit(self, X, y):
        """
        Huấn luyện Gradient Descent.
        """

        n_samples, n_features = X.shape

        self.initialize_parameters(n_features)

        self.loss_history = []

        for epoch in range(self.epochs):

            predictions = self.predict(X)

            error = predictions - y

            dw = (2 / n_samples) * np.dot(X.T, error)

            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw

            self.bias -= self.learning_rate * db

            loss = self.compute_loss(y, predictions)

            self.loss_history.append(loss)

            if (epoch + 1) % 100 == 0:

                print(
                    f"Epoch {epoch+1:4d}/{self.epochs} "
                    f"Loss = {loss:.6f}"
                )

        return self

    # --------------------------------------------------

    def score(self, X, y):
        """
        Tính R².
        """

        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)

        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - ss_res / ss_tot
    
if __name__ == "__main__":

    from preprocessing import preprocess_data

    from evaluate import (
        evaluate_model,
        print_metrics
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_data()

    model = LinearRegressionGD(
        learning_rate=0.01,
        epochs=1000
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = evaluate_model(
        y_test,
        predictions
    )

    print()

    print_metrics(metrics)