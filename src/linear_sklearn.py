"""
linear_sklearn.py

Linear Regression sử dụng scikit-learn.
"""

from sklearn.linear_model import LinearRegression


class SklearnLinearRegression:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        """
        Huấn luyện mô hình.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Dự đoán.
        """
        return self.model.predict(X)

    @property
    def coefficients(self):
        return self.model.coef_

    @property
    def intercept(self):
        return self.model.intercept_


if __name__ == "__main__":

    from data_loader import load_data
    from preprocessing import prepare_data
    from evaluate import regression_report

    df = load_data()

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    model = SklearnLinearRegression()

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    regression_report(y_test, prediction)

    print("\nCoefficients")
    print(model.coefficients)

    print("\nIntercept")
    print(model.intercept)