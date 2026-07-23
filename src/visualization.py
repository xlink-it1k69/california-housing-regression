"""
visualization.py
================

Module chịu trách nhiệm trực quan hóa dữ liệu.

Tất cả hình sẽ được lưu vào thư mục:

figures/

Notebook chỉ cần gọi các hàm trong file này.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------------
# Tạo thư mục figures nếu chưa tồn tại
# ------------------------------------------------------------------

FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Hàm lưu hình
# ------------------------------------------------------------------

def save_figure(filename: str):
    """
    Lưu hình vào thư mục figures.
    """

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------------
# Histogram
# ------------------------------------------------------------------

def plot_histograms(df: pd.DataFrame):
    """
    Vẽ histogram cho tất cả feature.
    """

    df.hist(
        figsize=(15, 10),
        bins=30
    )

    save_figure("histograms.png")


# ------------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame):
    """
    Heatmap hệ số tương quan.
    """

    plt.figure(figsize=(10, 8))

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    save_figure("heatmap.png")


# ------------------------------------------------------------------
# Boxplot
# ------------------------------------------------------------------

def plot_boxplots(df: pd.DataFrame):
    """
    Boxplot của toàn bộ feature.
    """

    plt.figure(figsize=(15, 8))

    sns.boxplot(data=df)

    plt.xticks(rotation=45)

    plt.title("Boxplots")

    save_figure("boxplots.png")


# ------------------------------------------------------------------
# Scatter Plot
# ------------------------------------------------------------------

def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str
):
    """
    Scatter plot giữa hai cột.
    """

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x=x,
        y=y
    )

    plt.title(f"{x} vs {y}")

    save_figure(f"scatter_{x}_{y}.png")


# ------------------------------------------------------------------
# Prediction vs Actual
# ------------------------------------------------------------------

def plot_prediction_vs_actual(
    y_true,
    y_pred
):
    """
    So sánh giá trị dự đoán và giá trị thật.
    """

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=y_true,
        y=y_pred
    )

    plt.xlabel("Actual")

    plt.ylabel("Prediction")

    plt.title("Prediction vs Actual")

    save_figure("prediction_vs_actual.png")


# ------------------------------------------------------------------
# Residual Plot
# ------------------------------------------------------------------

def plot_residuals(
    y_true,
    y_pred
):
    """
    Residual Plot.
    """

    residual = y_true - y_pred

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=y_pred,
        y=residual
    )

    plt.axhline(
        0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Prediction")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    save_figure("residual_plot.png")


# ------------------------------------------------------------------
# Coefficient Plot
# ------------------------------------------------------------------

def plot_coefficients(
    feature_names,
    coefficients
):
    """
    Biểu đồ hệ số của Linear Regression.
    """

    coef = pd.Series(
        coefficients,
        index=feature_names
    )

    coef.sort_values().plot(
        kind="barh",
        figsize=(8, 6)
    )

    plt.title("Feature Coefficients")

    save_figure("coefficients.png")


# ------------------------------------------------------------------
# Learning Curve
# ------------------------------------------------------------------

def plot_learning_curve(
    losses
):
    """
    Learning Curve của Gradient Descent.
    """

    plt.figure(figsize=(8, 5))

    plt.plot(losses)

    plt.xlabel("Iteration")

    plt.ylabel("Loss")

    plt.title("Learning Curve")

    save_figure("learning_curve.png")

if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()

    plot_histograms(df)
    plot_heatmap(df)
    plot_boxplots(df)

    print("Done!")