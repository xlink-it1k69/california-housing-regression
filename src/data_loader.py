"""
data_loader.py
==============

Module chịu trách nhiệm tải California Housing Dataset.

Không thực hiện:
- Tiền xử lý dữ liệu
- Chia train/test
- Visualization
- Huấn luyện mô hình

Chỉ có nhiệm vụ:
1. Tải dataset từ scikit-learn
2. Trả về pandas DataFrame
"""

from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_data() -> pd.DataFrame:
    """
    Load California Housing Dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame gồm:
        - 8 feature
        - 1 target (MedHouseVal)
    """

    housing = fetch_california_housing(as_frame=True)

    df = housing.frame.copy()

    return df


def get_feature_names() -> list[str]:
    """
    Trả về danh sách tên các feature.

    Returns
    -------
    list[str]
    """

    housing = fetch_california_housing()

    return housing.feature_names


def get_target_name() -> str:
    """
    Trả về tên biến mục tiêu.

    Returns
    -------
    str
    """

    return "MedHouseVal"


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    In thông tin cơ bản của dataset.

    Parameters
    ----------
    df : pd.DataFrame
    """

    print("=" * 60)
    print("California Housing Dataset")
    print("=" * 60)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 5 Rows:")
    print(df.head())


if __name__ == "__main__":

    df = load_data()

    print_dataset_info(df)