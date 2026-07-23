# California Housing Price Prediction

Machine Learning project using the California Housing dataset.

---

## Project Objectives

- Perform Exploratory Data Analysis (EDA)
- Build Linear Regression from scratch using NumPy
- Compare with Scikit-learn implementation
- Evaluate model performance
- Visualize results

---

## Dataset

Dataset: California Housing

Loaded directly from Scikit-learn:

```python
from sklearn.datasets import fetch_california_housing
```

Number of samples:

20640

Features:

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

Target:

Median House Value

---

## Project Structure

```text
california-housing-regression
│
├── data
├── figures
├── models
├── notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_LinearRegression_FromScratch.ipynb
│   ├── 04_Sklearn.ipynb
│   └── 05_ModelComparison.ipynb
│
├── src
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── evaluate.py
│   ├── linear_numpy.py
│   ├── linear_sklearn.py
│   ├── model_compare.py
│   └── train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/xlink-it1k69/california-housing-regression.git
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Install packages

```bash
pip install -r requirements.txt
```

---

## Run

Train model

```bash
python src/train.py
```

---

## Models

- Linear Regression (NumPy)
- Linear Regression (Scikit-learn)

Future

- Ridge Regression
- Lasso Regression

---

## Evaluation Metrics

- MAE
- MSE
- RMSE
- R² Score

---

## Visualizations

- Histogram
- Heatmap
- Boxplot
- Prediction vs Actual
- Residual Plot
- Learning Curve

---

## Author

Xuân Linh Đinh

Computer Science Student

Hanoi University of Science and Technology (HUST)