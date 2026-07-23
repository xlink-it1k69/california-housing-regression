# 🏡 California Housing Price Prediction

A complete Machine Learning project for predicting California housing prices using Linear Regression, Ridge Regression and Lasso Regression.

---

## 📌 Project Overview

This project aims to predict the median house value in California districts using the California Housing Dataset provided by scikit-learn.

The project covers the complete Machine Learning workflow:

- Data Loading
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Linear Regression from Scratch (NumPy)
- Linear Regression using Scikit-learn
- Ridge Regression
- Lasso Regression
- Model Evaluation
- Data Visualization

---

## 📂 Project Structure

```text
california-housing-regression/
│
├── data/
├── figures/
├── models/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── evaluate.py
│   ├── linear_numpy.py
│   ├── linear_sklearn.py
│   └── model_compare.py
│
├── train.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 📊 Dataset

Dataset:

California Housing Dataset

Source:

Scikit-learn

Number of samples:

20,640

Features:

8 numerical features

Target:

Median House Value

---

## ⚙️ Installation

Clone repository

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

## 🚀 Usage

Train all models

```bash
python train.py
```

---

## 📈 Models

- Linear Regression (NumPy)
- Linear Regression (Scikit-learn)
- Ridge Regression
- Lasso Regression

---

## 📉 Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📸 Visualizations

The project generates:

- Histogram
- Correlation Heatmap
- Boxplot
- Scatter Plot
- Residual Plot
- Prediction vs Actual
- Coefficient Comparison
- Learning Curve

---

## 📝 Results

| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| Linear Regression | - | - | - |
| Ridge Regression | - | - | - |
| Lasso Regression | - | - | - |

(These values will be updated after training.)

---

## 🔮 Future Work

- Polynomial Regression
- Elastic Net
- Random Forest
- XGBoost
- Hyperparameter Optimization
- Streamlit Deployment

---

## 👤 Author

Dinh Xuan Linh

Second-year Computer Science Student

Hanoi University of Science and Technology (HUST)

---

## 📄 License

This project is licensed under the MIT License.