# 🏥 Medical Insurance Cost Prediction

A complete end-to-end Machine Learning project that predicts an individual's medical insurance charges based on personal attributes such as age, BMI, smoking status, number of children, sex, and region — built using **Python, Pandas, Scikit-learn, and Linear Regression**.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

Medical insurance costs vary significantly between individuals based on their lifestyle and health-related attributes. This project builds a **regression model** that estimates a person's expected insurance charges, and wraps it in a simple interactive application for real-time predictions.

**Problem type:** Supervised Learning — Regression
**Target variable:** `charges` (medical insurance cost in USD)

---

## 📂 Dataset

- **Source:** [Medical Cost Personal Datasets — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Format:** CSV
- **Records:** 1,338 (1,337 after removing 1 duplicate)
- **Features:**

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `sex` | Gender (male / female) |
| `bmi` | Body Mass Index |
| `children` | Number of dependents |
| `smoker` | Smoking status (yes / no) |
| `region` | Residential region in the US |
| `charges` | **Target** — medical insurance cost billed |

---

## 🧹 Data Cleaning & Preprocessing

- Checked for missing values → **none found**
- Checked for duplicate records → **1 duplicate found and removed**
- Encoded categorical variables:
  - `sex`, `smoker` → Label encoding (binary: 0/1)
  - `region` → One-hot encoding (4 categories → 3 dummy columns)
- Verified final dataset was fully numeric and ready for model training

---

## 📊 Exploratory Data Analysis (EDA)

Key visualizations created: distribution of charges, age vs. charges scatter plot, smoker vs. charges boxplot, and a correlation heatmap.

**Key findings:**
- **`smoker` has the strongest correlation with charges (≈ 0.79)** — smokers pay ~4x more on average ($32,050 vs $8,440)
- `age` shows a moderate positive correlation (≈ 0.30)
- `bmi` shows a weak-to-moderate positive correlation (≈ 0.20)
- `sex` and `region` show minimal correlation with charges

---

## 🤖 Model

- **Algorithm:** Multiple Linear Regression (`scikit-learn`)
- **Train/test split:** 80% training / 20% testing (`random_state=42`)
- **Features used:** age, sex, bmi, children, smoker, region (encoded)

---

## 📈 Evaluation Results

| Metric | Value |
|---|---|
| MAE (Mean Absolute Error) | ≈ $4,177 |
| MSE (Mean Squared Error) | ≈ 35,478,020 |
| RMSE (Root Mean Squared Error) | ≈ $5,956 |
| R² Score | ≈ 0.807 (80.7%) |

The model explains approximately **80.7% of the variance** in insurance charges using the available features.

---

## 🖥️ Prediction Application

A simple interactive application (built with `ipywidgets` in Jupyter, and as a standalone Streamlit app for deployment) allows a user to enter:
- Age, Sex, BMI, Number of Children, Smoking Status, Region

...and instantly receive a predicted insurance cost.

**Example predictions:**
| Profile | Predicted Cost |
|---|---|
| Age 25, female, BMI 22.5, no children, non-smoker, northwest | $1,891.64 |
| Age 50, male, BMI 30.0, 2 children, smoker, southeast | $34,082.25 |

🔗 **Live app:** [https://medical-insurance-cost-prediction-uuft3ryc9nabu8b2kxf6zf.streamlit.app]

---

## 🗂️ Project Structure

```
medical-insurance-project/
├── data/
│   ├── insurance.csv               # Raw dataset
│   └── insurance_cleaned.csv       # Cleaned, encoded dataset
├── notebooks/
│   ├── phase2_dataset_exploration.ipynb
│   ├── phase3_data_cleaning.ipynb
│   ├── phase4_eda.ipynb
│   ├── phase5_linear_regression.ipynb
│   ├── phase6_model_evaluation.ipynb
│   └── phase7_prediction_app.ipynb
├── app.py                          # Standalone Streamlit prediction app
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run This Project Locally

```bash
# Clone the repository
git clone <your-repo-link>
cd medical-insurance-project

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## ⚠️ Limitations

- Linear Regression assumes a linear relationship between features and charges; more complex models (Random Forest, Gradient Boosting) could improve accuracy
- The dataset does not include health-related features like pre-existing conditions, which likely explains part of the unexplained variance
- Sample size (1,337 records) is relatively small for production-grade deployment

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `ipywidgets` · `Streamlit`

---

## 👤 Author

Built as part of a Machine Learning Internship project — Phases 2 through 8, covering the complete ML workflow from data collection to deployment.
