# Medical Insurance Cost Prediction

## Project Overview
Predicting medical insurance costs is crucial for helping insurance companies optimize premiums and helping patients understand their potential expenses. This project implements an end-to-end Machine Learning pipeline to predict individual medical costs based on demographic and lifestyle data.

**Key Insight:** The analysis revealed that smoking, especially when combined with a BMI > 30 (Obesity), creates a synergistic effect that drastically increases medical costs.

## Live Demo
*(Aquí pondremos el link si decidimos subirlo a la nube después, por ahora bórralo)*

## Technologies Used
* **Python** (Pandas, NumPy, Seaborn, Matplotlib)
* **Scikit-Learn** (Pipeline, ColumnTransformer, RandomForest, GridSearchCV)
* **Streamlit** (Web App Interface)
* **Joblib** (Model persistence)

## Methodology
1.  **EDA:** Analyzed correlations (Smoker & BMI interaction).
2.  **Preprocessing:** Built a robust pipeline with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
3.  **Modeling:** Implemented `RandomForestRegressor` wrapped in a `TransformedTargetRegressor` (Log-Scale) to handle skewed cost distribution.
4.  **Validation:** Achieved an RMSE of **$4,345** on the Test Set using Stratified K-Fold Cross Validation.

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/TU_USUARIO/medical-insurance-cost-prediction.git](https://github.com/TU_USUARIO/medical-insurance-cost-prediction.git)