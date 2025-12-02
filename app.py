import streamlit as st
import joblib
import pandas as pd

# Load the trained pipeline (preprocessing + model)
model = joblib.load("modelo_seguros_pipeline_v2.pkl")

st.title("Health Insurance Cost Calculator")

# User inputs for the features used during training
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker?", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

if st.button("Calculate Price"):
    # Build a single-row DataFrame with the input values
    data = pd.DataFrame([[
        age, sex, bmi, children, smoker, region
    ]], columns=["age", "sex", "bmi", "children", "smoker", "region"])

    # The pipeline handles preprocessing internally
    prediction = model.predict(data)[0]

    st.success(f"Estimated Cost: ${prediction:,.2f}")

st.caption("Model trained with Random Forest using a complete pipeline and GridSearchCV.")