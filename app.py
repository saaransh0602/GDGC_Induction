import streamlit as st
import pandas as pd
import joblib

model = joblib.load("pipeline_model.joblib")
info = joblib.load("feature_info.joblib")

num_cols = info["numerical_features"]
cat_cols = info["categorical_features"]
all_cols = num_cols + cat_cols

st.title("GDGC Induction")

st.write("Enter values for each feature below")

user_input = {}

st.header("Numerical Features")
for col in num_cols:
    user_input[col] = st.number_input(col, value=0.0)

st.header("Categorical Features")
for col in cat_cols:
    user_input[col] = st.text_input(col, value="")

if st.button("Predict"):
    df = pd.DataFrame([user_input], columns=all_cols)

    prediction = model.predict(df)[0]

    st.subheader("Prediction")
    st.success(round(prediction, 4))
