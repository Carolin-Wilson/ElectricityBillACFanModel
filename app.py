import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the trained model pipeline
@st.cache_resource
def load_model():
    with open("electric_bill_model_acfan.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Streamlit UI Configuration
st.set_page_config(
    page_title="Electric Bill Predictor", page_icon="⚡", layout="centered"
)

st.title("⚡ Electric Bill Prediction App")
st.write(
    "Enter the electricity consumption units for AC and Fans below to predict your estimated Electric Bill using a Polynomial Regression model."
)

# Sidebar inputs
st.sidebar.header("Input Parameters")
ac_units = st.sidebar.number_input(
    "AC Units Consumed", min_value=0.0, max_value=500.0, value=60.0, step=5.0
)
fan_units = st.sidebar.number_input(
    "Fan Units Consumed", min_value=0.0, max_value=500.0, value=70.0, step=5.0
)

# Main panel prediction display
if st.button("Predict Electric Bill", type="primary"):
    input_data = pd.DataFrame(
        [[ac_units, fan_units]], columns=["AC_Units", "Fan_Units"]
    )
    prediction = model.predict(input_data)[0]

    st.success("Prediction Complete!")
    st.metric(
        label="Estimated Electric Bill", value=f"₹ {prediction:,.2f}"
    )

    st.info(
        f"Prediction based on **{ac_units} AC Units** and **{fan_units} Fan Units**."
    )

# Footer
st.markdown("---")
st.caption("Christ College of Engineering | Data Science & Machine Learning Lab")
