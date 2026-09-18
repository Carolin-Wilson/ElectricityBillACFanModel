import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Electric Bill Predictor", page_icon="⚡", layout="centered"
)

# Load the saved model pipeline (.pkl)
@st.cache_resource
def load_model():
    return joblib.load("electric_bill_model_acfan.pkl")


model = load_model()

# App UI Header
st.title("⚡ Electric Bill Predictor")
st.write(
    "Enter the number of **AC Units** and **Fan Units** consumed to predict your estimated electricity bill using a trained Polynomial Regression model."
)

# User input form
with st.form("prediction_form"):
    ac_units = st.number_input(
        "AC Units Consumed",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
        help="Enter the total units consumed by your AC.",
    )

    fan_units = st.number_input(
        "Fan Units Consumed",
        min_value=0.0,
        max_value=500.0,
        value=30.0,
        step=1.0,
        help="Enter the total units consumed by your Fans.",
    )

    submit_button = st.form_submit_button(label="Predict Bill")

# Prediction logic
if submit_button:
    # Format input into a DataFrame matching model expectation (both features required)
    input_data = pd.DataFrame(
        [[ac_units, fan_units]], columns=["AC_Units", "Fan_Units"]
    )

    # Predict using the loaded model
    predicted_bill = model.predict(input_data)[0]

    # Display result
    st.success(f"### Estimated Electric Bill: ₹ {predicted_bill:,.2f}")
    st.info(
        f"Based on **{ac_units}** AC units and **{fan_units}** Fan units consumed."
    )

# Footer info
st.markdown("---")
st.caption(
    "Developed with Streamlit & Scikit-Learn | Christ College Project"
)
